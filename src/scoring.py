"""Game scoring from the venue scoreboard: who scored when, and a YouTube text.

Pipeline (stage 1, no player identification yet):

1. Cut the calibrated scoreboard out of every keyframe (``ffmpeg -skip_frame
   nokey``: ~1 frame/s on typical 360 cameras, cheap to decode), following
   camera bumps (``camera_shifts`` from calibration).
2. Find the moments the board changes, locally: per-pixel step detector on
   the LED digits, ignoring the always-changing game clock.  Between two
   changes the board is constant -> one "state" tile (median of a few crops).
3. Read the states with a vision LLM: up to ``tiles_per_request`` states per
   request, one small numbered image each, chronological, with the previous
   reading as context (scores only rise by 1-3).  ~10 requests for a game.
4. Clean the readings (single misreads, unreadable tiles) and turn every
   score change into an event ``{time, team, points, score}``.
5. Write ``scoring.json``, ``scoring.csv`` and ``youtube.txt`` (chapters per
   period + list of baskets with clickable timestamps).

Team names come from ``match`` in the per-video calibration file
(``{"home": "...", "guest": "..."}``, set on the dashboard calibration page).
The scoreboard's left score is the home team unless ``match.home_side`` is
``"right"``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .calibration import _jpeg_b64, pitch_to_y, shift_at, yaw_to_x
from .jerseys import jerseys_config
from .shots import apply_shots, find_shots, shots_config

logger = logging.getLogger("soccer360.scoring")

SIDES = ("home", "guest")


def scoring_config(config: dict) -> dict:
    sc = dict(config.get("scoring", {}) or {})
    sc.setdefault("enabled", False)
    sc.setdefault("led", "red")               # red LED digits | bright (any colour)
    sc.setdefault("change_threshold", 20.0)   # pixel step for a board change
    sc.setdefault("tiles_per_request", 20)    # board states per LLM request
    sc.setdefault("crop_margin", 0.15)        # extra around the board (fraction)
    sc.setdefault("tile_scale", 3.0)          # upscale crops sent to the LLM
    sc.setdefault("lead_sec", 6.0)            # timestamp = board update - lead
    sc.setdefault("max_points", 3)            # bigger jumps are flagged
    sc.setdefault("chapters", True)           # period chapters in youtube.txt
    llm = dict(config.get("calibration", {}).get("llm", {}) or {})
    llm.update(sc.get("llm", {}) or {})
    llm.setdefault("provider", "openai")
    llm.setdefault("model", "gpt-5")
    llm.setdefault("api_key_env", "OPENAI_API_KEY")
    llm.setdefault("timeout_sec", 180)
    sc["llm"] = llm
    return sc


def match_info(cal: dict | None) -> dict:
    m = dict((cal or {}).get("match") or {})
    m["home"] = (m.get("home") or "").strip() or "Gospodarze"
    m["guest"] = (m.get("guest") or "").strip() or "Goście"
    m["home_side"] = "right" if m.get("home_side") == "right" else "left"
    m["roster"] = m.get("roster") or {}
    return m


def player_label(e: dict, match: dict, uncertain: bool = False) -> str:
    """"#7 Kowalski" for an event's shooter ("" when unknown)."""
    sh = e.get("shot") or {}
    n = sh.get("number")
    if n is None or (sh.get("number_src") == "read_low" and not uncertain):
        return ""
    name = ((match.get("roster") or {}).get(e["team"]) or {}).get(str(n), "")
    return f"#{n} {name}".strip()


# ---------------------------------------------------------------------------
# 1. Scoreboard crops
# ---------------------------------------------------------------------------

def _board_px(sb: dict, w: int, h: int, margin: float, dyaw=0.0, dpitch=0.0):
    """(x0, y0, x1, y1) of the scoreboard (+margin) in a w x h equirect frame."""
    hw = float(sb["width_deg"]) * (1 + margin) / 2
    hh = float(sb["height_deg"]) * (1 + margin) / 2
    yaw, pitch = float(sb["yaw"]) + dyaw, float(sb["pitch"]) + dpitch
    x0, x1 = yaw_to_x(yaw - hw, w), yaw_to_x(yaw + hw, w)
    y0, y1 = pitch_to_y(pitch + hh, h), pitch_to_y(pitch - hh, h)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def extract_board_crops(
    video_path: str | Path, sb: dict, shifts: list[dict] | None,
    width: int, height: int, margin: float = 0.15, min_gap_sec: float = 0.9,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Scoreboard crops from keyframes: (times[s], [RGB crop, ...]).

    One ffmpeg pass decodes keyframes only and cuts a region that covers the
    board at every camera shift; each sample is then trimmed to the board
    position in effect at its time.
    """
    offs = [(0.0, 0.0)] + [(s["dyaw"], s["dpitch"]) for s in shifts or []]
    boxes = [_board_px(sb, width, height, margin, dy, dp) for dy, dp in offs]
    X0 = max(0, min(b[0] for b in boxes))
    Y0 = max(0, min(b[1] for b in boxes))
    X1 = min(width, max(b[2] for b in boxes))
    Y1 = min(height, max(b[3] for b in boxes))
    if X1 - X0 < 8 or Y1 - Y0 < 8 or any(b[2] <= b[0] for b in boxes):
        raise RuntimeError("Scoreboard region invalid (across the 360 seam?)")
    # chroma-subsampled video: ffmpeg crops on even offsets/sizes
    X0, Y0 = X0 - X0 % 2, Y0 - Y0 % 2
    X1, Y1 = min(width, X1 + (X1 - X0) % 2), min(height, Y1 + (Y1 - Y0) % 2)
    cw, ch = (X1 - X0) // 2 * 2, (Y1 - Y0) // 2 * 2
    vf = (f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{min_gap_sec})',"
          f"crop={cw}:{ch}:{X0}:{Y0},showinfo")
    cmd = [
        "ffmpeg", "-v", "info", "-nostats", "-skip_frame", "nokey",
        "-i", str(video_path), "-an", "-vf", vf, "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    logger.info("Extracting scoreboard crops (%dx%d at %d,%d) ...", cw, ch, X0, Y0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import threading

    err_lines: list[str] = []
    t_err = threading.Thread(
        target=lambda: err_lines.extend(proc.stderr.read().decode(errors="replace").splitlines()),
        daemon=True,
    )
    t_err.start()
    fsize = cw * ch * 3
    frames = []
    while True:
        buf = proc.stdout.read(fsize)
        if not buf or len(buf) < fsize:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(ch, cw, 3))
    proc.wait()
    t_err.join()
    times = [float(m.group(1)) for line in err_lines
             if (m := re.search(r"pts_time:\s*([-\d.]+)", line))]
    if proc.returncode != 0 and not frames:
        raise RuntimeError("ffmpeg failed: " + "\n".join(err_lines[-5:]))
    n = min(len(times), len(frames))
    times_arr = np.asarray(times[:n], float)
    crops = []
    for t, fr in zip(times_arr, frames[:n]):
        dy, dp = shift_at(shifts, float(t))
        x0, y0, x1, y1 = _board_px(sb, width, height, margin, dy, dp)
        crops.append(fr[max(0, y0 - Y0):y1 - Y0, max(0, x0 - X0):x1 - X0].copy())
    logger.info("Scoreboard crops: %d (%.0f-%.0fs)", n,
                times_arr[0] if n else 0, times_arr[-1] if n else 0)
    return times_arr, crops


# ---------------------------------------------------------------------------
# 2. When does the board change? (local, no LLM)
# ---------------------------------------------------------------------------

def _led_channel(crop: np.ndarray, size: tuple[int, int], led: str) -> np.ndarray:
    c = cv2.resize(crop, size).astype(np.int16)
    if led == "red":   # red LED digits: red minus the other channels
        return np.clip(c[..., 0] - (c[..., 1] + c[..., 2]) // 2, 0, 255)
    return cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.int16)


def detect_changes(
    crops: list[np.ndarray], *, led: str = "red", window: int = 4,
    threshold: float = 20.0, margin: float = 0.15, busy_frac: float = 0.03,
    min_gap: int = 5, chunk: int = 500,
) -> list[int]:
    """Indices where the board content changes persistently (score, fouls, ...).

    Per pixel: |median of the next ``window`` samples - median of the previous
    ``window``| (robust to single-frame noise / occlusion).  Only LED pixels
    inside the board count; pixels that change all the time (the running
    clock) are ignored.  A change = mean of the strongest 2% pixel steps
    above ``threshold``, local maximum within ``min_gap`` samples.  Over-
    detection is harmless (the reader sees an unchanged score).
    """
    from numpy.lib.stride_tricks import sliding_window_view as swv

    n, k = len(crops), window
    if n < 2 * k + 1:
        return []
    h, w = crops[0].shape[:2]
    g = np.stack([_led_channel(c, (w, h), led) for c in crops])
    step = np.zeros((n, h, w), np.uint8)
    for a in range(k, n - k + 1, chunk):
        b = min(a + chunk, n - k + 1)
        before = np.median(swv(g[a - k:b - 1], k, axis=0), axis=-1)
        after = np.median(swv(g[a:b + k - 1], k, axis=0), axis=-1)
        step[a:b] = np.clip(np.abs(after - before), 0, 255).astype(np.uint8)
    lit = np.percentile(g[:: max(1, n // 1000)], 90, axis=0) > 40
    busy = (step > threshold).mean(0) >= busy_frac
    m = margin / (1 + margin) / 2          # crop margin -> board only
    inner = np.zeros((h, w), bool)
    inner[int(m * h):h - int(m * h), int(m * w):w - int(m * w)] = True
    mask = lit & inner & ~busy
    if mask.sum() < 10:
        logger.warning("Scoreboard change detection: no stable LED pixels found")
        return []
    vals = step[:, mask].astype(np.float32)
    top = max(8, int(0.02 * mask.sum()))
    sig = -np.sort(-vals, axis=1)[:, :top].mean(1)
    peaks: list[int] = []
    for i in sorted(np.flatnonzero(sig >= threshold), key=lambda j: (-sig[j], j)):
        a = b = int(i)  # flat top (same value): take its middle
        while a > 0 and sig[a - 1] == sig[i]:
            a -= 1
        while b < n - 1 and sig[b + 1] == sig[i]:
            b += 1
        c = (a + b) // 2
        if c > 0 and all(abs(c - p) > min_gap for p in peaks):
            peaks.append(c)
    peaks.sort()
    logger.info("Scoreboard: %d candidate changes (%d LED px, %d clock px ignored)",
                len(peaks), int(mask.sum()), int((lit & busy).sum()))
    return peaks


def segment_states(
    times: np.ndarray, crops: list[np.ndarray], changes: list[int], max_avg: int = 5,
) -> list[dict]:
    """Constant-board segments between changes: ``{"t", "i0", "i1", "tile"}``.

    ``tile`` = pixel median of up to ``max_avg`` samples from the middle of the
    segment (less noise, no passers-by).
    """
    bounds = [0] + [c for c in changes if 0 < c < len(crops)] + [len(crops)]
    h, w = crops[0].shape[:2]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        mid = (a + b) // 2
        lo = max(a, mid - max_avg // 2)
        pick = [cv2.resize(c, (w, h)) for c in crops[lo:min(b, lo + max_avg)]]
        tile = np.median(np.stack(pick), axis=0).astype(np.uint8)
        out.append({"t": round(float(times[a]), 2), "i0": a, "i1": b, "tile": tile})
    return out


# ---------------------------------------------------------------------------
# 3. Reading the states (vision LLM, numbered images, chronological)
# ---------------------------------------------------------------------------

_READ_PROMPT = """The numbered tiles (#0, #1, ...) show the SAME electronic basketball scoreboard (LED 7-segment digits, filmed from far away, blurry) in CHRONOLOGICAL order, left to right, row by row. Each tile is a stretch of the game during which the board did not change; between neighbouring tiles something changed: usually one team's score went up by 1, 2 or 3, but it may also be only fouls, timeouts, the period, or nothing relevant.

Top row of the board: the LEFT team score, the period (small, middle), the RIGHT team score. Ignore the game clock, fouls and timeouts.
{context}
A score usually rises by 1, 2 or 3 between neighbouring images; the table operator sometimes corrects a mistake, so a score CAN go down - always report what the board shows, do not \"fix\" it. Compare neighbouring images digit by digit to decide blurry digits (5 vs 6 vs 8, 1 vs 7, 3 vs 8). Use null only when the board is off, occluded or unreadable.

Respond with JSON only:
{{"tiles": [{{"i": 0, "left": int|null, "right": int|null, "period": int|null}}, ...]}}"""


def read_prompt(prev: dict | None) -> str:
    ctx = ""
    if prev and prev.get("left") is not None and prev.get("right") is not None:
        ctx = (f"\nJust before tile #0 the board showed: left {prev['left']}, "
               f"right {prev['right']}, period {prev.get('period') or '?'}.\n")
    return _READ_PROMPT.format(context=ctx)


class OpenAIScoreReader:
    """Reads numbered scoreboard crops with an OpenAI vision model."""

    def __init__(self, llm_cfg: dict):
        from openai import OpenAI  # optional dependency

        env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"{env} is not set")
        self.model = llm_cfg.get("model", "gpt-5")
        self.effort = llm_cfg.get("reasoning_effort")
        self.client = OpenAI(api_key=key, timeout=float(llm_cfg.get("timeout_sec", 180)))
        self.calls = 0

    def __call__(self, tiles: list[np.ndarray], prompt: str, scale: float = 3.0
                 ) -> list[dict | None]:
        # one small image per state: the API downsizes big images (short side
        # <= 768 px), which would blur the digits of a grid of crops
        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, t in enumerate(tiles):
            big = cv2.resize(t, (int(t.shape[1] * scale), int(t.shape[0] * scale)),
                             interpolation=cv2.INTER_CUBIC)
            content.append({"type": "text", "text": f"#{i}"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{_jpeg_b64(big, max_w=768)}",
                "detail": "high",
            }})
        extra = {"reasoning_effort": self.effort} if self.effort else {}
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}], **extra,
        )
        self.calls += 1
        data = json.loads(resp.choices[0].message.content or "{}")
        return parse_tile_answer(data, len(tiles))


def _int_or_none(v) -> int | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        iv = int(v)
        return iv if 0 <= iv <= 300 else None
    except (TypeError, ValueError):
        return None


def parse_tile_answer(data: dict, n: int) -> list[dict | None]:
    out: list[dict | None] = [None] * n
    for item in (data or {}).get("tiles") or []:
        if not isinstance(item, dict):
            continue
        i = _int_or_none(item.get("i"))
        if i is None or i >= n:
            continue
        p = _int_or_none(item.get("period"))
        out[i] = {
            "left": _int_or_none(item.get("left")),
            "right": _int_or_none(item.get("right")),
            "period": p if p is not None and 1 <= p <= 9 else None,
        }
    return out


ReadFn = Callable[[list, str], list]


def _valid(r: dict | None) -> bool:
    return bool(r) and r.get("left") is not None and r.get("right") is not None


def suspicious_states(reads: list[dict | None], max_points: int = 3) -> list[int]:
    """Indices whose reading is an impossible step from the last valid one
    (a score went down, or rose by more than ``max_points``)."""
    bad, prev = [], None
    for j, r in enumerate(reads):
        if not _valid(r):
            continue
        if prev is not None:
            for side in ("left", "right"):
                d = r[side] - prev[side]
                if d < 0 or d > max_points:
                    bad.append(j)
                    break
        prev = r
    return bad


def read_states(
    states: list[dict], reader: ReadFn, per_request: int = 20,
    cache: dict[str, dict | None] | None = None, max_points: int = 3,
    verify_radius: int = 3,
) -> list[dict]:
    """Read every state in chronological batches; returns the timeline.

    Each request gets the last reading of the previous one as context.  Then
    a verify pass re-reads a small window (+-``verify_radius`` states) around
    every impossible step, with the last plausible reading as context; the
    re-read replaces the first one.  ``cache`` maps ``f"{t:.2f}"`` (segment
    start) -> reading, updated in place; a rerun asks nothing new.
    """
    cache = cache if cache is not None else {}
    keys = [f"{s['t']:.2f}" for s in states]

    def ask(idxs: list[int], prev: dict | None) -> None:
        try:
            answers = reader([states[i]["tile"] for i in idxs], read_prompt(prev))
        except Exception as exc:  # noqa: BLE001 - one failed request != failed job
            logger.warning("Scoreboard read failed for %d tiles: %s", len(idxs), exc)
            answers = [None] * len(idxs)
        for i, a in zip(idxs, answers):
            cache[keys[i]] = a

    def last_valid(before: int) -> dict | None:
        for i in range(before - 1, -1, -1):
            if _valid(cache.get(keys[i])):
                return cache[keys[i]]
        return None

    for k in range(0, len(states), per_request):
        idxs = list(range(k, min(k + per_request, len(states))))
        if not all(keys[i] in cache for i in idxs):
            ask(idxs, last_valid(k))

    verified_key = "__verified__"
    stamp = {"n": len(keys), "first": keys[0] if keys else None}
    if (cache.get(verified_key) or {}).get("stamp") != stamp:
        bad = suspicious_states([cache.get(key) for key in keys], max_points)
        windows: list[list[int]] = []
        for j in bad:
            a, b = max(0, j - verify_radius), min(len(states), j + verify_radius + 1)
            if windows and a <= windows[-1][1]:
                windows[-1][1] = b
            else:
                windows.append([a, b])
        for a, b in windows:
            prev = last_valid(a)
            for k in range(a, b, per_request):
                ask(list(range(k, min(k + per_request, b))), prev)
                prev = last_valid(min(k + per_request, b))
        logger.info("Scoreboard verify: %d suspicious steps, %d windows re-read",
                    len(bad), len(windows))
        cache[verified_key] = {"stamp": stamp, "bad_before": len(bad)}
    return [{"t": s["t"], **(cache.get(key) or {})} for s, key in zip(states, keys)]


# ---------------------------------------------------------------------------
# 4. Cleaning + events
# ---------------------------------------------------------------------------

def _clean_series(vals: list[int | None], max_points: int | None = None,
                  blips: list | None = None) -> list[int | None]:
    """Fix lone misreads: v[i] differs from equal neighbours -> neighbours' value.

    Neighbours = nearest readable values on each side.  With ``max_points``:
    a lone value 1..max_points above equal neighbours is also what a scorer's
    mistake looks like (points added, then taken back) -> recorded in
    ``blips`` as ``(index, points)``; and a value that overshoots a later
    smaller rise (10, 13, 12: a 3 corrected to a 2) is kept, so the basket
    keeps its original time and :func:`reconcile_corrections` fixes the points.
    """
    out = list(vals)
    idx = [i for i, v in enumerate(vals) if v is not None]
    for k in range(1, len(idx) - 1):
        a, i, b = vals[idx[k - 1]], vals[idx[k]], vals[idx[k + 1]]
        if a == b and i != a:
            out[idx[k]] = a
            if blips is not None and max_points and 0 < i - a <= max_points:
                blips.append((idx[k], i - a))
    # value seen once and not between its neighbours (e.g. 18 between 12 and 14)
    idx = [i for i, v in enumerate(out) if v is not None]
    for k in range(1, len(idx) - 1):
        a, i, b = out[idx[k - 1]], out[idx[k]], out[idx[k + 1]]
        if not (min(a, b) <= i <= max(a, b)):
            if max_points and a < b < i and i - a <= max_points:
                continue  # scorer's correction of the points, not a misread
            out[idx[k]] = a
    return out


def reconcile_corrections(result: dict, blips: list[dict] | None = None,
                          window_sec: float = 180.0, swap_sec: float = 20.0) -> dict:
    """Undo the scorer's mistakes the board later corrected (in place).

    * a score drop of d for a team is taken off that team's latest baskets
      (within ``window_sec``): +3 then -1 -> a 2-pointer at the original time,
      +2 then -2 -> the basket is cancelled (points 0, flag ``cancelled``);
    * points cancelled for one team and given to the other team within
      ``swap_sec`` of the correction (the scorer fixing both at once) are the
      same basket entered for the wrong team: the other team's basket gets
      the original time (flag ``moved``);
    * ``blips`` (points shown only briefly, see :func:`_clean_series`) can be
      such a wrong-team entry too.
    Running scores are recounted.
    """
    events = sorted(result["events"], key=lambda e: e["board_t"])
    wrong: list[tuple[float, str, int, float]] = []  # (orig t, team, points, corr t)
    for j, e in enumerate(events):
        if e["points"] >= 0:
            continue
        need = -e["points"]
        for c in reversed(events[:j]):
            if need <= 0 or c["board_t"] < e["board_t"] - window_sec:
                break
            if c["team"] != e["team"] or c["points"] <= 0 or c.get("flag") == "check":
                continue  # a >3 jump needs a human look anyway
            take = min(c["points"], need)
            c.setdefault("board_points", c["points"])
            c["points"] -= take
            need -= take
            c["flag"] = "cancelled" if c["points"] == 0 else "corrected"
            wrong.append((c["board_t"], c["team"], take, e["board_t"]))
        e["flag"] = "correction"
        e["unmatched"] = need
    for b in blips or []:
        wrong.append((b["t"], b["team"], b["points"], b["t_end"]))
    moved: set[int] = set()
    for t0, team, pts, t_corr in sorted(wrong):
        for k, f in enumerate(events):
            if (k in moved or f["team"] == team or f["points"] != pts
                    or f.get("flag") or f["board_t"] <= t0
                    or abs(f["board_t"] - t_corr) > swap_sec):
                continue
            f["entered_t"] = f["board_t"]
            f["board_t"] = round(t0 + 0.01, 2)  # key differs from the cancelled one
            f["flag"] = "moved"
            moved.add(k)
            break
    result["events"] = sorted(events, key=lambda e: e["board_t"])
    recount(result)
    return result


def recount(result: dict) -> None:
    """Running score after every event (corrections count only the part
    that could not be taken off an earlier basket)."""
    run = {s: (result.get("start") or {}).get(s) for s in SIDES}
    for ev in result["events"]:
        eff = -ev.get("unmatched", 0) if ev.get("flag") == "correction" else ev["points"]
        run[ev["team"]] = (run[ev["team"]] or 0) + eff
        ev["home"], ev["guest"] = run["home"], run["guest"]


def apply_event_edits(result: dict, edits: dict | None) -> dict:
    """Manual fixes from the review page (in place): ``{key: {"cancel": true}
    | {"team": "home"|"guest"} | {"points": 1..3}}`` (``number`` handled by
    the shots).  Originals are kept in ``orig`` so removing an edit restores
    them."""
    for e in result["events"]:
        o = e.pop("orig", None)
        if o:
            e.update(o)
            if e.get("flag") == "edited":
                e.pop("flag")
    for e in result["events"]:
        ed = (edits or {}).get(f"{float(e['board_t']):.2f}") or {}
        change = {}
        if ed.get("cancel"):
            change["points"] = 0
        else:
            if ed.get("team") in SIDES and ed["team"] != e["team"]:
                change["team"] = ed["team"]
            if isinstance(ed.get("points"), int) and 1 <= ed["points"] <= 3:
                change["points"] = ed["points"]
        if change and e.get("flag") != "correction":
            e["orig"] = {k: e.get(k) for k in ("team", "points", "flag")}
            e.update(change)
            e["flag"] = "edited"
    recount(result)
    return result


def load_edits(out_dir: str | Path) -> dict:
    """Manual fixes per event key (``edits.json``, written by the dashboard)."""
    p = Path(out_dir) / "edits.json"
    try:
        return json.loads(p.read_text()).get("events", {}) if p.is_file() else {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def derive_events(timeline: list[dict], home_side: str = "left",
                  max_points: int = 3) -> dict:
    """Score changes from cleaned readings.

    Returns ``{"events": [...], "periods": [...], "final": {...},
    "start": {...}}``; event = ``{"board_t", "team", "points", "home",
    "guest", "period", "flag"?}``.  ``board_t`` is when the board changed.
    """
    ts = [r["t"] for r in timeline]
    side_key = {"home": home_side, "guest": "right" if home_side == "left" else "left"}
    raw_blips: dict[str, list] = {s: [] for s in SIDES}
    series = {s: _clean_series([r.get(side_key[s]) for r in timeline], max_points,
                               raw_blips[s]) for s in SIDES}
    period = _clean_series([r.get("period") for r in timeline])

    events: list[dict] = []
    cur = {s: None for s in SIDES}
    start = {}
    cur_period = None
    periods: list[dict] = []
    for j, t in enumerate(ts):
        p = period[j]
        if p is not None and p != cur_period:
            if cur_period is None or p > cur_period:
                periods.append({"board_t": t, "period": p})
            cur_period = p
        for s in SIDES:
            v = series[s][j]
            if v is None:
                continue
            if cur[s] is None:
                cur[s] = v
                start[s] = v
                continue
            if v == cur[s]:
                continue
            delta = v - cur[s]
            cur[s] = v
            ev = {"board_t": t, "team": s, "points": delta,
                  "period": cur_period}
            if delta < 0:
                ev["flag"] = "correction"
            elif delta > max_points:
                ev["flag"] = "check"
            events.append(ev)
    blips = [{"t": ts[j], "t_end": ts[min(j + 1, len(ts) - 1)], "team": s, "points": p}
             for s in SIDES for j, p in raw_blips[s]]
    result = {"start": start, "events": events, "periods": periods,
              "final": {s: cur[s] for s in SIDES}}
    return reconcile_corrections(result, blips)


# ---------------------------------------------------------------------------
# 5. Outputs
# ---------------------------------------------------------------------------

def fmt_ts(sec: float) -> str:
    s = int(max(0, sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _period_name(p: int) -> str:
    return f"{p}. kwarta" if p <= 4 else f"Dogrywka {p - 4}" if p > 5 else "Dogrywka"


def _score(ev: dict) -> str:
    h = "?" if ev.get("home") is None else ev["home"]
    g = "?" if ev.get("guest") is None else ev["guest"]
    return f"{h}:{g}"


def scorers_lines(pts: list[dict], match: dict, names: dict) -> list[str]:
    """"Strzelcy:" section: points per known player, per team."""
    out = []
    for side in SIDES:
        tally: dict[str, int] = {}
        for e in pts:
            who = player_label(e, match)
            if e["team"] == side and who:
                tally[who] = tally.get(who, 0) + e["points"]
        if tally:
            best = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
            out.append(f"{names[side]}: " + ", ".join(f"{w} {p}" for w, p in best))
    return (["Strzelcy:"] + out + [""]) if out else []


def youtube_text(result: dict, match: dict, lead_sec: float = 6.0,
                 chapters: bool = True, players: bool = True) -> str:
    names = {"home": match["home"], "guest": match["guest"]}
    fin = result.get("final") or {}
    lines = [f"{names['home']} – {names['guest']}  "
             f"{fin.get('home', '?')}:{fin.get('guest', '?')}", ""]

    if chapters:
        ch: list[tuple[float, str]] = []
        for p in result.get("periods") or []:
            t = max(0.0, p["board_t"] - lead_sec)
            if ch and t - ch[-1][0] < 10:  # YouTube: chapters >= 10 s apart
                ch[-1] = (ch[-1][0], _period_name(p["period"]))
                continue
            ch.append((t, _period_name(p["period"])))
        if ch and ch[0][0] >= 10:
            ch.insert(0, (0.0, "Przed meczem"))
        elif ch:
            ch[0] = (0.0, ch[0][1])
        if len(ch) >= 3:
            lines.append("Rozdziały:")
            lines += [f"{fmt_ts(t)} {name}" for t, name in ch]
            lines.append("")

    pts = [e for e in result.get("events") or [] if e["points"] > 0]
    if pts:
        lines.append("Punkty:")
        for e in pts:
            t = e.get("t", e["board_t"] - lead_sec)
            who = player_label(e, match) if players else ""
            lines.append(f"{fmt_ts(t)} {_score(e)} {names[e['team']]} +{e['points']}"
                         + (f" – {who}" if who else ""))
        lines.append("")
        if players:
            lines += scorers_lines(pts, match, names)
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(out_dir: Path, result: dict, match: dict, sc_cfg: dict) -> dict:
    out_dir = Path(out_dir)
    lead = float(sc_cfg.get("lead_sec", 6.0))
    shot_lead = float(shots_config(sc_cfg)["shot_lead_sec"])
    for e in result["events"]:
        if e.get("shot_t") is not None:  # stage 2 found the shot itself
            e["t"] = round(max(0.0, e["shot_t"] - shot_lead), 2)
        else:
            e["t"] = round(max(0.0, e["board_t"] - lead), 2)
        e["timestamp"] = fmt_ts(e["t"])
        e["team_name"] = match[e["team"]]
    doc = {"match": match, "lead_sec": lead, **result}
    paths = {
        "json": out_dir / "scoring.json",
        "csv": out_dir / "scoring.csv",
        "youtube": out_dir / "youtube.txt",
    }
    paths["json"].write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    with paths["csv"].open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "t_sec", "board_t_sec", "period", "team",
                    "team_name", "points", "home", "guest", "flag", "shot_t_sec",
                    "shooter_colour", "colour_ok", "number", "player", "number_src",
                    "number_check", "shot_crop", "board_points", "entered_t_sec"])
        for e in result["events"]:
            sh = e.get("shot") or {}
            ok = sh.get("colour_ok")
            w.writerow([e["timestamp"], e["t"], e["board_t"], e.get("period"),
                        e["team"], e["team_name"], e["points"], e.get("home"),
                        e.get("guest"), e.get("flag", ""), sh.get("t", ""),
                        sh.get("colour", ""), "" if ok is None else int(ok),
                        "" if sh.get("number") is None else sh["number"],
                        player_label(e, match, uncertain=True),
                        sh.get("number_src") or "", int(bool(sh.get("number_check"))),
                        sh.get("crop", ""), e.get("board_points", ""),
                        e.get("entered_t", "")])
    paths["youtube"].write_text(
        youtube_text(result, match, lead, bool(sc_cfg.get("chapters", True)),
                     bool(jerseys_config(sc_cfg)["in_youtube"]))
    )
    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_scoring(
    config: dict, video_path: str | Path, cal: dict, width: int, height: int,
    out_dir: str | Path, reader: ReadFn | None = None, *, fps: float = 30.0,
    detections_path: str | Path | None = None, shot_reader=None,
) -> dict | None:
    """Scoring for one video (board + shooters). Returns output paths or None.

    Shooters (stage 2, ``scoring.shooters.enabled``) need player detections:
    ``detections_path`` or ``<out_dir>/detections.jsonl``.
    """
    sc = scoring_config(config)
    sb = cal.get("scoreboard") if cal else None
    if not sb:
        logger.warning("Scoring skipped: no scoreboard in calibration")
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    match = match_info(cal)

    cache_path = out_dir / "scoreboard_reads.json"
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text()).get("reads", {})
        except (json.JSONDecodeError, AttributeError):
            cache = {}

    times, crops = extract_board_crops(
        video_path, sb, cal.get("camera_shifts"), width, height,
        margin=float(sc["crop_margin"]),
    )
    changes = detect_changes(
        crops, led=str(sc["led"]), threshold=float(sc["change_threshold"]),
        margin=float(sc["crop_margin"]),
    )
    states = segment_states(times, crops, changes)
    if reader is None:
        llm = OpenAIScoreReader(sc["llm"])
        scale = float(sc["tile_scale"])
        reader = lambda tiles, prompt: llm(tiles, prompt, scale)  # noqa: E731
    try:
        timeline = read_states(
            states, reader, per_request=int(sc["tiles_per_request"]), cache=cache,
            max_points=int(sc["max_points"]),
        )
    finally:
        cache_path.write_text(json.dumps({"scoreboard": sb, "reads": cache}, indent=1))
    result = derive_events(timeline, match["home_side"], int(sc["max_points"]))
    if shots_config(sc)["enabled"]:
        det = Path(detections_path) if detections_path else out_dir / "detections.jsonl"
        if det.is_file():
            try:
                doc = find_shots(config, sc, video_path, cal, result["events"],
                                 width, height, fps, det, out_dir, reader=shot_reader)
                apply_shots(result, doc, load_edits(out_dir))
            except Exception:
                logger.exception("Shooters failed; scores without shots")
        else:
            logger.warning("Shooters skipped: no detections at %s", det)
    apply_event_edits(result, load_edits(out_dir))
    paths = write_outputs(out_dir, result, match, sc)
    logger.info(
        "Scoring: %d events, final %s %s:%s %s -> %s",
        len(result["events"]), match["home"], result["final"].get("home"),
        result["final"].get("guest"), match["guest"], paths["youtube"],
    )
    return paths


def run_shots_only(config: dict, video_path: str | Path, cal: dict, width: int,
                   height: int, fps: float, out_dir: str | Path,
                   shot_reader=None) -> dict:
    """Stage 2 on top of an existing ``scoring.json`` (no scoreboard pass)."""
    out_dir = Path(out_dir)
    sc = scoring_config(config)
    doc = json.loads((out_dir / "scoring.json").read_text())
    events = [{k: e[k] for k in ("board_t", "team", "points", "period") if k in e}
              for e in doc["events"]]
    find_shots(config, sc, video_path, cal, events, width, height, fps,
               out_dir / "detections.jsonl", out_dir, reader=shot_reader)
    return regenerate_outputs(config, out_dir, cal)


def regenerate_outputs(config: dict, out_dir: str | Path, cal: dict | None) -> dict:
    """Rewrite csv/youtube from an existing scoring.json (e.g. new team names)."""
    out_dir = Path(out_dir)
    doc = json.loads((out_dir / "scoring.json").read_text())
    match = match_info(cal) if cal and cal.get("match") else match_info({"match": doc.get("match")})
    old_side = (doc.get("match") or {}).get("home_side", "left")
    if match["home_side"] != old_side:  # sides swapped: rebuild from readings
        reads = json.loads((out_dir / "scoreboard_reads.json").read_text())["reads"]
        timeline = [{"t": float(t), **(r or {})} for t, r in
                    sorted(((t, r) for t, r in reads.items() if not t.startswith("__")),
                           key=lambda kv: float(kv[0]))]
        result = derive_events(timeline, match["home_side"],
                               int(scoring_config(config)["max_points"]))
    else:
        result = {k: doc[k] for k in ("start", "events", "periods", "final")}
    for e in result["events"]:  # re-derived below (mapping depends on sides)
        e.pop("shot", None)
        e.pop("shot_t", None)
    edits = load_edits(out_dir)
    shots = out_dir / "shots.json"
    if shots.is_file():
        apply_shots(result, json.loads(shots.read_text()), edits)
    apply_event_edits(result, edits)
    return write_outputs(out_dir, result, match, scoring_config(config))
