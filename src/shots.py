"""Who shot: the scoring shot, its shooter and the team jersey colours.

Stage 2 of scoring (stage 1 = scoreboard, :mod:`src.scoring`).  The board
tells *that* a team scored and roughly when (the scorer updates it a few
seconds later).  Here, per score event:

1. **Attacked basket.** Player detections (``detections.jsonl``) in the
   seconds before the board update say which basket the play is at.  One
   vote per event decides which basket each team attacks per half (teams
   swap at half time), so a single counter-attack does not fool it.
2. **Frames.** ``window_sec`` of video before the board update, one frame per
   ``step_sec``, cut around the attacked basket at full resolution (exact
   frame numbers, so detection boxes line up).
3. **Team colours.** Torso colours of all detected players in those frames
   are clustered (k-means, 3 clusters: two teams + referees/others); the
   two biggest clusters are the teams.
4. **Shot.** A vision LLM gets the frames with lettered player boxes and the
   two team colours, and names the release frame, the shooter's box and his
   team colour.  Which colour belongs to home/guest is voted over all events
   against the board; events whose shooter colour disagrees with the board
   are asked again with the scoring team's colour as a hint.
5. Results go to ``shots.json`` (cache of LLM answers + colours) and a crop
   of every shooter to ``shots/`` (input for jersey numbers, stage 3).

:func:`apply_shots` merges ``shots.json`` into the score events (shot time
for the YouTube timestamps, shooter colour, colour/board agreement).  It is
pure, so team-name or side changes re-apply it without new LLM calls.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .calibration import pitch_to_y, shift_at, x_to_yaw, y_to_pitch, yaw_to_x

logger = logging.getLogger("soccer360.shots")

SHOTS_VERSION = 1
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def shots_config(sc_cfg: dict) -> dict:
    """``scoring.shooters`` with defaults (``sc_cfg`` = scoring config)."""
    c = dict(sc_cfg.get("shooters", {}) or {})
    c.setdefault("enabled", False)
    c.setdefault("window_sec", 15.0)      # video before the board update
    c.setdefault("step_sec", 1.0)         # one frame per step
    c.setdefault("shot_lead_sec", 3.0)    # YouTube timestamp = shot - lead
    c.setdefault("min_conf", 0.25)        # player detections used
    c.setdefault("workers", 4)            # parallel LLM requests
    c.setdefault("verify", True)          # re-ask colour/board disagreements
    c.setdefault("max_w", 1100)           # px per frame sent to the LLM
    c.setdefault("player_class", 0)
    own = c.get("llm") or {}
    c["llm"] = {**(sc_cfg.get("llm") or {}), **own}
    if "model" not in own:  # measured: as good as gpt-5 here at ~1/7 of the cost
        c["llm"]["model"] = "gpt-5-mini"
    return c


def det_resolution(config: dict) -> tuple[int, int]:
    """Pixel space of ``detections.jsonl`` boxes."""
    if "detection" in config:
        s = int(config["detection"].get("img_size", 960))
        return 2 * s, s
    w, h = config.get("detector", {}).get("resolution", [1920, 960])
    return int(w), int(h)


# ---------------------------------------------------------------------------
# Player detections
# ---------------------------------------------------------------------------

class Players:
    """Player boxes from ``detections.jsonl``, indexed by frame."""

    def __init__(self, frames: np.ndarray, boxes: np.ndarray, det_w: int, det_h: int):
        order = np.argsort(frames, kind="stable")
        self.frames = frames[order].astype(np.int64)
        self.boxes = boxes[order].astype(np.float32)
        self.det_w, self.det_h = det_w, det_h

    @classmethod
    def load(cls, path: str | Path, det_w: int, det_h: int, *,
             player_class: int = 0, min_conf: float = 0.4) -> "Players":
        fr, bx = [], []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("class_id") != player_class or d.get("conf", 0) < min_conf:
                    continue
                fr.append(d["frame_index"])
                bx.append(d["bbox_xyxy"])
        return cls(np.array(fr, np.int64), np.array(bx, np.float32).reshape(-1, 4),
                   det_w, det_h)

    def range(self, f0: int, f1: int) -> slice:
        """Rows with frame in [f0, f1)."""
        return slice(int(np.searchsorted(self.frames, f0)),
                     int(np.searchsorted(self.frames, f1)))

    def at(self, frame: int) -> np.ndarray:
        return self.boxes[self.range(frame, frame + 1)]

    def feet(self, sl: slice) -> tuple[np.ndarray, np.ndarray]:
        b = self.boxes[sl]
        return (x_to_yaw((b[:, 0] + b[:, 2]) / 2, self.det_w),
                y_to_pitch(b[:, 3], self.det_h))


def _hoops_at(cal: dict, t: float) -> list[dict]:
    """Calibrated hoops shifted to time ``t``, sorted by yaw."""
    dy, dp = shift_at(cal.get("camera_shifts"), t)
    hs = [{**h, "yaw": h["yaw"] + dy, "rim_pitch": h["rim_pitch"] + dp}
          for h in cal.get("hoops") or []]
    return sorted(hs, key=lambda h: h["yaw"])


def play_hoop(players: Players, cal: dict, t0: float, t1: float, fps: float) -> int | None:
    """Index (into yaw-sorted hoops) of the basket the play is at in [t0, t1)."""
    hoops = _hoops_at(cal, t1)
    if len(hoops) != 2:
        return None
    yaw, pitch = players.feet(players.range(int(t0 * fps), int(t1 * fps)))
    lo, hi = hoops[0]["yaw"] - 10, hoops[1]["yaw"] + 10
    keep = (yaw > lo) & (yaw < hi) & (pitch < min(h["rim_pitch"] for h in hoops))
    if keep.sum() < 3:
        return None
    mid = (hoops[0]["yaw"] + hoops[1]["yaw"]) / 2
    return int(np.mean(yaw[keep]) > mid)


def _half(period) -> int:
    return 1 if (period or 1) <= 2 else 2


def attack_map(events: list[dict], votes: list[int | None]) -> dict | None:
    """``{(team, half): hoop}`` from per-event play positions.

    Teams attack opposite baskets and swap at half time, so every vote says
    something about one bit: the basket home attacks in the first half.
    """
    tally = [0, 0]
    for e, v in zip(events, votes):
        if v is None:
            continue
        home_h1 = v if (e["team"] == "home") == (_half(e.get("period")) == 1) else 1 - v
        tally[home_h1] += 1
    if not any(tally):
        return None
    k = int(tally[1] > tally[0])
    return {("home", 1): k, ("guest", 1): 1 - k, ("home", 2): 1 - k, ("guest", 2): k}


def view_box(hoop: dict, other: dict, width: int, height: int,
             cal: dict) -> tuple[int, int, int, int]:
    """Full-res pixel box around a basket, extended towards mid-court."""
    s = 1.0 if other["yaw"] > hoop["yaw"] else -1.0
    a, b = sorted((hoop["yaw"] - 15 * s, hoop["yaw"] + 35 * s))
    corners = (cal.get("court") or {}).get("corners") or []
    bottom = min([c["pitch"] for c in corners] + [-26.0]) - 6
    top = hoop["rim_pitch"] + 10
    x0, x1 = int(yaw_to_x(a, width)), int(yaw_to_x(b, width))
    if x1 <= x0:  # wraps around +-180: keep the side with the hoop
        x0, x1 = (0, x1) if s > 0 else (x0, width)
    y0, y1 = int(pitch_to_y(top, height)), int(pitch_to_y(bottom, height))
    x0, y0 = max(0, x0 - x0 % 2), max(0, y0 - y0 % 2)
    x1, y1 = min(width, x1 - (x1 - x0) % 2), min(height, y1 - (y1 - y0) % 2)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def grab_frames(video_path: str | Path, t0: float, n: int, every: int,
                box: tuple[int, int, int, int]) -> list[np.ndarray]:
    """``n`` frames from ``t0``, every ``every``-th frame, cropped to ``box``.

    Frames are picked by number after an accurate seek (an ``fps`` filter
    would shift them by half a step against the detections).
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, t0):.3f}", "-i", str(video_path),
        "-vf", f"select='not(mod(n\\,{every}))',crop={w}:{h}:{x0}:{y0}",
        "-fps_mode", "passthrough", "-frames:v", str(n),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    k = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: k * w * h * 3], np.uint8).reshape(k, h, w, 3)
    return [a.copy() for a in arr]


def torso_lab(img: np.ndarray, box) -> np.ndarray | None:
    """Median Lab colour of the shirt (upper-middle part of a person box)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if h < 30 or w < 8:
        return None
    c = img[int(max(0, y0 + 0.2 * h)):int(max(0, y0 + 0.45 * h)),
            int(max(0, x0 + 0.3 * w)):int(max(0, x1 - 0.3 * w))]
    if c.size < 30:
        return None
    lab = cv2.cvtColor(np.ascontiguousarray(c), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return np.median(lab, 0).astype(np.float32)


_WEIGHT = np.array([0.5, 1.0, 1.0], np.float32)  # lighting moves L the most


def cluster_colours(feats: np.ndarray, k: int = 3) -> dict | None:
    """Two team colours from shirt samples: biggest two of ``k`` clusters."""
    if len(feats) < 4 * k:
        return None
    x = feats.astype(np.float32) * _WEIGHT
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    cv2.setRNGSeed(7)
    _, lab, cent = cv2.kmeans(x, k, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lab.ravel(), minlength=k)
    top = [int(i) for i in np.argsort(-counts)[:2]]
    cent = cent / _WEIGHT
    teams = []
    for i in top:
        rgb = lab_to_rgb(cent[i])
        teams.append({"lab": [round(float(v), 1) for v in cent[i]],
                      "rgb": rgb, "name": colour_name(rgb),
                      "samples": int(counts[i])})
    return {"teams": teams, "centres": [[round(float(v), 1) for v in c] for c in cent],
            "team_clusters": top}


def nearest_team_colour(lab: np.ndarray | None, colours: dict | None) -> int | None:
    """0/1 = which team colour a shirt is, None = other cluster (referee...)."""
    if lab is None or not colours:
        return None
    cent = np.array(colours["centres"], np.float32) * _WEIGHT
    d = np.linalg.norm(cent - np.asarray(lab, np.float32) * _WEIGHT, axis=1)
    i = int(np.argmin(d))
    tc = colours["team_clusters"]
    return tc.index(i) if i in tc else None


def lab_to_rgb(lab) -> list[int]:
    px = np.clip(np.array(lab, np.float32), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    return [int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2RGB)[0, 0]]


def colour_name(rgb) -> dict:
    """Plain colour words for a shirt colour: ``{"en": ..., "pl": ...}``."""
    px = np.array(rgb, np.uint8).reshape(1, 1, 3)
    h, s, v = cv2.cvtColor(px, cv2.COLOR_RGB2HSV_FULL)[0, 0]
    h, s, v = float(h) / 255 * 360, float(s) / 255, float(v) / 255
    if v < 0.22:
        return {"en": "black", "pl": "czarne"}
    if s < 0.18:
        return ({"en": "white", "pl": "białe"} if v > 0.7
                else {"en": "grey", "pl": "szare"})
    if h < 10 or h >= 340:  # indoor light pulls orange down to ~13 deg
        return {"en": "red", "pl": "czerwone"}
    if h < 45:
        if v < 0.5:
            return {"en": "brown", "pl": "brązowe"}
        if s < 0.5:
            return {"en": "beige", "pl": "beżowe"}
        return {"en": "orange", "pl": "pomarańczowe"}
    if h < 70:
        return ({"en": "beige", "pl": "beżowe"} if s < 0.45
                else {"en": "yellow", "pl": "żółte"})
    if h < 165:
        return {"en": "green", "pl": "zielone"}
    if h < 200:
        return {"en": "turquoise", "pl": "turkusowe"}
    if h < 255:
        return ({"en": "navy blue", "pl": "granatowe"} if v < 0.45
                else {"en": "blue", "pl": "niebieskie"})
    if h < 290:
        return {"en": "purple", "pl": "fioletowe"}
    return {"en": "pink", "pl": "różowe"}


def draw_boxes(img: np.ndarray, boxes: list) -> np.ndarray:
    """Copy of ``img`` with lettered boxes (letter i = boxes[i])."""
    out = img.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(out, p0, p1, (255, 255, 0), 2)
        cv2.putText(out, LETTERS[i], (p0[0], max(14, p0[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return out


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_SHOT_PROMPT = """\
These are {n} consecutive frames, {step:g} s apart (#0 earliest), from a fixed \
camera at a basketball game, cropped around one basket. During this clip a \
team scored {points} point(s) ({kind}) at this basket. {teams}{hint}
Yellow boxes with letters mark detected people in each frame (letters are per \
frame, not consistent across frames; not every person has a box).

Find the scoring shot: the frame where the shooter releases the ball (or the \
closest frame) and the shooter. JSON only:
{{"shot_frame": int, "shooter_box": "letter in that frame, or null if the \
shooter has no box", "shooter_xy": [x, y] (shooter's centre in that frame, \
fractions 0-1 of width and height), "shooter_team": {team_enum}, \
"jersey_colour": "short colour description of the shooter's shirt", \
"jersey_number": int or null (only if clearly legible), \
"confidence": "high" | "medium" | "low", "notes": "short"}}"""


def _kind(points: int) -> str:
    return {1: "a free throw", 3: "a three-pointer"}.get(points, "a field goal")


def shot_prompt(n: int, step: float, points: int, colours: dict | None,
                hint: int | None = None) -> str:
    if colours:
        a, b = (c["name"]["en"] for c in colours["teams"])
        teams = f"One team wears {a} shirts (team \"A\"), the other {b} (team \"B\"). "
        team_enum = '"A" | "B" | null'
        hint_s = (f"The scorer's table says team {'AB'[hint]} "
                  f"({colours['teams'][hint]['name']['en']}) scored."
                  if hint is not None else "")
    else:
        teams, team_enum, hint_s = "", "null", ""
    return _SHOT_PROMPT.format(n=n, step=step, points=points, kind=_kind(points),
                               teams=teams, hint=hint_s, team_enum=team_enum)


ShotFn = Callable[[list, str], dict]


class OpenAIShotReader:
    """Asks an OpenAI vision model about one clip (list of RGB frames)."""

    def __init__(self, llm_cfg: dict, max_w: int = 1100):
        from openai import OpenAI  # optional dependency

        env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"{env} is not set")
        self.model = llm_cfg.get("model", "gpt-5")
        self.effort = llm_cfg.get("reasoning_effort")  # minimal|low|medium|high
        self.usage = {"input": 0, "output": 0, "reasoning": 0}
        self.client = OpenAI(api_key=key, timeout=float(llm_cfg.get("timeout_sec", 300)),
                             max_retries=4)
        self.max_w = max_w
        self.calls = 0
        self.dead: str | None = None   # set when the account has no credits
        self._lock = threading.Lock()

    def __call__(self, frames: list[np.ndarray], prompt: str) -> dict:
        if self.dead:  # don't upload frames for requests that cannot run
            raise RuntimeError(self.dead)
        try:
            return self._call(frames, prompt)
        except Exception as exc:
            if "insufficient_quota" in str(exc) or "no credits" in str(exc):
                self.dead = f"OpenAI: no credits left ({exc.__class__.__name__})"
            raise

    def _call(self, frames: list[np.ndarray], prompt: str) -> dict:
        # Frames go up through the Files API and the request only names them:
        # big inline base64 requests were dropped mid-upload ("Server
        # disconnected"), small file uploads are reliable.
        ids: list[str] = []
        try:
            content: list[dict] = [{"type": "input_text", "text": prompt}]
            for i, f in enumerate(frames):
                ids.append(self.client.files.create(
                    file=(f"frame{i}.jpg", _jpeg_bytes(f, self.max_w), "image/jpeg"),
                    purpose="vision").id)
                content.append({"type": "input_text", "text": f"#{i}"})
                content.append({"type": "input_image", "file_id": ids[-1], "detail": "high"})
            extra = {"reasoning": {"effort": self.effort}} if self.effort else {}
            resp = self.client.responses.create(
                model=self.model, input=[{"role": "user", "content": content}],
                text={"format": {"type": "json_object"}}, **extra,
            )
        finally:
            for fid in ids:
                try:
                    self.client.files.delete(fid)
                except Exception:  # noqa: BLE001 - cleanup only
                    logger.debug("Could not delete uploaded frame %s", fid)
        with self._lock:
            self.calls += 1
            u = getattr(resp, "usage", None)
            if u is not None:
                self.usage["input"] += int(getattr(u, "input_tokens", 0) or 0)
                self.usage["output"] += int(getattr(u, "output_tokens", 0) or 0)
                det = getattr(u, "output_tokens_details", None)
                self.usage["reasoning"] += int(getattr(det, "reasoning_tokens", 0) or 0)
        return json.loads(resp.output_text or "{}")

    def summary(self) -> str:
        u = self.usage
        return (f"{self.model} ({self.effort or 'default'} effort): {self.calls} requests, "
                f"{u['input']} input / {u['output']} output tokens "
                f"({u['reasoning']} reasoning)")


def _jpeg_bytes(img_rgb: np.ndarray, max_w: int) -> bytes:
    h, w = img_rgb.shape[:2]
    if w > max_w:
        img_rgb = cv2.resize(img_rgb, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img_rgb[..., ::-1]),
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def parse_shot_answer(data: dict, n_frames: int, n_boxes: list[int]) -> dict:
    """Validate an LLM answer; unknown/invalid fields become None."""
    out: dict = {"frame": None, "box": None, "xy": None, "team": None,
                 "number": None, "colour": None, "confidence": None}
    if not isinstance(data, dict):
        return out
    f = data.get("shot_frame")
    if isinstance(f, int) and not isinstance(f, bool) and 0 <= f < n_frames:
        out["frame"] = f
        b = data.get("shooter_box")
        if isinstance(b, str) and len(b.strip()) == 1:
            i = LETTERS.find(b.strip().upper())
            if 0 <= i < n_boxes[f]:
                out["box"] = i
    xy = data.get("shooter_xy")
    if (isinstance(xy, (list, tuple)) and len(xy) == 2
            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in xy)):
        out["xy"] = [round(float(xy[0]), 3), round(float(xy[1]), 3)]
    t = data.get("shooter_team")
    if t in ("A", "B"):
        out["team"] = "AB".index(t)
    num = data.get("jersey_number")
    if isinstance(num, int) and not isinstance(num, bool) and 0 <= num <= 99:
        out["number"] = num
    if isinstance(data.get("jersey_colour"), str):
        out["colour"] = data["jersey_colour"][:60]
    if data.get("confidence") in ("high", "medium", "low"):
        out["confidence"] = data["confidence"]
    return out


# ---------------------------------------------------------------------------
# Applying results to score events
# ---------------------------------------------------------------------------

def _ekey(e: dict) -> str:
    return f"{float(e['board_t']):.2f}"


def colour_teams(events: list[dict], reads: dict) -> dict | None:
    """``{"home": 0|1, "guest": 1|0}``: team colour index per team, by vote."""
    tally = [0, 0]  # [home=A, home=B]
    for e in events:
        r = reads.get(_ekey(e)) or {}
        c = r.get("team")
        if c is None or e.get("points", 0) <= 0:
            continue
        tally[int((c == 0) != (e["team"] == "home"))] += 1
    if not any(tally):
        return None
    h = int(tally[1] > tally[0])
    return {"home": h, "guest": 1 - h, "votes": tally}


def pick_number(read: dict, num: dict | None, edit: dict | None) -> dict:
    """Final jersey number: manual edit > stage-3 read > stage-2 guess.

    ``number_src``: edit | read | shot | read_low (shown only as a hint);
    ``number_check``: the two automatic reads disagree.
    """
    if edit is not None and "number" in edit:
        return {"number": edit["number"], "number_src": "edit"}
    guess = read.get("number")
    n3 = (num or {}).get("number")
    c3 = (num or {}).get("confidence")
    if n3 is not None and c3 in ("high", "medium"):
        out = {"number": n3, "number_src": "read"}
        if guess is not None and guess != n3:
            out["number_check"] = True
        return out
    if guess is not None:
        out = {"number": guess, "number_src": "shot"}
        if n3 is not None and n3 != guess:
            out["number_check"] = True
        return out
    if n3 is not None:
        return {"number": n3, "number_src": "read_low"}
    return {"number": None, "number_src": None}


def apply_shots(result: dict, doc: dict | None, edits: dict | None = None) -> dict:
    """Merge ``shots.json`` into ``result`` events (in place); returns it.

    Adds per event ``shot`` (time, shooter colour, agreement with the board,
    crop file, jersey-number guess) and ``shot_t``; ``result["colours"]``
    maps home/guest to the team colours.
    """
    if not doc or not doc.get("reads"):
        return result
    reads = doc["reads"]
    colours = doc.get("colours")
    mapping = colour_teams(result["events"], reads) if colours else None
    if mapping and colours:
        result["colours"] = {
            s: {"rgb": colours["teams"][mapping[s]]["rgb"],
                "name": colours["teams"][mapping[s]]["name"]["pl"]}
            for s in ("home", "guest")
        }
    for e in result["events"]:
        r = reads.get(_ekey(e))
        if not r or r.get("shot_t") is None or e.get("points", 0) <= 0:
            continue
        shot = {"t": r["shot_t"], "confidence": r.get("confidence"),
                "number_guess": r.get("number"), "crop": r.get("crop"),
                "grid": r.get("grid")}
        shot.update(pick_number(r, (doc.get("numbers") or {}).get(_ekey(e)),
                                (edits or {}).get(_ekey(e))))
        c = r.get("team")
        if c is not None and colours:
            shot["colour"] = colours["teams"][c]["name"]["pl"]
            if mapping:
                shot["colour_ok"] = mapping[e["team"]] == c
        e["shot"] = shot
        e["shot_t"] = r["shot_t"]
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_doc(path: Path) -> dict:
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("version") == SHOTS_VERSION:
                return d
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"version": SHOTS_VERSION, "reads": {}}


def find_shots(
    config: dict, sc_cfg: dict, video_path: str | Path, cal: dict,
    events: list[dict], width: int, height: int, fps: float,
    detections_path: str | Path, out_dir: str | Path,
    reader: ShotFn | None = None, players: Players | None = None,
) -> dict | None:
    """Run stage 2 for all score events; writes ``shots.json`` + crops."""
    cfg = shots_config(sc_cfg)
    out_dir = Path(out_dir)
    if len(cal.get("hoops") or []) != 2:
        logger.warning("Shooters skipped: need two calibrated hoops")
        return None
    if players is None:
        det_w, det_h = det_resolution(config)
        players = Players.load(detections_path, det_w, det_h,
                               player_class=int(cfg["player_class"]),
                               min_conf=float(cfg["min_conf"]))
    sx, sy = width / players.det_w, height / players.det_h
    doc_path = out_dir / "shots.json"
    doc = _load_doc(doc_path)
    reads: dict = doc["reads"]
    crops_dir = out_dir / "shots"
    crops_dir.mkdir(parents=True, exist_ok=True)

    evs = [e for e in events if e.get("points", 0) > 0]
    votes = [play_hoop(players, cal, e["board_t"] - 8, e["board_t"] - 2, fps) for e in evs]
    amap = attack_map(evs, votes)
    if amap is None:
        logger.warning("Shooters skipped: no players near the baskets")
        return None

    step = float(cfg["step_sec"])
    every = max(1, int(round(step * fps)))
    n = max(2, int(round(float(cfg["window_sec"]) / step)))

    def clip(e):
        """Frames, per-frame boxes (full-res crop coords) and the crop box."""
        hoops = _hoops_at(cal, e["board_t"])
        k = amap[(e["team"], _half(e.get("period")))]
        box = view_box(hoops[k], hoops[1 - k], width, height, cal)
        f0 = int(round((e["board_t"] - n * step) * fps))
        frames = grab_frames(video_path, f0 / fps, n, every, box)
        x0, y0, x1, y1 = box
        per = []
        for i in range(len(frames)):
            b = players.at(f0 + i * every).copy()
            b[:, [0, 2]] = b[:, [0, 2]] * sx - x0
            b[:, [1, 3]] = b[:, [1, 3]] * sy - y0
            inside = (b[:, 2] > 0) & (b[:, 0] < x1 - x0) & (b[:, 3] > 0) & (b[:, 1] < y1 - y0)
            b = b[inside]
            b = b[np.argsort(b[:, 0])][: len(LETTERS)]
            per.append(b)
        return frames, per, box, f0

    # -- pass 1: shirt colours (once per video), then one read per event ------
    need = [e for e in evs if _ekey(e) not in reads]
    llm: list = [reader]

    def read(frames, prompt):  # LLM client only when something is asked
        if llm[0] is None:
            llm[0] = OpenAIShotReader(cfg["llm"], max_w=int(cfg["max_w"]))
        return llm[0](frames, prompt)

    feats: list[np.ndarray] = []

    def ask(e, frames, per, box, f0, colours, hint=None):
        prompt = shot_prompt(len(frames), step, e["points"], colours, hint)
        shown = [draw_boxes(f, b) for f, b in zip(frames, per)]
        try:
            ans = read(shown, prompt)
        except Exception as exc:  # network etc.: leave the event unread
            logger.warning("Shot read failed at %.0fs: %s", e["board_t"], exc)
            return None
        r = parse_shot_answer(ans, len(frames), [len(b) for b in per])
        r["pass"] = 2 if hint is not None else 1
        if r["frame"] is not None:
            i = r["frame"]
            r["shot_t"] = round((f0 + i * every) / fps, 2)
            r["view"] = [int(v) for v in box]
            bx = None
            if r["box"] is not None:
                bx = per[i][r["box"]]
            elif r["xy"] is not None and len(per[i]):
                h, w = frames[i].shape[:2]
                cx, cy = r["xy"][0] * w, r["xy"][1] * h
                d = [np.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy) for b in per[i]]
                if min(d) < 0.06 * w:  # the LLM's point is next to a detected person
                    bx = per[i][int(np.argmin(d))]
            if bx is not None:
                x0, y0 = box[0], box[1]
                r["shooter_box"] = [round(float(bx[0] + x0)), round(float(bx[1] + y0)),
                                    round(float(bx[2] + x0)), round(float(bx[3] + y0))]
                lab = torso_lab(frames[i], bx)
                r["box_lab"] = None if lab is None else [round(float(v), 1) for v in lab]
            _crop_shot(e, r, frames[i], crops_dir)
        return r

    if doc.get("colours") is None:
        logger.info("Shooters: sampling shirt colours from %d events", len(evs))
        for e in evs:
            frames, per, box, f0 = clip(e)
            for f, bs in zip(frames, per):
                for b in bs:
                    lab = torso_lab(f, b)
                    if lab is not None:
                        feats.append(lab)
        doc["colours"] = cluster_colours(np.array(feats, np.float32)) if feats else None
        if doc["colours"]:
            logger.info("Team colours: %s",
                        ", ".join(f"{c['name']['pl']} {c['rgb']}" for c in doc["colours"]["teams"]))
    colours = doc.get("colours")

    def work(e, hint=None):
        frames, per, box, f0 = clip(e)
        return e, ask(e, frames, per, box, f0, colours, hint)

    def run_all(items):
        with ThreadPoolExecutor(max_workers=max(1, int(cfg["workers"]))) as ex:
            for e, r in ex.map(lambda a: work(*a), items):
                if r is not None:
                    reads[_ekey(e)] = r
                    doc_path.write_text(json.dumps(doc, indent=1))

    if need:
        logger.info("Shooters: reading %d events", len(need))
        run_all([(e, None) for e in need])

    # -- pass 2: shooter colour disagrees with the board -> ask with a hint ---
    if cfg["verify"] and colours:
        mapping = colour_teams(evs, reads)
        if mapping:
            redo = [e for e in evs
                    if (reads.get(_ekey(e)) or {}).get("pass") == 1
                    and (reads[_ekey(e)].get("team") != mapping[e["team"]])]
            if redo:
                logger.info("Shooters: re-reading %d events with the team hint", len(redo))
                run_all([(e, mapping[e["team"]]) for e in redo])
    _ensure_crops(evs, reads, video_path, crops_dir, fps)
    doc_path.write_text(json.dumps(doc, indent=1))
    if isinstance(llm[0], OpenAIShotReader):
        logger.info("Shooters LLM: %s", llm[0].summary())
    from .jerseys import jerseys_config, read_numbers
    jc = jerseys_config(sc_cfg)
    if jc["enabled"]:
        nllm: list = [reader]

        def nread(frames, prompt):
            if nllm[0] is None:
                nllm[0] = OpenAIShotReader(jc["llm"], max_w=2000)
            return nllm[0](frames, prompt)

        read_numbers(sc_cfg, video_path, evs, doc, players, width, height, fps, out_dir, nread)
        if isinstance(nllm[0], OpenAIShotReader):
            logger.info("Jersey numbers LLM: %s", nllm[0].summary())
    return doc


def shot_region(r: dict) -> tuple[float, float, float, float] | None:
    """Full-res region to cut for the shooter of a read.

    A matched detection box (plus margin), else a generous window around the
    LLM's point (shooters without a box are mostly free-throw shooters the
    detector merged or missed; the LLM point is only roughly right).
    """
    if r.get("shooter_box"):
        x0, y0, x1, y1 = r["shooter_box"]
        w, h = x1 - x0, y1 - y0
        return x0 - 0.35 * w, y0 - 0.18 * h, x1 + 0.35 * w, y1 + 0.18 * h
    view, xy = r.get("view"), r.get("xy")
    if not view or not xy:
        return None
    vx0, vy0, vx1, vy1 = view
    vw, vh = vx1 - vx0, vy1 - vy0
    cx, cy = vx0 + xy[0] * vw, vy0 + xy[1] * vh
    return cx - 0.12 * vw, cy - 0.28 * vh, cx + 0.12 * vw, cy + 0.28 * vh


def _crop_shot(e: dict, r: dict, view_img: np.ndarray, crops_dir: Path) -> None:
    """Save the shooter crop from the frame cut to ``r["view"]``."""
    reg = shot_region(r)
    if reg is None:
        return
    vx0, vy0 = r["view"][0], r["view"][1]
    H, W = view_img.shape[:2]
    a, b = int(max(0, reg[0] - vx0)), int(max(0, reg[1] - vy0))
    c, d = int(min(W, reg[2] - vx0)), int(min(H, reg[3] - vy0))
    if c - a < 4 or d - b < 4:
        return
    crop = np.ascontiguousarray(view_img[b:d, a:c])
    scale = 240.0 / crop.shape[0]
    if scale > 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), 240), interpolation=cv2.INTER_CUBIC)
    name = f"{e['board_t']:07.1f}.jpg"
    if cv2.imwrite(str(crops_dir / name), crop[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90]):
        r["crop"] = f"shots/{name}"


def _ensure_crops(evs: list[dict], reads: dict, video_path, crops_dir: Path,
                  fps: float) -> None:
    """Re-cut missing shooter crops from the video (no LLM calls)."""
    for e in evs:
        r = reads.get(_ekey(e))
        if not r or r.get("shot_t") is None or not r.get("view"):
            continue
        if r.get("crop") and (crops_dir.parent / r["crop"]).is_file():
            continue
        try:
            fr = grab_frames(video_path, round(r["shot_t"] * fps) / fps, 1, 1,
                             tuple(r["view"]))
        except subprocess.CalledProcessError:
            continue
        if fr:
            r.pop("crop", None)
            _crop_shot(e, r, fr[0], crops_dir)

