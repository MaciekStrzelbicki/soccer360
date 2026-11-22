"""Venue calibration: where are the scoreboard and the baskets in the 360 frame.

Sources, highest priority first:

1. manual   -- ``<calibration.dir>/<video-stem>.json`` saved from the dashboard
               calibration page (or edited by hand)
2. llm      -- a vision LLM (OpenAI) looks at a gridded equirect frame and
               names approximate yaw/pitch; a second zoomed pass refines it
3. auto     -- local heuristics: scoreboard = lit pixels that change every
               second (game clock); baskets = ends of the player-cluster range
               with height from player size
4. config   -- static values already in camera.keep_in_frame /
               reframer.scoreboard_overlay

Local scoreboard refinement (temporal clock search) is also applied around an
LLM estimate, since LLMs name objects well but place them only roughly.

Calibration dict format (angles in degrees, equirect convention):

    {
      "source": "manual|llm|auto|config",
      "scoreboard": {"yaw", "pitch", "width_deg", "height_deg"} | None,
      "hoops": [{"yaw", "rim_pitch", "board_top_pitch"}, ...],
      "video_path": "...",          # manual files only
    }
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("soccer360.calibration")

# Assumed real-world sizes used to turn player bbox height into hoop height.
_PLAYER_HEIGHT_M = 1.55   # mixed youth/adult; lower = hoop estimated higher (safer)
_RIM_HEIGHT_M = 3.05
_BOARD_TOP_M = 3.95


# ---------------------------------------------------------------------------
# Config / persistence
# ---------------------------------------------------------------------------

def calibration_config(config: dict) -> dict:
    cal = dict(config.get("calibration", {}) or {})
    cal.setdefault("enabled", False)
    cal.setdefault("dir", "./data/state/calibrations")
    cal.setdefault("sample_time_frac", 0.5)
    cal.setdefault("refine_scoreboard", True)
    cal.setdefault("hoops_from_players", True)
    cal.setdefault("hoop_margin_deg", 3.0)
    llm = dict(cal.get("llm", {}) or {})
    llm.setdefault("enabled", False)
    llm.setdefault("provider", "openai")
    llm.setdefault("model", "gpt-5")
    llm.setdefault("api_key_env", "OPENAI_API_KEY")
    llm.setdefault("timeout_sec", 120)
    cal["llm"] = llm
    return cal


def manual_path(config: dict, video_path: str | Path) -> Path:
    return Path(calibration_config(config)["dir"]) / f"{Path(video_path).stem}.json"


def load_manual(config: dict, video_path: str | Path) -> dict | None:
    p = manual_path(config, video_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable calibration %s: %s", p, exc)
        return None
    if not data.get("scoreboard") and not data.get("hoops"):
        return None  # registered for the UI but not calibrated yet
    data["source"] = "manual"
    return data


def save_manual(config: dict, video_path: str | Path, data: dict) -> Path:
    p = manual_path(config, video_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "source": "manual",
        "video_path": str(Path(video_path).resolve()),
        "scoreboard": data.get("scoreboard"),
        "hoops": data.get("hoops") or [],
    }
    p.write_text(json.dumps(out, indent=2))
    return p


def register_video(config: dict, video_path: str | Path) -> Path:
    """Remember a video path so the dashboard may show its frames."""
    p = manual_path(config, video_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            data = {}
    data["video_path"] = str(Path(video_path).resolve())
    data.setdefault("scoreboard", None)
    data.setdefault("hoops", [])
    p.write_text(json.dumps(data, indent=2))
    return p


# ---------------------------------------------------------------------------
# Frame access
# ---------------------------------------------------------------------------

def grab_equirect(video_path: str | Path, t_sec: float, width: int = 3840) -> np.ndarray:
    """One RGB equirect frame at ``t_sec`` scaled to width x width/2."""
    h = width // 2
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, t_sec):.3f}",
        "-i", str(video_path), "-frames:v", "1",
        "-vf", f"scale={width}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if len(raw) < width * h * 3:
        raise RuntimeError(f"Could not read frame at {t_sec:.1f}s from {video_path}")
    return np.frombuffer(raw[: width * h * 3], np.uint8).reshape(h, width, 3).copy()


def _burst(video_path: str | Path, t_sec: float, n: int, fps: float, width: int):
    h = width // 2
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, t_sec):.3f}",
        "-i", str(video_path), "-t", f"{n / fps + 0.05:.3f}",
        "-vf", f"fps={fps},scale={width}:{h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    k = len(raw) // (width * h * 3)
    return np.frombuffer(raw[: k * width * h * 3], np.uint8).reshape(k, h, width, 3)


def yaw_to_x(yaw: float, w: int) -> float:
    return ((yaw + 180.0) % 360.0) / 360.0 * w


def pitch_to_y(pitch: float, h: int) -> float:
    return (90.0 - pitch) / 180.0 * h


def x_to_yaw(x: float, w: int) -> float:
    return x / w * 360.0 - 180.0


def y_to_pitch(y: float, h: int) -> float:
    return 90.0 - y / h * 180.0


# ---------------------------------------------------------------------------
# Scoreboard: temporal search for the running game clock
# ---------------------------------------------------------------------------

def clock_score_from_frames(frames: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate (lit-count, changed-at-lag count) for a stack of RGB frames."""
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames])
    lit = (hsv[..., 2] > 150) & (hsv[..., 1] > 110)
    pres = lit.sum(0).astype(np.float32)
    chg = np.zeros_like(pres)
    for i in range(len(lit) - lag):
        chg += lit[i] ^ lit[i + lag]
    return pres, chg


def scoreboard_clock_score(
    video_path: str | Path,
    duration_sec: float,
    *,
    yaw_range: tuple[float, float] = (-180.0, 180.0),
    pitch_range: tuple[float, float] = (-20.0, 45.0),
    bursts: int = 12,
    width: int = 3840,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Per-pixel score: LED-lit often AND changing at a 1 s lag (clock digits).

    Returns (score, (x0, y0, x1, y1)) where the box is the searched region in
    the ``width`` x ``width/2`` equirect image.
    """
    h = width // 2
    y0 = int(max(0, pitch_to_y(pitch_range[1], h)))
    y1 = int(min(h, pitch_to_y(pitch_range[0], h)))
    x0 = int(max(0, yaw_to_x(yaw_range[0], width))) if yaw_range[0] > -180 else 0
    x1 = int(min(width, yaw_to_x(yaw_range[1], width))) if yaw_range[1] < 180 else width
    if x1 <= x0:  # window across the seam: search everything
        x0, x1 = 0, width

    fps, n, lag = 5.0, 11, 5  # 5 fps bursts, compare frames 1.0 s apart
    pres = chg = None
    nfr = npairs = 0
    span = max(duration_sec - 6.0, 1.0)
    for k in range(bursts):
        t = 3.0 + k * span / bursts
        try:
            fr = _burst(video_path, t, n, fps, width)
        except subprocess.CalledProcessError:
            continue
        if len(fr) <= lag:
            continue
        p, c = clock_score_from_frames(fr[:, y0:y1, x0:x1], lag)
        pres = p if pres is None else pres + p
        chg = c if chg is None else chg + c
        nfr += len(fr)
        npairs += len(fr) - lag
    if pres is None or npairs == 0:
        return np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.float32), (x0, y0, x1, y1)
    score = np.minimum(pres / nfr, chg / npairs)
    score = cv2.GaussianBlur(score, (0, 0), 3)
    return score, (x0, y0, x1, y1)


def pick_scoreboard_peak(score, width, candidates=30):
    """Best clock-score peak, weighted by compactness.

    A scoreboard is a compact island of activity; moving jerseys or a
    fidgeting crowd produce peaks inside widespread activity. Compactness =
    share of above-threshold pixels near the peak (8x4 deg) vs. around it
    (30x12 deg).
    """
    px = width / 360.0
    s = score.copy()
    best, best_val = None, 0.0
    for _ in range(candidates):
        yy, xx = np.unravel_index(int(np.argmax(s)), s.shape)
        val = float(s[yy, xx])
        if val <= 0:
            break
        m = score > 0.35 * val

        def win(hy, hx):
            return m[max(0, int(yy - hy * px)):int(yy + hy * px),
                     max(0, int(xx - hx * px)):int(xx + hx * px)].sum()

        compact = win(2, 4) / max(win(6, 15), 1)
        rank = val * compact ** 2
        if rank > best_val:
            best, best_val = (yy, xx, val), rank
        s[max(0, int(yy - 3 * px)):int(yy + 3 * px),
          max(0, int(xx - 6 * px)):int(xx + 6 * px)] = 0
    return best


def score_to_scoreboard(score, box, width, min_score=0.08):
    """Best clock-score peak -> scoreboard centre/size (degrees)."""
    if score.size == 0 or float(score.max()) < min_score:
        return None
    h = width // 2
    picked = pick_scoreboard_peak(score, width)
    if picked is None:
        return None
    yy, xx, peak = picked
    # only look at a window around the peak (a scoreboard is < ~20 x 8 deg)
    px = width / 360.0
    wy0, wx0 = max(0, int(yy - 5 * px)), max(0, int(xx - 12 * px))
    sub = score[wy0:int(yy + 5 * px), wx0:int(xx + 12 * px)]
    yy, xx = yy - wy0, xx - wx0
    box = (box[0] + wx0, box[1] + wy0)
    score = sub
    # digits blob: pixels above 35% of the peak connected to it
    mask = (score > 0.35 * peak).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    lab = labels[yy, xx]
    x, y, bw, bh = (int(v) for v in stats[lab, :4])
    # merge nearby blobs on the same row (home score / clock / guest score)
    gx0, gy0, gx1, gy1 = x, y, x + bw, y + bh
    reach = 3 * max(gy1 - gy0, 8)
    for i in range(1, n):
        if i == lab:
            continue
        sx, sy, sw, sh, area = (int(v) for v in stats[i])
        if area < 6:
            continue
        same_row = abs((sy + sh / 2) - (gy0 + gy1) / 2) < 2.5 * max(gy1 - gy0, 8)
        close = sx < gx1 + reach and sx + sw > gx0 - reach
        if same_row and close:
            gx0, gy0 = min(gx0, sx), min(gy0, sy)
            gx1, gy1 = max(gx1, sx + sw), max(gy1, sy + sh)
    cx = box[0] + (gx0 + gx1) / 2
    cy = box[1] + (gy0 + gy1) / 2
    deg_per_px = 360.0 / width
    w_deg = max((gx1 - gx0) * deg_per_px * 1.5, 3.0)
    h_deg = max((gy1 - gy0) * deg_per_px * 2.2, 1.8)
    return {
        "yaw": round(x_to_yaw(cx, width), 2),
        "pitch": round(y_to_pitch(cy, h), 2),
        "width_deg": round(w_deg, 2),
        "height_deg": round(h_deg, 2),
        "score": round(peak, 3),
    }


def detect_scoreboard_auto(
    video_path, duration_sec, *, prior: dict | None = None, width: int = 3840,
    bursts: int = 12,
) -> dict | None:
    """Find the scoreboard; with ``prior`` search only a window around it."""
    if prior:
        win_y = max(12.0, 1.5 * float(prior.get("width_deg", 8.0)))
        win_p = max(8.0, 1.5 * float(prior.get("height_deg", 5.0)))
        yr = (prior["yaw"] - win_y, prior["yaw"] + win_y)
        pr = (prior["pitch"] - win_p, prior["pitch"] + win_p)
    else:
        yr, pr = (-180.0, 180.0), (-20.0, 45.0)
    score, box = scoreboard_clock_score(
        video_path, duration_sec, yaw_range=yr, pitch_range=pr,
        bursts=bursts, width=width,
    )
    return score_to_scoreboard(score, box, width)


# ---------------------------------------------------------------------------
# Hoops from player detections (fallback without LLM)
# ---------------------------------------------------------------------------

def estimate_hoops_from_players(
    detections_path: str | Path,
    player_cluster_path: str | Path | None,
    det_w: int,
    det_h: int,
    *,
    player_class: int = 0,
    min_conf: float = 0.5,
) -> list[dict]:
    """Rough basket positions: ends of the play range + player-size height."""
    if player_cluster_path is None or not Path(player_cluster_path).exists():
        return []
    clusters = json.loads(Path(player_cluster_path).read_text())
    cyaw = np.array([
        x_to_yaw(c["cluster"]["x"], det_w) for c in clusters if c.get("cluster")
    ])
    if len(cyaw) < 100:
        return []
    ends = [float(np.percentile(cyaw, 2)), float(np.percentile(cyaw, 98))]
    if ends[1] - ends[0] < 40:  # play never spans the court
        return []

    yaws, feet, hts = [], [], []
    with open(detections_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("class_id") != player_class or d.get("conf", 0) < min_conf:
                continue
            x0, y0, x1, y1 = d["bbox_xyxy"]
            yaws.append(x_to_yaw((x0 + x1) / 2, det_w))
            feet.append(y_to_pitch(y1, det_h))
            hts.append((y1 - y0) / det_h * 180.0)
    if not yaws:
        return []
    yaws, feet, hts = map(np.array, (yaws, feet, hts))

    hoops = []
    for end in ends:
        near = np.abs(yaws - end) < 8.0
        if near.sum() < 50:
            continue
        fp = float(np.median(feet[near]))
        ph = float(np.median(hts[near]))
        hoops.append({
            "yaw": round(end, 2),
            "rim_pitch": round(fp + ph * _RIM_HEIGHT_M / _PLAYER_HEIGHT_M, 2),
            "board_top_pitch": round(fp + ph * _BOARD_TOP_M / _PLAYER_HEIGHT_M, 2),
        })
    return hoops


# ---------------------------------------------------------------------------
# Vision LLM (OpenAI)
# ---------------------------------------------------------------------------

def draw_grid(
    img: np.ndarray,
    yaw_range: tuple[float, float],
    pitch_range: tuple[float, float],
    step: float,
) -> np.ndarray:
    """Label a (cropped) equirect image with a yaw/pitch degree grid."""
    out = img.copy()
    h, w = out.shape[:2]
    (ya, yb), (pa, pb) = yaw_range, pitch_range
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, w / 2400)
    thick = max(1, int(round(scale * 2)))

    def X(yaw):
        return int(round((yaw - ya) / (yb - ya) * (w - 1)))

    def Y(p):
        return int(round((pb - p) / (pb - pa) * (h - 1)))

    v = math.ceil(ya / step) * step
    while v <= yb + 1e-6:
        x = X(v)
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 255), 1)
        lab = f"{v:g}"
        for yy in (int(22 * scale) + 4, h - 6):
            cv2.putText(out, lab, (x + 3, yy), font, scale, (0, 0, 0), thick + 2)
            cv2.putText(out, lab, (x + 3, yy), font, scale, (0, 255, 255), thick)
        v += step
    v = math.ceil(pa / step) * step
    while v <= pb + 1e-6:
        y = Y(v)
        cv2.line(out, (0, y), (w - 1, y), (255, 255, 0), 1)
        lab = f"{v:g}"
        cv2.putText(out, lab, (3, y - 3), font, scale, (0, 0, 0), thick + 2)
        cv2.putText(out, lab, (3, y - 3), font, scale, (255, 255, 0), thick)
        v += step
    return out


def crop_equirect(frame, yaw_range, pitch_range):
    """Crop a yaw/pitch window; returns (crop, effective yaw_range)."""
    h, w = frame.shape[:2]
    x0 = int(yaw_to_x(yaw_range[0], w)) if yaw_range[0] > -180 else 0
    x1 = int(yaw_to_x(yaw_range[1], w)) if yaw_range[1] < 180 else w
    y0 = int(pitch_to_y(pitch_range[1], h))
    y1 = int(pitch_to_y(pitch_range[0], h))
    if x1 <= x0:
        x0, x1 = 0, w
        yaw_range = (-180.0, 180.0)
    return frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)], yaw_range


def _jpeg_b64(img_rgb: np.ndarray, max_w: int = 2048) -> str:
    h, w = img_rgb.shape[:2]
    if w > max_w:
        img_rgb = cv2.resize(img_rgb, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


_OVERVIEW_PROMPT = """\
This is part of an equirectangular 360-degree frame from a camera mounted in \
an indoor sports hall during a basketball game. Yellow vertical lines are YAW \
(horizontal angle in degrees, labelled at top and bottom); cyan horizontal \
lines are PITCH (vertical angle in degrees, labelled on the left; up is positive).

Find:
1. The electronic SCOREBOARD showing the game score and clock (usually LED \
digits on a dark panel on a wall). Not banners, not windows.
2. Every BASKETBALL HOOP used for the game being played (backboard + rim). \
Ignore hoops folded up to the ceiling and unused side hoops when the game is \
clearly played towards other hoops.

Read the angles from the grid. Respond with JSON only:
{"scoreboard": {"yaw": number, "pitch": number, "width_deg": number, \
"height_deg": number} or null,
 "hoops": [{"yaw": number, "rim_pitch": number, "board_top_pitch": number}],
 "notes": "short"}"""

_ZOOM_PROMPT = """\
Zoomed crop of the same equirectangular frame with a finer degree grid \
(yellow vertical = yaw, cyan horizontal = pitch). It should contain {what}. \
Give its precise position read from the grid. JSON only:
{schema}
or {{"found": false}} if it is not visible."""


class OpenAILocator:
    """Ask an OpenAI vision model where the scoreboard and hoops are."""

    def __init__(self, llm_cfg: dict):
        from openai import OpenAI  # optional dependency

        env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"{env} is not set")
        self.model = llm_cfg.get("model", "gpt-5")
        self.client = OpenAI(api_key=key, timeout=float(llm_cfg.get("timeout_sec", 120)))

    def _ask(self, prompt: str, img_rgb: np.ndarray) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{_jpeg_b64(img_rgb)}",
                        "detail": "high",
                    }},
                ],
            }],
        )
        return json.loads(resp.choices[0].message.content or "{}")

    def locate(self, frame: np.ndarray, yaw_range=(-180.0, 180.0)) -> dict:
        yr, pr = tuple(yaw_range), (-45.0, 45.0)
        crop, yr = crop_equirect(frame, yr, pr)
        overview = self._ask(_OVERVIEW_PROMPT, draw_grid(crop, yr, pr, 10.0))
        logger.info("LLM overview: %s", overview)

        result = {"scoreboard": None, "hoops": []}
        sb = overview.get("scoreboard")
        if isinstance(sb, dict) and "yaw" in sb and "pitch" in sb:
            result["scoreboard"] = self._refine(
                frame, sb, "the electronic scoreboard (score + game clock)",
                '{"found": true, "yaw": number, "pitch": number, '
                '"width_deg": number, "height_deg": number}',
                half=(12.0, 8.0),
            ) or sb
        for hp in overview.get("hoops") or []:
            if not isinstance(hp, dict) or "yaw" not in hp:
                continue
            seed = {
                "yaw": hp["yaw"],
                "pitch": hp.get("rim_pitch", hp.get("board_top_pitch", 0.0)),
            }
            fine = self._refine(
                frame, seed, "a basketball hoop (backboard and rim)",
                '{"found": true, "yaw": number, "rim_pitch": number, '
                '"board_top_pitch": number}',
                half=(15.0, 10.0),
            )
            result["hoops"].append(fine or hp)
        return result

    def _refine(self, frame, seed, what, schema, half):
        try:
            yr = (float(seed["yaw"]) - half[0], float(seed["yaw"]) + half[0])
            pr = (float(seed["pitch"]) - half[1], float(seed["pitch"]) + half[1])
            crop, yr = crop_equirect(frame, yr, pr)
            if crop.size == 0:
                return None
            big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                             interpolation=cv2.INTER_CUBIC)
            ans = self._ask(
                _ZOOM_PROMPT.format(what=what, schema=schema),
                draw_grid(big, yr, pr, 2.0),
            )
            logger.info("LLM zoom (%s): %s", what, ans)
            if not ans.get("found", True):
                return None
            ans.pop("found", None)
            # answers outside the crop are made up -> ignore
            try:
                y = float(ans["yaw"])
                ps = [float(ans[k]) for k in ("pitch", "rim_pitch", "board_top_pitch") if k in ans]
            except (KeyError, TypeError, ValueError):
                return None
            if not (yr[0] - 1 <= y <= yr[1] + 1) or any(
                not (pr[0] - 1 <= v <= pr[1] + 1) for v in ps
            ):
                logger.warning("LLM zoom answer outside window, ignored: %s", ans)
                return None
            return ans
        except Exception as exc:  # noqa: BLE001 - refinement is best-effort
            logger.warning("LLM refine failed for %s: %s", what, exc)
            return None


def clean_scoreboard(sb: dict | None) -> dict | None:
    if not sb:
        return None
    try:
        return {
            "yaw": round(float(sb["yaw"]), 2),
            "pitch": round(float(sb["pitch"]), 2),
            "width_deg": round(float(sb.get("width_deg") or 8.0), 2),
            "height_deg": round(float(sb.get("height_deg") or 4.0), 2),
        }
    except (KeyError, TypeError, ValueError):
        return None


def clean_hoop(hp: dict | None) -> dict | None:
    if not hp:
        return None
    try:
        rim = float(hp.get("rim_pitch", hp.get("board_top_pitch")))
        top = float(hp.get("board_top_pitch", rim + 3.0))
        return {
            "yaw": round(float(hp["yaw"]), 2),
            "rim_pitch": round(rim, 2),
            "board_top_pitch": round(max(top, rim), 2),
        }
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _foi_yaw_range(config: dict) -> tuple[float, float]:
    """Yaw window to show the LLM: the field of interest (+ margin)."""
    foi = config.get("field_of_interest", {}) or {}
    if not foi.get("enabled") or foi.get("center_mode", "fixed") != "fixed":
        return (-180.0, 180.0)
    c = float(foi.get("center_yaw_deg", 0.0))
    half = float(foi.get("yaw_window_deg", 360.0)) / 2 + 10.0
    if half >= 180:
        return (-180.0, 180.0)
    lo, hi = c - half, c + half
    if lo < -180 or hi > 180:  # crosses the seam: keep it simple
        return (-180.0, 180.0)
    return (lo, hi)


def merge_hoops(llm_hoops: list[dict], player_hoops: list[dict],
                max_dyaw: float = 12.0) -> list[dict]:
    """Combine LLM hoops with the player-geometry estimate.

    The player estimate says *which* ends of the court are used (robust);
    an LLM hoop near such an end gives the more exact yaw. Backboard height
    takes the higher of both (safer: more headroom). LLM hoops that match no
    end of play are dropped.
    """
    out = []
    for ph in player_hoops:
        near = [h for h in llm_hoops if abs(h["yaw"] - ph["yaw"]) <= max_dyaw]
        if not near:
            out.append(dict(ph))
            continue
        lh = min(near, key=lambda h: abs(h["yaw"] - ph["yaw"]))
        out.append({
            "yaw": lh["yaw"],
            "rim_pitch": round(max(lh["rim_pitch"], ph["rim_pitch"]), 2),
            "board_top_pitch": round(max(lh["board_top_pitch"], ph["board_top_pitch"]), 2),
        })
    return out


def from_static_config(config: dict) -> dict:
    """Calibration implied by hand-written camera/reframer config values."""
    out = {"source": "config", "scoreboard": None, "hoops": []}
    ov = config.get("reframer", {}).get("scoreboard_overlay") or {}
    if "yaw" in ov and "pitch" in ov:
        fov_h = float(ov.get("fov_h", 7.0))
        out["scoreboard"] = {
            "yaw": float(ov["yaw"]), "pitch": float(ov["pitch"]),
            "width_deg": round(fov_h / 1.1, 2),
            "height_deg": round(fov_h / float(ov.get("aspect", 1.9)) / 1.1, 2),
        }
    for pt in config.get("camera", {}).get("keep_in_frame") or []:
        out["hoops"].append({
            "yaw": float(pt["yaw"]), "rim_pitch": float(pt["pitch"]) - 3.0,
            "board_top_pitch": float(pt["pitch"]),
        })
    return out


def resolve_calibration(
    config: dict,
    video_path: str | Path,
    duration_sec: float,
    *,
    detections_path: str | Path | None = None,
    player_cluster_path: str | Path | None = None,
    det_size: tuple[int, int] | None = None,
    locator=None,
) -> dict:
    """Pick the best available calibration (see module docstring)."""
    cal_cfg = calibration_config(config)
    static = from_static_config(config)
    if not cal_cfg["enabled"]:
        return static

    manual = load_manual(config, video_path)
    if manual:
        logger.info("Calibration: manual (%s)", manual_path(config, video_path))
        return manual

    result = {"source": "auto", "scoreboard": None, "hoops": []}
    llm_cfg = cal_cfg["llm"]
    if locator is None and llm_cfg["enabled"]:
        try:
            locator = OpenAILocator(llm_cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM calibration unavailable: %s", exc)
    if locator is not None:
        try:
            frame = grab_equirect(video_path, duration_sec * cal_cfg["sample_time_frac"])
            try:
                found = locator.locate(frame, yaw_range=_foi_yaw_range(config))
            except TypeError:  # simple locators without yaw_range
                found = locator.locate(frame)
            result["scoreboard"] = clean_scoreboard(found.get("scoreboard"))
            result["hoops"] = [h for h in map(clean_hoop, found.get("hoops") or []) if h]
            result["source"] = "llm"
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM calibration failed, falling back: %s", exc)

    if cal_cfg["refine_scoreboard"]:
        try:
            refined = detect_scoreboard_auto(
                video_path, duration_sec, prior=result["scoreboard"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scoreboard clock search failed: %s", exc)
            refined = None
        if refined:
            if result["scoreboard"]:
                logger.info("Scoreboard refined by clock search: %s -> %s",
                            result["scoreboard"], refined)
            result["scoreboard"] = {k: v for k, v in refined.items() if k != "score"}

    if cal_cfg["hoops_from_players"] and detections_path and det_size:
        player_hoops = estimate_hoops_from_players(
            detections_path, player_cluster_path, det_size[0], det_size[1]
        )
        if player_hoops:
            result["hoops"] = merge_hoops(result["hoops"], player_hoops)

    # Anything still missing comes from static config
    if not result["scoreboard"]:
        result["scoreboard"] = static["scoreboard"]
    if not result["hoops"]:
        result["hoops"] = static["hoops"]
    logger.info("Calibration (%s): scoreboard=%s hoops=%s",
                result["source"], result["scoreboard"], result["hoops"])
    return result


def apply_calibration(cal: dict, camera, reframer, config: dict) -> None:
    """Push a calibration into a CameraPathGenerator and a Reframer."""
    margin = calibration_config(config)["hoop_margin_deg"]
    if cal.get("hoops"):
        camera.keep_in_frame = [
            {"yaw": h["yaw"], "pitch": h["board_top_pitch"], "margin_deg": margin}
            for h in cal["hoops"]
        ]
    sb = cal.get("scoreboard")
    if sb and reframer.scoreboard_overlay is not None:
        ov = dict(reframer.scoreboard_overlay)
        ov["yaw"] = sb["yaw"]
        ov["pitch"] = sb["pitch"]
        ov["fov_h"] = round(sb["width_deg"] * 1.1, 2)
        ov["aspect"] = round(
            max(1.0, min(4.0, sb["width_deg"] / max(sb["height_deg"], 0.5))), 3
        )
        reframer.scoreboard_overlay = ov


def preview_image(frame: np.ndarray, cal: dict) -> np.ndarray:
    """Equirect frame with calibration marks drawn (for checking)."""
    out = frame.copy()
    h, w = out.shape[:2]
    lw = max(2, w // 1000)
    sb = cal.get("scoreboard")
    if sb:
        cx, cy = yaw_to_x(sb["yaw"], w), pitch_to_y(sb["pitch"], h)
        hw = sb["width_deg"] / 360 * w / 2
        hh = sb["height_deg"] / 180 * h / 2
        cv2.rectangle(out, (int(cx - hw), int(cy - hh)), (int(cx + hw), int(cy + hh)),
                      (0, 255, 0), lw)
    for hp in cal.get("hoops") or []:
        x = int(yaw_to_x(hp["yaw"], w))
        for key, col in (("rim_pitch", (255, 128, 0)), ("board_top_pitch", (255, 0, 255))):
            y = int(pitch_to_y(hp[key], h))
            cv2.drawMarker(out, (x, y), col, cv2.MARKER_CROSS, max(20, w // 120), lw)
    return out
