"""Jersey numbers of the shooters (scoring stage 3).

Stage 2 (:mod:`src.shots`) finds the shooter's box at the release frame.
A number is rarely readable in one frame, so here the shooter is followed
through the player detections for ``track_sec`` around the shot and up to
``samples`` crops (players not overlapping others, biggest first) are put on
one grid image per basket.  A vision LLM reads the number from the grid,
several baskets per request.  Answers are cached in ``shots.json``
(``numbers``); manual corrections from the dashboard live in ``edits`` and
always win (see :func:`src.shots.apply_shots`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .shots import Players, ShotFn, _ekey, grab_frames, nearest_team_colour, torso_lab

logger = logging.getLogger("soccer360.jerseys")


def jerseys_config(sc_cfg: dict) -> dict:
    c = dict(sc_cfg.get("numbers", {}) or {})
    c.setdefault("enabled", False)
    c.setdefault("track_sec", 2.0)      # follow the shooter +- this around the shot
    c.setdefault("step_frames", 6)      # tracking step (5 Hz at 30 fps)
    c.setdefault("samples", 8)          # crops per basket on the grid
    c.setdefault("per_request", 6)      # baskets per LLM request
    c.setdefault("in_youtube", True)    # "#7 Kowalski" in youtube.txt
    llm = {**(sc_cfg.get("llm") or {}), **(c.get("llm") or {})}
    if "model" not in (c.get("llm") or {}):
        llm["model"] = "gpt-5-mini"     # plain OCR: the small model is enough
        llm.setdefault("reasoning_effort", "low")
    c["llm"] = llm
    return c


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def track_player(players: Players, frame: int, box, span: int, step: int) -> dict[int, np.ndarray]:
    """Follow one detection box from ``frame`` forwards and backwards.

    Returns ``{frame: box}`` (detection space).  Association: best IoU with
    the previous box (>= 0.3), else nearest centre within a third of a box
    height; the box height may not jump (crossing players swap boxes).
    """
    out = {frame: np.asarray(box, np.float32)}
    for direction in (1, -1):
        prev = np.asarray(box, np.float32)
        for k in range(1, span // step + 1):
            f = frame + direction * k * step
            cand = players.at(f)
            if not len(cand):
                break
            ious = [_iou(prev, c) for c in cand]
            j = int(np.argmax(ious))
            if ious[j] < 0.3:
                pc = ((prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2)
                d = [np.hypot((c[0] + c[2]) / 2 - pc[0], (c[1] + c[3]) / 2 - pc[1]) for c in cand]
                j = int(np.argmin(d))
                if d[j] > 0.35 * (prev[3] - prev[1]):
                    break
            ratio = (cand[j][3] - cand[j][1]) / max(1e-6, prev[3] - prev[1])
            if not 0.7 <= ratio <= 1.4:
                break
            prev = cand[j]
            out[f] = prev
    return out


def pick_samples(track: dict[int, np.ndarray], players: Players, n: int) -> list[int]:
    """Frames whose box is free of other players, biggest boxes first,
    spread over the track."""
    good = []
    for f, b in track.items():
        others = [c for c in players.at(f) if not np.allclose(c, b)]
        if any(_iou(b, c) > 0.1 for c in others):
            continue
        good.append((float((b[3] - b[1]) * (b[2] - b[0])), f))
    if not good:  # everyone overlapping: take what there is
        good = [(float((b[3] - b[1]) * (b[2] - b[0])), f) for f, b in track.items()]
    good.sort(reverse=True)
    keep = sorted(f for _, f in good[: 2 * n])
    if len(keep) > n:  # spread over time
        idx = np.linspace(0, len(keep) - 1, n).round().astype(int)
        keep = [keep[i] for i in idx]
    return keep


def make_grid(crops: list[np.ndarray], cell_h: int = 300, cols: int = 4) -> np.ndarray:
    """Crops scaled to one height on a grid with index labels."""
    cells = []
    for i, c in enumerate(crops):
        s = cell_h / max(1, c.shape[0])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), cell_h),
                       interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        c = c[:, : int(cell_h * 0.75)]
        cell = np.zeros((cell_h + 24, int(cell_h * 0.75), 3), np.uint8)
        cell[24:, : c.shape[1]] = c
        cv2.putText(cell, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cells.append(cell)
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


_NUMBER_PROMPT = """\
Each image (#0, #1, ...) belongs to one basket in a youth basketball game and \
shows ONE player (the shooter) several times: numbered crops over a few \
seconds, filmed from far away. Read the shooter's JERSEY NUMBER (front or \
back of the shirt). Crops can show other players at the edges: use the \
player in the centre of each crop, the same person in all crops of an image.

{lines}

Only answer a number you can actually see in at least one crop (digits can \
be blurry; 1-2 digits, 0-99). JSON only:
{{"reads": [{{"image": int, "number": int or null, "confidence": "high" | \
"medium" | "low", "crops": [crop indices where it is visible]}}]}}"""


def number_prompt(shirts: list[str]) -> str:
    lines = "\n".join(f"#{i}: shooter in a {s} shirt" if s else f"#{i}: shooter"
                      for i, s in enumerate(shirts))
    return _NUMBER_PROMPT.format(lines=lines)


def parse_numbers(data: dict, n: int) -> list[dict | None]:
    out: list[dict | None] = [None] * n
    for r in (data or {}).get("reads") or []:
        if not isinstance(r, dict):
            continue
        i = r.get("image")
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < n:
            continue
        num = r.get("number")
        ok = isinstance(num, int) and not isinstance(num, bool) and 0 <= num <= 99
        conf = r.get("confidence") if r.get("confidence") in ("high", "medium", "low") else None
        out[i] = {"number": num if ok else None, "confidence": conf if ok else None}
    return out


def read_numbers(
    sc_cfg: dict, video_path: str | Path, events: list[dict], doc: dict,
    players: Players, width: int, height: int, fps: float, out_dir: Path,
    reader: ShotFn,
) -> int:
    """Stage 3 for all shots without a number read; returns #requests."""
    cfg = jerseys_config(sc_cfg)
    reads = doc.get("reads") or {}
    numbers = doc.setdefault("numbers", {})
    colours = doc.get("colours")
    sx, sy = width / players.det_w, height / players.det_h
    step = int(cfg["step_frames"])
    span = int(round(float(cfg["track_sec"]) * fps))
    todo = [e for e in events
            if (reads.get(_ekey(e)) or {}).get("shot_t") is not None
            and _ekey(e) not in numbers]
    grids: list[tuple[dict, np.ndarray, str]] = []
    calls = 0

    def flush():
        nonlocal calls
        if not grids:
            return
        prompt = number_prompt([s for _, _, s in grids])
        try:
            ans = reader([g for _, g, _ in grids], prompt)
            calls += 1
        except Exception as exc:  # noqa: BLE001 - network/credits: retry next run
            logger.warning("Number read failed: %s", exc)
            grids.clear()
            return
        for (e, _, _), r in zip(grids, parse_numbers(ans, len(grids))):
            numbers[_ekey(e)] = r or {"number": None, "confidence": None}
        (out_dir / "shots.json").write_text(json.dumps(doc, indent=1))
        grids.clear()

    for e in todo:
        r = reads[_ekey(e)]
        grid = shooter_grid(video_path, r, players, sx, sy, fps, span, step,
                            int(cfg["samples"]), colours)
        if grid is None:  # not cached: cheap to retry, no LLM involved
            continue
        name = f"{e['board_t']:07.1f}_n.jpg"
        cv2.imwrite(str(out_dir / "shots" / name), grid[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        r["grid"] = f"shots/{name}"
        shirt = ""
        if colours and r.get("team") is not None:
            shirt = colours["teams"][r["team"]]["name"]["en"]
        grids.append((e, grid, shirt))
        if len(grids) >= int(cfg["per_request"]):
            flush()
    flush()
    (out_dir / "shots.json").write_text(json.dumps(doc, indent=1))
    return calls


def seed_from_point(r: dict, players: Players, sx: float, sy: float, f0: int,
                    span: int, step: int) -> tuple[int, list[float]] | None:
    """Shooter without a box (often a free throw): the detection nearest to
    the LLM's point in the frames around the release, if close enough."""
    view, xy = r.get("view"), r.get("xy")
    if not view or not xy:
        return None
    vw = view[2] - view[0]
    px = (view[0] + xy[0] * vw) / sx
    py = (view[1] + xy[1] * (view[3] - view[1])) / sy
    best = None
    for k in range(0, span // step + 1):
        for f in {f0 - k * step, f0 + k * step}:
            for b in players.at(f):
                d = np.hypot((b[0] + b[2]) / 2 - px, (b[1] + b[3]) / 2 - py)
                if d < 0.06 * vw / sx and (best is None or d < best[0]):
                    best = (d, f, [float(v) for v in b])
        if best is not None:
            return best[1], best[2]
    return None


def shooter_grid(video_path, r: dict, players: Players, sx: float, sy: float,
                 fps: float, span: int, step: int, samples: int,
                 colours: dict | None = None) -> np.ndarray | None:
    """Grid image of the shooter around the shot (None when not found).

    With team ``colours``, crops whose shirt is not the shooter's team
    colour are dropped (the track switched to an opponent)."""
    view = r.get("view")
    if not view:
        return None
    f0 = int(round(r["shot_t"] * fps))
    if r.get("shooter_box"):
        b = r["shooter_box"]
        seed = (f0, [b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy])
    else:
        seed = seed_from_point(r, players, sx, sy, f0, span, step)
        if seed is None:
            return None
    track = track_player(players, seed[0], seed[1], span, step)
    frames = pick_samples(track, players, samples)
    boxes = {f: [track[f][0] * sx, track[f][1] * sy, track[f][2] * sx, track[f][3] * sy]
             for f in frames}
    if not frames:
        return None
    first, last = min(frames), max(frames)
    every = step
    n = (last - first) // every + 1
    imgs = grab_frames(video_path, first / fps, n, every, tuple(view))
    crops = []
    vx0, vy0 = view[0], view[1]
    for f in frames:
        k = (f - first) // every
        if k >= len(imgs) or (f - first) % every:
            continue
        x0, y0, x1, y1 = boxes[f]
        w, h = x1 - x0, y1 - y0
        a, b = int(max(0, x0 - 0.15 * w - vx0)), int(max(0, y0 - 0.05 * h - vy0))
        c, d = int(x1 + 0.15 * w - vx0), int(y1 + 0.05 * h - vy0)
        crop = imgs[k][b:d, a:c]
        if not (crop.size and crop.shape[0] > 8 and crop.shape[1] > 4):
            continue
        team = r.get("team")
        if colours and team is not None:
            lab = torso_lab(imgs[k], (x0 - vx0, y0 - vy0, x1 - vx0, y1 - vy0))
            if lab is not None and nearest_team_colour(lab, colours) == 1 - team:
                continue  # opponent's shirt: the track switched players
        crops.append(np.ascontiguousarray(crop))
    return make_grid(crops) if crops else None
