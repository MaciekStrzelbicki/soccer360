"""Equirect -> perspective projection with horizon levelling and vertical
(keystone) correction.

Conventions (same as py360convert.e2p as used by the reframer):
yaw = (x/W)*360-180 (positive = right), pitch positive = up.
Direction vectors: x right (yaw +90), y up, z forward (yaw 0).

Horizon levelling: the raw 360 frame's zenith ``z`` (world "up" seen in raw
coordinates) defines a rotation R with R @ (0,1,0) = z. Views are built in
the levelled world frame and sampled from the raw frame via R.

Vertical correction ``k`` (0..1): instead of tilting the virtual camera by
the whole pitch, tilt it by (1-k)*pitch and shift the image plane by the
rest (like a shift lens). World verticals stay vertical as k -> 1; the
frame centre still points at the requested pitch.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])


def angles(d: np.ndarray) -> tuple[float, float]:
    d = d / np.linalg.norm(d)
    return (
        math.degrees(math.atan2(d[0], d[2])),
        math.degrees(math.asin(max(-1.0, min(1.0, float(d[1]))))),
    )


def zenith_from_roll_tilt(roll_deg: float, tilt_deg: float) -> np.ndarray:
    """Raw-frame zenith for a camera rolled right by roll and tilted by tilt."""
    z = np.array([-math.tan(math.radians(roll_deg)), 1.0, math.tan(math.radians(tilt_deg))])
    return z / np.linalg.norm(z)


def roll_tilt_from_zenith(z: np.ndarray) -> tuple[float, float]:
    return (
        math.degrees(math.atan2(-z[0], z[1])),
        math.degrees(math.atan2(z[2], z[1])),
    )


def level_rotation(z: np.ndarray | None) -> np.ndarray:
    """Rotation R with R @ (0,1,0) == z (shortest arc)."""
    if z is None:
        return np.eye(3)
    z = np.asarray(z, dtype=float)
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, z)
    s = np.linalg.norm(v)
    c = float(up @ z)
    if s < 1e-12:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s**2)


def raw_to_world(yaw: float, pitch: float, R: np.ndarray) -> tuple[float, float]:
    """Raw-frame angles (e.g. from detections) -> levelled world angles."""
    return angles(R.T @ direction(yaw, pitch))


def _view_rotation(yaw: float, pitch: float) -> np.ndarray:
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return ry @ rx


def view_maps(
    src_w: int, src_h: int, out_w: int, out_h: int,
    fov_h: float, yaw: float, pitch: float,
    R: np.ndarray | None = None, vertical_correction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps sampling a raw equirect (src_w x src_h) for one view."""
    k = float(min(max(vertical_correction, 0.0), 1.0))
    f = (out_w / 2) / math.tan(math.radians(fov_h / 2))
    shift = f * math.tan(math.radians(pitch * k))
    xs = (np.arange(out_w, dtype=np.float32) - out_w / 2 + 0.5)
    ys = (out_h / 2 - np.arange(out_h, dtype=np.float32) - 0.5) + np.float32(shift)
    X, Y = np.meshgrid(xs, ys)
    d = np.stack([X, Y, np.full_like(X, f)], -1)
    M = _view_rotation(yaw, pitch * (1 - k))
    if R is not None:
        M = R @ M
    d = d @ M.T.astype(np.float32)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    mx = (lon / (2 * np.pi) + 0.5) * src_w - 0.5
    my = (0.5 - lat / np.pi) * src_h - 0.5
    return mx.astype(np.float32), my.astype(np.float32)


def render_view(
    frame: np.ndarray, out_w: int, out_h: int, fov_h: float, yaw: float, pitch: float,
    R: np.ndarray | None = None, vertical_correction: float = 0.0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    mx, my = view_maps(w, h, out_w, out_h, fov_h, yaw, pitch, R, vertical_correction)
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def view_top_pitch(pitch: float, half_v_deg: float, vertical_correction: float = 0.0) -> float:
    """World pitch of the top-centre pixel of a view."""
    k = float(min(max(vertical_correction, 0.0), 1.0))
    return pitch * (1 - k) + math.degrees(
        math.atan(math.tan(math.radians(half_v_deg)) + math.tan(math.radians(pitch * k)))
    )


# ---------------------------------------------------------------------------
# Zenith estimation from vertical structures (walls, windows, posts)
# ---------------------------------------------------------------------------

def estimate_zenith(
    frame: np.ndarray, *, max_tilt_deg: float = 15.0, view_w: int = 1280,
    view_h: int = 960, fov: float = 90.0, seed: int = 0,
) -> tuple[np.ndarray | None, int]:
    """Zenith of a raw equirect frame from near-vertical line segments.

    Every world-vertical line lies on a great circle through the zenith, so
    the zenith is orthogonal to all those circles' normals (RANSAC + least
    squares). Returns (zenith or None, inlier count).
    """
    lsd = cv2.createLineSegmentDetector()
    normals, weights = [], []
    h, w = frame.shape[:2]
    for yaw in range(-180, 180, 30):
        for pitch in (0, -25):
            mx, my = view_maps(w, h, view_w, view_h, fov, yaw, pitch)
            img = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            lines = lsd.detect(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[0]
            if lines is None:
                continue
            f = (view_w / 2) / math.tan(math.radians(fov / 2))
            M = _view_rotation(yaw, pitch)
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 60 or math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1))) > 25:
                    continue
                p1 = M @ np.array([x1 - view_w / 2, view_h / 2 - y1, f])
                p2 = M @ np.array([x2 - view_w / 2, view_h / 2 - y2, f])
                n = np.cross(p1, p2)
                nn = np.linalg.norm(n)
                if nn > 1e-9:
                    normals.append(n / nn)
                    weights.append(length)
    if len(normals) < 20:
        return None, 0
    N, Wt = np.array(normals), np.array(weights)
    rng = np.random.default_rng(seed)
    thr = math.sin(math.radians(0.7))
    min_up = math.cos(math.radians(max_tilt_deg))
    best, best_c = None, -1.0
    for _ in range(3000):
        i, j = rng.choice(len(N), 2, replace=False)
        z = np.cross(N[i], N[j])
        nz = np.linalg.norm(z)
        if nz < 1e-9:
            continue
        z = z / nz
        if z[1] < 0:
            z = -z
        if z[1] < min_up:
            continue
        c = float(Wt[np.abs(N @ z) < thr].sum())
        if c > best_c:
            best_c, best = c, z
    if best is None:
        return None, 0
    inl = np.abs(N @ best) < thr
    if inl.sum() < 20:
        return None, int(inl.sum())
    m = (N[inl] * Wt[inl, None]).T @ N[inl]
    _, vecs = np.linalg.eigh(m)
    z = vecs[:, 0]
    if z[1] < 0:
        z = -z
    return z, int(inl.sum())
