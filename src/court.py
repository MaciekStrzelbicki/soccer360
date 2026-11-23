"""Court geometry from four clicked corners in the 360 frame.

Floor points seen from the camera lie on rays below the horizon. Projecting
each ray onto the plane one unit under the camera (gnomonic projection about
the nadir) maps the floor plane to that plane by a homography -- whatever the
camera height, tilt or roll. Four corners of a court of known size fix it, so
any detection's feet can be placed on the court in metres, and court points
can be projected back to (yaw, pitch).

Court coordinates: X along the court (0 = left baseline as seen from the
camera, ``length`` = right baseline), Y across it (0 = near sideline,
``width`` = far sideline). Angles follow the equirect convention
(yaw -180..180 left to right, pitch +90 up).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("soccer360.court")

COURT_LENGTH_M = 28.0
COURT_WIDTH_M = 15.0
_MIN_DOWN = 1e-3  # rays this close to the horizon never reach the floor


def _to_plane(yaw, pitch):
    """(yaw, pitch) deg -> nadir-gnomonic (u, v); NaN at/above the horizon."""
    yaw = np.radians(np.asarray(yaw, dtype=float))
    pitch = np.radians(np.asarray(pitch, dtype=float))
    dx = np.cos(pitch) * np.sin(yaw)
    dy = np.cos(pitch) * np.cos(yaw)
    down = -np.sin(pitch)
    ok = down > _MIN_DOWN
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(ok, dx / down, np.nan)
        v = np.where(ok, dy / down, np.nan)
    return u, v


def _from_plane(u, v):
    """Nadir-gnomonic (u, v) -> (yaw, pitch) deg."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = np.sqrt(u * u + v * v + 1.0)
    yaw = np.degrees(np.arctan2(u, v))
    pitch = np.degrees(np.arcsin(-1.0 / n))
    return yaw, pitch


def order_corners(corners: list[dict]) -> list[dict]:
    """Sort four corners as far-left, far-right, near-right, near-left.

    Far corners are the two higher in the frame (less negative pitch); left
    is the smaller yaw relative to the pair's mean direction.
    """
    pts = sorted(corners, key=lambda c: float(c["pitch"]), reverse=True)
    far, near = pts[:2], pts[2:]

    def lr(pair):
        a, b = pair
        d = (float(b["yaw"]) - float(a["yaw"]) + 180.0) % 360.0 - 180.0
        return (a, b) if d > 0 else (b, a)

    fl, fr = lr(far)
    nl, nr = lr(near)
    return [fl, fr, nr, nl]


class CourtModel:
    """Floor <-> frame mapping for one court."""

    def __init__(
        self,
        corners: list[dict],
        length_m: float = COURT_LENGTH_M,
        width_m: float = COURT_WIDTH_M,
    ):
        if len(corners) != 4:
            raise ValueError("court needs exactly 4 corners")
        self.corners = order_corners(corners)
        self.length = float(length_m)
        self.width = float(width_m)
        u, v = _to_plane(
            [c["yaw"] for c in self.corners], [c["pitch"] for c in self.corners]
        )
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)):
            raise ValueError("court corners must be below the horizon")
        src = np.stack([u, v], axis=1).astype(np.float64)
        L, W = self.length, self.width
        dst = np.array([[0, W], [L, W], [L, 0], [0, 0]], dtype=np.float64)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            raise ValueError("degenerate court corners")
        self.H = H
        self.Hinv = np.linalg.inv(H)

    @classmethod
    def from_calibration(cls, cal: dict | None) -> "CourtModel | None":
        court = (cal or {}).get("court")
        if not court or len(court.get("corners") or []) != 4:
            return None
        try:
            return cls(
                court["corners"],
                court.get("length_m", COURT_LENGTH_M),
                court.get("width_m", COURT_WIDTH_M),
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring invalid court calibration: %s", exc)
            return None

    def to_court(self, yaw, pitch):
        """(yaw, pitch) -> court (X, Y) metres; NaN above the horizon."""
        u, v = _to_plane(yaw, pitch)
        p = np.stack([u, v, np.ones_like(u)], axis=0).reshape(3, -1)
        q = self.H @ p
        with np.errstate(divide="ignore", invalid="ignore"):
            X = (q[0] / q[2]).reshape(np.shape(u))
            Y = (q[1] / q[2]).reshape(np.shape(u))
        return X, Y

    def to_angles(self, X, Y):
        """Court (X, Y) metres -> (yaw, pitch) degrees."""
        X = np.asarray(X, dtype=float)
        p = np.stack([X, np.asarray(Y, dtype=float), np.ones_like(X)], axis=0).reshape(3, -1)
        q = self.Hinv @ p
        u = (q[0] / q[2]).reshape(X.shape)
        v = (q[1] / q[2]).reshape(X.shape)
        return _from_plane(u, v)

    def inside(self, X, Y, margin_m: float = 0.0):
        """Boolean mask: (X, Y) within the court (+ margin)."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        m = float(margin_m)
        with np.errstate(invalid="ignore"):
            return (
                (X >= -m) & (X <= self.length + m) & (Y >= -m) & (Y <= self.width + m)
            )
