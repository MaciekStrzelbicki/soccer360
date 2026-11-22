"""Tests for levelled / vertical-corrected projection."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src import projection as pj


def _noise_equirect(w=960, h=480, seed=0):
    rng = np.random.default_rng(seed)
    eq = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(eq, (0, 0), 3)


def test_matches_py360convert_without_correction():
    py360convert = pytest.importorskip("py360convert")
    eq = _noise_equirect()
    a = pj.render_view(eq, 320, 180, 90, 30, -20)
    fv = 2 * math.degrees(math.atan(math.tan(math.radians(45)) * 180 / 320))
    b = py360convert.e2p(eq, fov_deg=(90, fv), u_deg=30, v_deg=-20,
                         out_hw=(180, 320), mode="bilinear")
    assert np.abs(a.astype(float) - b).mean() < 2.0


@pytest.mark.parametrize("k", [0.0, 0.75, 1.0])
def test_centre_points_at_requested_angles(k):
    mx, my = pj.view_maps(960, 480, 641, 361, 80, 30, -35, None, k)
    yaw = (mx[180, 320] + 0.5) / 960 * 360 - 180
    pitch = 90 - (my[180, 320] + 0.5) / 480 * 180
    assert yaw == pytest.approx(30, abs=0.3)
    assert pitch == pytest.approx(-35, abs=0.3)


def test_full_correction_keeps_meridians_vertical():
    mx, _ = pj.view_maps(960, 480, 640, 360, 80, 30, -35, None, 1.0)
    lon = (mx + 0.5) / 960 * 360 - 180
    cols = {int(np.argmin(np.abs(lon[r] - 40))) for r in range(0, 360, 30)}
    assert max(cols) - min(cols) <= 1
    mx0, _ = pj.view_maps(960, 480, 640, 360, 80, 30, -35, None, 0.0)
    lon0 = (mx0 + 0.5) / 960 * 360 - 180
    cols0 = [int(np.argmin(np.abs(lon0[r] - 40))) for r in range(0, 360, 30)]
    assert max(cols0) - min(cols0) > 10  # plain perspective: verticals lean


def test_roll_tilt_roundtrip_and_rotation():
    z = pj.zenith_from_roll_tilt(2.0, -1.5)
    assert pj.roll_tilt_from_zenith(z) == pytest.approx((2.0, -1.5), abs=1e-6)
    R = pj.level_rotation(z)
    assert R @ np.array([0, 1.0, 0]) == pytest.approx(z)
    assert R @ R.T == pytest.approx(np.eye(3))
    assert pj.level_rotation(None) == pytest.approx(np.eye(3))


def test_raw_to_world_identity_and_inverse():
    assert pj.raw_to_world(25, -10, np.eye(3)) == pytest.approx((25, -10))
    R = pj.level_rotation(pj.zenith_from_roll_tilt(3, 2))
    raw = pj.angles(R @ pj.direction(40, -20))
    assert pj.raw_to_world(*raw, R) == pytest.approx((40, -20), abs=1e-6)


@pytest.mark.parametrize("k", [0.0, 0.5, 0.75])
@pytest.mark.parametrize("pitch", [-35.0, -20.0, 5.0])
def test_view_top_pitch_matches_maps(k, pitch):
    w, h, fov = 641, 361, 80.0
    half_v = math.degrees(math.atan(math.tan(math.radians(fov / 2)) * h / w))
    _, my = pj.view_maps(3600, 1800, w, h, fov, 0.0, pitch, None, k)
    top = 90 - (my[0, 320] + 0.5) / 1800 * 180
    assert pj.view_top_pitch(pitch, half_v, k) == pytest.approx(top, abs=0.2)


def _synthetic_hall(w=1920, h=960, zenith=None):
    """Equirect of vertical poles (meridians) + a horizon band, optionally tilted."""
    world = np.full((h, w, 3), 40, np.uint8)
    for yaw in range(-180, 180, 6):
        x = int((yaw + 180) / 360 * w)
        world[:, x:x + 3] = 230
    world[h // 2 + 40: h // 2 + 80] = (60, 120, 200)
    if zenith is None:
        return world
    # raw = R @ world: sample world image at R^T @ raw direction
    R = pj.level_rotation(zenith)
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    lon = np.radians(xs / w * 360 - 180)
    lat = np.radians(90 - ys / h * 180)
    d = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    dw = d @ R  # (R^T d) per pixel
    lw = np.arctan2(dw[..., 0], dw[..., 2])
    bw = np.arcsin(np.clip(dw[..., 1], -1, 1))
    mx = ((lw / (2 * np.pi) + 0.5) * w - 0.5).astype(np.float32)
    my = ((0.5 - bw / np.pi) * h - 0.5).astype(np.float32)
    return cv2.remap(world, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def test_estimate_zenith_recovers_known_tilt():
    true = pj.zenith_from_roll_tilt(3.0, -2.0)
    frame = _synthetic_hall(zenith=true)
    z, inliers = pj.estimate_zenith(frame, view_w=640, view_h=480)
    assert z is not None and inliers > 20
    roll, tilt = pj.roll_tilt_from_zenith(z)
    assert roll == pytest.approx(3.0, abs=0.5)
    assert tilt == pytest.approx(-2.0, abs=0.5)


def test_estimate_zenith_level_frame():
    z, _ = pj.estimate_zenith(_synthetic_hall(), view_w=640, view_h=480)
    roll, tilt = pj.roll_tilt_from_zenith(z)
    assert abs(roll) < 0.5 and abs(tilt) < 0.5


def test_reframer_projection_settings(test_config):
    from src.reframer import Reframer

    assert Reframer(test_config)._projection() is None
    test_config["reframer"]["vertical_correction"] = 0.75
    proj = Reframer(test_config)._projection()
    assert proj["k"] == 0.75 and np.allclose(proj["R"], np.eye(3))
    test_config["reframer"]["level"] = {"enabled": True, "roll_deg": 1.0, "tilt_deg": 0.0}
    proj = Reframer(test_config)._projection()
    assert not np.allclose(proj["R"], np.eye(3))


def test_render_segment_with_projection(synthetic_video, tmp_path):
    from src.reframer import _render_segment

    out = tmp_path / "seg.mp4"
    cams = [{"yaw": 0.0, "pitch": -20.0, "fov": 90.0}] * 10
    proj = {"R": pj.level_rotation(pj.zenith_from_roll_tilt(1, 0)).tolist(), "k": 0.75}
    _render_segment(str(synthetic_video), cams, 0, 10.0, None, None, 160, 90,
                    "libx264", 30, "ultrafast", str(out), 0, None, proj)
    assert out.exists() and out.stat().st_size > 0
