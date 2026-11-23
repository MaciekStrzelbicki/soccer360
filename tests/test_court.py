"""Court geometry: 4 clicked corners -> floor metres and back."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.camera import CameraPathGenerator
from src.court import CourtModel, order_corners
from src.utils import VideoMeta

# Synthetic camera: 5 m up, 1 m behind the near sideline, 4 m left of centre.
CAM = (10.0, -1.0, 5.0)


def angles(X, Y, cam=CAM):
    cx, cy, h = cam
    dx, dy = X - cx, Y - cy
    return math.degrees(math.atan2(dx, dy)), -math.degrees(math.atan2(h, math.hypot(dx, dy)))


def corners(L=28.0, W=15.0):
    return [dict(zip(("yaw", "pitch"), angles(x, y))) for x, y in
            ((0, W), (L, W), (L, 0), (0, 0))]


def test_roundtrip_and_any_click_order():
    cs = corners()
    m = CourtModel([cs[2], cs[0], cs[3], cs[1]])  # shuffled clicks
    assert [c["yaw"] for c in m.corners] == [c["yaw"] for c in order_corners(cs)]
    for X, Y in ((1.575, 7.5), (14.0, 7.5), (26.0, 2.0), (3.0, 14.0)):
        yaw, pitch = angles(X, Y)
        gx, gy = m.to_court(yaw, pitch)
        assert gx == pytest.approx(X, abs=1e-6) and gy == pytest.approx(Y, abs=1e-6)
        by, bp = m.to_angles(X, Y)
        assert by == pytest.approx(yaw, abs=1e-4) and bp == pytest.approx(pitch, abs=1e-4)


def test_inside_and_horizon():
    m = CourtModel(corners())
    X, Y = m.to_court(*map(np.array, zip(angles(5, 5), angles(14, -2.5), (0.0, 10.0))))
    assert list(m.inside(X, Y, 0.5)) == [True, False, False]  # player, bench, above horizon


def test_invalid_corners():
    with pytest.raises(ValueError):
        CourtModel(corners()[:3])
    bad = corners()
    bad[0] = {"yaw": 0.0, "pitch": 5.0}
    with pytest.raises(ValueError):
        CourtModel(bad)
    assert CourtModel.from_calibration({"court": {"corners": bad}}) is None
    assert CourtModel.from_calibration({}) is None


def test_clean_and_save_court(tmp_path):
    from src import calibration as cal

    c = {"calibration": {"enabled": True, "dir": str(tmp_path / "cal")}}
    v = tmp_path / "m.mp4"
    assert cal.clean_court({"corners": corners()[:2]}) is None
    cal.save_manual(c, v, {"court": {"corners": corners(), "length_m": 28}})
    got = cal.load_manual(c, v)
    assert got["court"]["width_m"] == 15.0 and len(got["court"]["corners"]) == 4
    assert CourtModel.from_calibration(got) is not None


class TestCourtCamera:
    FPS = 30.0
    N = 300

    def _det(self, gen, i, X, Y, height_px=60.0):
        yaw, pitch = angles(X, Y)
        x = (yaw + 180.0) / 360.0 * gen.det_width
        y = (90.0 - pitch) / 180.0 * gen.det_height
        return {"frame_index": i, "class_id": 0, "conf": 0.9,
                "bbox_xyxy": [x - 10, y - height_px, x + 10, y]}

    def _run(self, test_config, tmp_path, players, bench=()):
        test_config["camera"]["court_fit"] = {"enabled": True, "max_fov": 115.0}
        test_config["center_of_play"]["enabled"] = True
        gen = CameraPathGenerator(test_config)
        gen.court = CourtModel(corners())
        dets = tmp_path / "det.jsonl"
        with open(dets, "w") as fh:
            for i in range(self.N):
                for X, Y in list(players) + list(bench):
                    fh.write(json.dumps(self._det(gen, i, X, Y)) + "\n")
        tracks = tmp_path / "tracks.json"
        tracks.write_text(json.dumps([{"frame": i, "ball": None} for i in range(self.N)]))
        meta = VideoMeta(width=640, height=320, fps=self.FPS,
                         duration=self.N / self.FPS, total_frames=self.N, codec="h264")
        out = tmp_path / "cam.json"
        gen.generate(tracks, meta, out, detections_path=dets)
        return json.loads(out.read_text()), gen

    def test_bench_is_ignored(self, test_config, tmp_path):
        players = [(5, 6), (6, 9), (7, 4), (8, 8), (4, 10)]
        bench = [(20, -2.5), (21, -2.5), (22, -2.5), (23, -2.5)]
        _, gen = self._run(test_config, tmp_path, players, bench)
        assert np.nanmax(gen._court_mean_x) < 9.0
        assert len(gen._court_pts[0]) == len(players) * self.N

    def test_zoom_widens_to_keep_players_in_frame(self, test_config, tmp_path):
        # players spread over the whole left half: needs a wide view
        players = [(2, 13), (12, 13), (3, 3), (12, 3), (7, 8)]
        path, gen = self._run(test_config, tmp_path, players)
        floor = gen._court_fit_floor(path, self.FPS)
        mid = path[self.N // 2]
        assert mid["fov"] >= floor[self.N // 2] - 1.0
        assert mid["fov"] > gen.default_fov
