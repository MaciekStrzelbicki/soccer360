"""Tests for venue calibration (scoreboard / hoops)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src import calibration as cal


def _cfg(tmp_path: Path, **over) -> dict:
    c = {
        "calibration": {
            "enabled": True,
            "dir": str(tmp_path / "cal"),
            "refine_scoreboard": False,
            "hoops_from_players": False,
            "llm": {"enabled": False},
        },
        "camera": {"keep_in_frame": [{"yaw": 50.0, "pitch": -3.0, "margin_deg": 3.0}]},
        "reframer": {"scoreboard_overlay": {
            "enabled": True, "yaw": 8.0, "pitch": 3.0, "fov_h": 6.6, "aspect": 2.0}},
    }
    c["calibration"].update(over)
    return c


class TestClockScore:
    def test_blinking_digits_beat_moving_blob_and_static_light(self):
        """Clock digits (lit + change every 1 s) win over a moving jersey and a lamp."""
        n, h, w = 11, 120, 400
        frames = np.zeros((n, h, w, 3), np.uint8)
        red = (230, 30, 30)
        for i in range(n):
            # static red lamp: always lit, never changes
            frames[i, 10:20, 10:30] = red
            # clock digit: two patterns alternating every 5 frames (1 s @ 5 fps)
            if (i // 5) % 2 == 0:
                frames[i, 50:70, 200:204] = red
                frames[i, 50:54, 200:212] = red
            else:
                frames[i, 50:70, 208:212] = red
                frames[i, 66:70, 200:212] = red
            frames[i, 50:70, 220:224] = red  # always-lit segment next to it
            # orange jersey running across
            x = 20 + i * 30
            frames[i, 90:110, x:x + 12] = (240, 120, 20)
        pres, chg = cal.clock_score_from_frames(frames, lag=5)
        score = np.minimum(pres / n, chg / (n - 5))
        yy, xx = np.unravel_index(int(np.argmax(score)), score.shape)
        assert 50 <= yy < 70 and 198 <= xx < 214

    def test_score_to_scoreboard_geometry(self):
        width = 3600  # 0.1 deg per px
        score = np.zeros((200, 400), np.float32)
        score[90:110, 150:250] = 0.5  # 100 x 20 px digits blob
        box = (1800, 800, 2200, 1000)  # crop origin at yaw 0, pitch 10
        sb = cal.score_to_scoreboard(score, box, width)
        assert sb["yaw"] == pytest.approx(20.0, abs=0.2)
        assert sb["pitch"] == pytest.approx(0.0, abs=0.2)
        assert sb["width_deg"] == pytest.approx(15.0, abs=0.5)  # 10 deg * 1.5

    def test_compact_peak_beats_higher_scattered_activity(self):
        width = 3600  # 10 px per degree
        score = np.zeros((400, 1200), np.float32)
        rng = np.random.default_rng(0)
        # crowd: many scattered blobs with the highest single values
        for _ in range(60):
            y, x = rng.integers(220, 380), rng.integers(600, 1150)
            score[y:y + 6, x:x + 6] = 0.30
        # scoreboard: one compact blob, slightly lower peak
        score[100:120, 200:260] = 0.25
        sb = cal.score_to_scoreboard(score, (0, 0, 1200, 400), width)
        cx = (200 + 260) / 2 / width * 360 - 180
        assert sb["yaw"] == pytest.approx(cx, abs=0.5)
        assert sb["width_deg"] < 15

    def test_no_signal_returns_none(self):
        assert cal.score_to_scoreboard(np.zeros((10, 10), np.float32), (0, 0, 10, 10), 3840) is None


class TestHoopsFromPlayers:
    def test_ends_and_height(self, tmp_path):
        det_w, det_h = 3600, 1800  # 0.1 deg per px
        clusters = []
        dets = []
        rng = np.random.default_rng(1)
        for f in range(400):
            # play alternates between the two ends (yaw -50 and +60)
            yaw = -50.0 if (f // 100) % 2 == 0 else 60.0
            yaw += rng.uniform(-2, 2)
            x = (yaw + 180) / 360 * det_w
            clusters.append({"frame": f, "cluster": {"x": x, "y": 1100, "spread_x_deg": 10}})
            # player: feet at pitch -20 (y=1100), 8 deg tall (80 px)
            dets.append({"frame_index": f, "class_id": 0, "conf": 0.9,
                         "bbox_xyxy": [x - 10, 1020, x + 10, 1100]})
        cp = tmp_path / "pc.json"
        cp.write_text(json.dumps(clusters))
        dp = tmp_path / "d.jsonl"
        dp.write_text("\n".join(json.dumps(d) for d in dets))
        hoops = cal.estimate_hoops_from_players(dp, cp, det_w, det_h)
        assert len(hoops) == 2
        assert hoops[0]["yaw"] == pytest.approx(-50.0, abs=3)
        assert hoops[1]["yaw"] == pytest.approx(60.0, abs=3)
        exp_top = -20.0 + 8.0 * 3.95 / 1.55
        assert hoops[0]["board_top_pitch"] == pytest.approx(exp_top, abs=0.5)
        assert hoops[0]["rim_pitch"] < hoops[0]["board_top_pitch"]

    def test_missing_cluster_file(self, tmp_path):
        assert cal.estimate_hoops_from_players(tmp_path / "x", None, 100, 50) == []


class FakeLocator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def locate(self, frame):
        self.calls += 1
        return self.result


class TestResolve:
    def test_disabled_uses_static_config(self, tmp_path):
        c = _cfg(tmp_path)
        c["calibration"]["enabled"] = False
        r = cal.resolve_calibration(c, tmp_path / "v.mp4", 60)
        assert r["source"] == "config"
        assert r["scoreboard"]["yaw"] == 8.0
        assert r["hoops"][0]["board_top_pitch"] == -3.0

    def test_manual_wins(self, tmp_path):
        c = _cfg(tmp_path)
        video = tmp_path / "match1.mp4"
        cal.save_manual(c, video, {
            "scoreboard": {"yaw": 1, "pitch": 2, "width_deg": 6, "height_deg": 3},
            "hoops": [{"yaw": -40, "rim_pitch": -8, "board_top_pitch": -4}]})
        loc = FakeLocator({})
        r = cal.resolve_calibration(c, video, 60, locator=loc)
        assert r["source"] == "manual" and r["scoreboard"]["yaw"] == 1
        assert loc.calls == 0

    def test_registered_without_values_is_not_manual(self, tmp_path):
        c = _cfg(tmp_path)
        video = tmp_path / "match2.mp4"
        cal.register_video(c, video)
        assert cal.load_manual(c, video) is None

    def test_llm_result_used_and_cleaned(self, tmp_path):
        c = _cfg(tmp_path)
        loc = FakeLocator({
            "scoreboard": {"yaw": "9.5", "pitch": 2.5, "width_deg": 7, "height_deg": 3.5},
            "hoops": [{"yaw": -51, "rim_pitch": -8, "board_top_pitch": -4},
                      {"yaw": 62, "rim_pitch": -7}, {"bad": 1}],
        })
        with patch.object(cal, "grab_equirect", return_value=np.zeros((10, 20, 3), np.uint8)):
            r = cal.resolve_calibration(c, tmp_path / "v.mp4", 60, locator=loc)
        assert r["source"] == "llm"
        assert r["scoreboard"]["yaw"] == 9.5
        assert len(r["hoops"]) == 2
        assert r["hoops"][1]["board_top_pitch"] == -4.0  # rim + 3 default

    def test_llm_failure_falls_back_to_config(self, tmp_path):
        c = _cfg(tmp_path)

        class Boom:
            def locate(self, frame):
                raise RuntimeError("api down")

        with patch.object(cal, "grab_equirect", return_value=np.zeros((10, 20, 3), np.uint8)):
            r = cal.resolve_calibration(c, tmp_path / "v.mp4", 60, locator=Boom())
        assert r["scoreboard"]["yaw"] == 8.0
        assert r["hoops"][0]["yaw"] == 50.0

    def test_clock_refinement_overrides_llm_position(self, tmp_path):
        c = _cfg(tmp_path, refine_scoreboard=True)
        loc = FakeLocator({"scoreboard": {"yaw": 12, "pitch": 5, "width_deg": 7, "height_deg": 4}})
        refined = {"yaw": 8.2, "pitch": 3.3, "width_deg": 6.5, "height_deg": 3.4, "score": 0.3}
        with patch.object(cal, "grab_equirect", return_value=np.zeros((10, 20, 3), np.uint8)), \
             patch.object(cal, "detect_scoreboard_auto", return_value=refined) as det:
            r = cal.resolve_calibration(c, tmp_path / "v.mp4", 60, locator=loc)
        assert det.call_args.kwargs["prior"]["yaw"] == 12
        assert r["scoreboard"] == {"yaw": 8.2, "pitch": 3.3, "width_deg": 6.5, "height_deg": 3.4}

    def test_missing_api_key_is_soft(self, tmp_path, monkeypatch):
        c = _cfg(tmp_path)
        c["calibration"]["llm"] = {"enabled": True, "api_key_env": "NO_SUCH_KEY_XYZ"}
        monkeypatch.delenv("NO_SUCH_KEY_XYZ", raising=False)
        r = cal.resolve_calibration(c, tmp_path / "v.mp4", 60)
        assert r["source"] == "auto"
        assert r["scoreboard"]["yaw"] == 8.0  # static fallback


class TestApply:
    def test_apply_sets_camera_and_overlay(self, tmp_path):
        class Cam:
            keep_in_frame = []

        class Ref:
            scoreboard_overlay = {"enabled": True, "width_px": 380, "yaw": 0, "pitch": 0}

        c = _cfg(tmp_path)
        cam, ref = Cam(), Ref()
        cal.apply_calibration({
            "scoreboard": {"yaw": 8.2, "pitch": 3.3, "width_deg": 6.0, "height_deg": 3.0},
            "hoops": [{"yaw": -51, "rim_pitch": -8, "board_top_pitch": -4}],
        }, cam, ref, c)
        assert cam.keep_in_frame == [{"yaw": -51, "pitch": -4, "margin_deg": 3.0}]
        assert ref.scoreboard_overlay["yaw"] == 8.2
        assert ref.scoreboard_overlay["fov_h"] == pytest.approx(6.6)
        assert ref.scoreboard_overlay["aspect"] == pytest.approx(2.0)
        assert ref.scoreboard_overlay["width_px"] == 380

    def test_overlay_disabled_stays_disabled(self, tmp_path):
        class Cam:
            keep_in_frame = []

        class Ref:
            scoreboard_overlay = None

        ref = Ref()
        cal.apply_calibration({"scoreboard": {"yaw": 1, "pitch": 1, "width_deg": 5, "height_deg": 2},
                               "hoops": []}, Cam(), ref, _cfg(tmp_path))
        assert ref.scoreboard_overlay is None


class TestOpenAILocator:
    def test_two_stage_locate(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
        loc = cal.OpenAILocator({"api_key_env": "TEST_OPENAI_KEY", "model": "m"})
        answers = iter([
            {"scoreboard": {"yaw": 10, "pitch": 4, "width_deg": 8, "height_deg": 4},
             "hoops": [{"yaw": -50, "rim_pitch": -8, "board_top_pitch": -4}]},
            {"found": True, "yaw": 8.3, "pitch": 3.2, "width_deg": 6.5, "height_deg": 3.4},
            {"found": False},
        ])
        prompts = []

        def fake_ask(prompt, img):
            prompts.append((prompt, img.shape))
            return next(answers)

        monkeypatch.setattr(loc, "_ask", fake_ask)
        frame = np.zeros((1920, 3840, 3), np.uint8)
        res = loc.locate(frame)
        assert len(prompts) == 3
        assert res["scoreboard"]["yaw"] == 8.3          # zoom pass wins
        assert res["hoops"][0]["yaw"] == -50            # zoom said not found -> overview
        assert "yaw" in prompts[0][0].lower()

    def test_missing_key_raises(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.delenv("NOPE_KEY", raising=False)
        with pytest.raises(RuntimeError):
            cal.OpenAILocator({"api_key_env": "NOPE_KEY"})


def test_draw_grid_and_preview_shapes():
    img = np.zeros((900, 1800, 3), np.uint8)
    g = cal.draw_grid(img, (-180, 180), (-45, 45), 10)
    assert g.shape == img.shape and g.sum() > 0
    p = cal.preview_image(np.zeros((1920, 3840, 3), np.uint8), {
        "scoreboard": {"yaw": 0, "pitch": 0, "width_deg": 10, "height_deg": 5},
        "hoops": [{"yaw": 40, "rim_pitch": -5, "board_top_pitch": -2}]})
    assert p.sum() > 0


class TestDashboardCalibration:
    @pytest.fixture
    def client(self, tmp_path, synthetic_video):
        from src.dashboard import create_app
        from fastapi.testclient import TestClient

        ingest = tmp_path / "ingest"
        ingest.mkdir()
        (ingest / "game.mp4").write_bytes(Path(synthetic_video).read_bytes())
        config = {
            "paths": {"ingest": str(ingest), "stagging": str(tmp_path / "stg")},
            "dashboard": {"db_path": str(tmp_path / "d.db")},
            "calibration": {"enabled": True, "dir": str(tmp_path / "cal"),
                            "llm": {"enabled": False}},
        }
        return TestClient(create_app(config))

    def test_list_get_frame_save_clear(self, client):
        vids = client.get("/api/calibration").json()["videos"]
        assert [v["stem"] for v in vids] == ["game"]
        d = client.get("/api/calibration/game").json()
        assert d["manual"] is None and d["duration"] > 0
        fr = client.get("/api/calibration/game/frame?t=1&w=960")
        assert fr.status_code == 200 and fr.headers["content-type"] == "image/jpeg"
        r = client.post("/api/calibration/game", json={
            "scoreboard": {"yaw": 5, "pitch": 2, "width_deg": 6, "height_deg": 3},
            "hoops": [{"yaw": -40, "rim_pitch": -8, "board_top_pitch": -4}]})
        assert r.status_code == 200
        d = client.get("/api/calibration/game").json()
        assert d["manual"]["scoreboard"]["yaw"] == 5
        assert client.get("/api/calibration").json()["videos"][0]["has_manual"]
        assert client.delete("/api/calibration/game/manual").status_code == 200
        assert client.get("/api/calibration/game").json()["manual"] is None

    def test_unknown_video_404(self, client):
        assert client.get("/api/calibration/nope").status_code == 404
        assert client.get("/api/calibration/nope/frame").status_code == 404

    def test_empty_save_rejected(self, client):
        assert client.post("/api/calibration/game", json={}).status_code == 400

    def test_page_served(self, client):
        r = client.get("/calibrate/game")
        assert r.status_code == 200 and "Venue calibration" in r.text
