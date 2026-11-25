"""Stage-2 scoring: attacked basket, shirt colours, shooter reads."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import scoring, shots

ORANGE = (235, 110, 30)
BEIGE = (205, 180, 135)
BLACK = (20, 20, 20)


def _lab(rgb):
    px = np.array(rgb, np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


class TestColours:
    @pytest.mark.parametrize("rgb,name", [
        (ORANGE, "orange"), (BEIGE, "beige"), (BLACK, "black"),
        ((240, 240, 240), "white"), ((30, 70, 200), "blue"), ((200, 30, 40), "red"),
        ((20, 30, 90), "navy blue"), ((40, 160, 60), "green"),
        ((178, 89, 64), "orange"), ((129, 98, 82), "beige"),
    ])
    def test_colour_name(self, rgb, name):
        assert shots.colour_name(rgb)["en"] == name

    def test_cluster_two_teams_and_referees(self):
        rng = np.random.default_rng(0)
        feats = np.vstack([
            _lab(ORANGE) + rng.normal(0, 2, (40, 3)),
            _lab(BEIGE) + rng.normal(0, 2, (30, 3)),
            _lab(BLACK) + rng.normal(0, 2, (8, 3)),
        ]).astype(np.float32)
        c = shots.cluster_colours(feats)
        assert [t["name"]["en"] for t in c["teams"]] == ["orange", "beige"]
        assert shots.nearest_team_colour(_lab(ORANGE), c) == 0
        assert shots.nearest_team_colour(_lab(BEIGE), c) == 1
        assert shots.nearest_team_colour(_lab(BLACK), c) is None

    def test_too_few_samples(self):
        assert shots.cluster_colours(np.zeros((5, 3), np.float32)) is None


class TestAttack:
    def test_majority_and_half_swap(self):
        evs = [{"team": "home", "period": 1}, {"team": "guest", "period": 2},
               {"team": "guest", "period": 3}, {"team": "home", "period": 4},
               {"team": "home", "period": 1}]
        # home attacks hoop 1 in the 1st half; the last vote is a counter-attack
        votes = [1, 0, 1, 0, 0]
        m = shots.attack_map(evs, votes)
        assert m == {("home", 1): 1, ("guest", 1): 0, ("home", 2): 0, ("guest", 2): 1}

    def test_no_votes(self):
        assert shots.attack_map([{"team": "home"}], [None]) is None

    def test_play_hoop(self):
        cal = {"hoops": [{"yaw": -60, "rim_pitch": -6, "board_top_pitch": -3},
                         {"yaw": 60, "rim_pitch": -6, "board_top_pitch": -3}]}
        # det space 360x180: 1 px per degree; feet at pitch -20 (y=110)
        fr = np.repeat(np.arange(0, 100), 4)
        x = np.tile([230, 235, 240, 245], 100).astype(np.float32)  # yaw 50..65
        boxes = np.stack([x - 2, np.full_like(x, 90), x + 2, np.full_like(x, 110)], 1)
        p = shots.Players(fr, boxes, 360, 180)
        assert shots.play_hoop(p, cal, 1.0, 5.0, 10.0) == 1
        assert shots.play_hoop(p, cal, 20.0, 25.0, 10.0) is None  # no players


class TestParse:
    def test_valid(self):
        r = shots.parse_shot_answer(
            {"shot_frame": 2, "shooter_box": "b", "shooter_xy": [0.5, 0.25],
             "shooter_team": "B", "jersey_number": 7, "confidence": "high"},
            5, [0, 0, 3, 0, 0])
        assert r["frame"] == 2 and r["box"] == 1 and r["team"] == 1
        assert r["xy"] == [0.5, 0.25] and r["number"] == 7

    def test_invalid(self):
        r = shots.parse_shot_answer(
            {"shot_frame": 9, "shooter_box": "Z", "shooter_xy": [2, 0],
             "shooter_team": "C", "jersey_number": 120}, 5, [1] * 5)
        assert all(r[k] is None for k in ("frame", "box", "xy", "team", "number"))
        assert shots.parse_shot_answer("nope", 5, [1] * 5)["frame"] is None

    def test_box_letter_must_exist_in_frame(self):
        r = shots.parse_shot_answer({"shot_frame": 0, "shooter_box": "C"}, 2, [2, 5])
        assert r["frame"] == 0 and r["box"] is None


COLOURS = {"teams": [{"rgb": list(ORANGE), "name": {"en": "orange", "pl": "pomarańczowe"}},
                     {"rgb": list(BEIGE), "name": {"en": "beige", "pl": "beżowe"}}],
           "centres": [], "team_clusters": [0, 1]}


class TestApply:
    def _result(self):
        return {"events": [
            {"board_t": 10.0, "team": "home", "points": 2},
            {"board_t": 20.0, "team": "guest", "points": 2},
            {"board_t": 30.0, "team": "home", "points": 1},
            {"board_t": 40.0, "team": "guest", "points": 3},
        ], "periods": [], "final": {}, "start": {}}

    def test_vote_and_flags(self):
        reads = {"10.00": {"shot_t": 7.0, "team": 1},
                 "20.00": {"shot_t": 16.0, "team": 0, "number": 5, "crop": "shots/x.jpg"},
                 "30.00": {"shot_t": 29.0, "team": 1},
                 "40.00": {"shot_t": 33.0, "team": 1}}  # disagrees with the board
        res = shots.apply_shots(self._result(), {"reads": reads, "colours": COLOURS})
        assert res["colours"]["home"]["name"] == "beżowe"
        assert res["colours"]["guest"]["name"] == "pomarańczowe"
        ev = res["events"]
        assert [e["shot"]["colour_ok"] for e in ev] == [True, True, True, False]
        assert ev[1]["shot_t"] == 16.0 and ev[1]["shot"]["number_guess"] == 5

    def test_no_doc(self):
        res = shots.apply_shots(self._result(), None)
        assert "shot" not in res["events"][0]

    def test_youtube_uses_shot_time(self, tmp_path):
        res = shots.apply_shots(self._result(), {"reads": {"20.00": {"shot_t": 16.0, "team": 0}},
                                                 "colours": COLOURS})
        sc = scoring.scoring_config({})
        scoring.write_outputs(tmp_path, res, scoring.match_info(None), sc)
        doc = json.loads((tmp_path / "scoring.json").read_text())
        assert doc["events"][1]["t"] == 13.0          # shot - 3 s
        assert doc["events"][0]["t"] == 4.0           # board - 6 s (no shot)
        assert "0:13 " in (tmp_path / "youtube.txt").read_text()
        header = (tmp_path / "scoring.csv").read_text().splitlines()[0]
        assert "shooter_colour" in header


# ---------------------------------------------------------------------------
# End to end on a tiny synthetic video
# ---------------------------------------------------------------------------

W, H, FPS, SECS = 1440, 720, 10, 9


def _person(f, x, colour):
    """Person at full-res x: legs dark, shirt coloured; returns its box."""
    y0, y1 = 380, 440
    f[y0:y1, x:x + 16] = (60, 60, 60)
    f[y0 + 6:y0 + 26, x:x + 16] = colour
    return [x, y0, x + 16, y1]


@pytest.fixture
def clip(tmp_path):
    frames, boxes = [], []
    for i in range(FPS * SECS):
        f = np.full((H, W, 3), (40, 120, 200), np.uint8)   # blue floor
        f[: H // 2 - 10] = (230, 230, 230)
        # hoop at yaw 60 -> x 960; the view spans yaw 25..75 (x 820..1020)
        b = [_person(f, x, c) for x, c in
             ((840, ORANGE), (870, BEIGE), (900, ORANGE), (930, BEIGE), (960, ORANGE),
              (990, BLACK))]
        f[0:4] = 2 * i  # frame number as a grey stripe
        frames.append(f)
        boxes.append(b)
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-g", str(FPS),
         "-pix_fmt", "yuv444p", "-crf", "0", str(video)],
        input=b"".join(f.tobytes() for f in frames), check=True,
    )
    det = tmp_path / "detections.jsonl"
    with det.open("w") as fh:
        for i, bs in enumerate(boxes):
            for b in bs:  # detections at half resolution (det space 720x360)
                fh.write(json.dumps({"frame_index": i, "class_id": 0, "conf": 0.9,
                                     "bbox_xyxy": [v / 2 for v in b]}) + "\n")
    return video, det


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_grab_frames_exact_numbers(clip):
    video, _ = clip
    fr = shots.grab_frames(video, 2.0, 3, 5, (0, 0, W, 4))
    assert [int(round(float(np.median(f[:, :, 1])) / 2)) for f in fr] == [20, 25, 30]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_find_shots_end_to_end(clip, tmp_path):
    video, det = clip
    cal = {"hoops": [{"yaw": -60, "rim_pitch": -6, "board_top_pitch": -3},
                     {"yaw": 60, "rim_pitch": -6, "board_top_pitch": -3}]}
    config = {"detection": {"img_size": 360}}
    sc = scoring.scoring_config({"scoring": {"shooters": {
        "enabled": True, "window_sec": 5, "min_conf": 0.4, "workers": 1}}})
    events = [{"board_t": 7.0, "team": "home", "points": 2, "period": 1},
              {"board_t": 8.0, "team": "home", "points": 2, "period": 2},
              {"board_t": 8.5, "team": "guest", "points": 3, "period": 3}]
    prompts = []

    def reader(frames, prompt):
        prompts.append(prompt)
        assert len(frames) == 5 and frames[0].shape[0] > 50
        hinted_b = "says team B" in prompt
        return {"shot_frame": 2, "shooter_box": "B" if hinted_b else "A",
                "shooter_team": "B" if hinted_b else "A", "jersey_number": 4,
                "confidence": "high"}

    doc = shots.find_shots(config, sc, video, cal, events, W, H, FPS, det, tmp_path,
                           reader=reader)
    assert [t["name"]["en"] for t in doc["colours"]["teams"]] == ["orange", "beige"]
    assert "orange shirts" in prompts[0]
    assert len(prompts) == 4                   # 3 reads + 1 re-read with a hint
    r = doc["reads"]["8.50"]
    assert r["pass"] == 2 and r["team"] == 1
    assert r["shot_t"] == pytest.approx(8.5 - 5 + 2)
    assert (tmp_path / r["crop"]).is_file()
    assert shots.nearest_team_colour(r["box_lab"], doc["colours"]) == 1
    # an unfinished hint pass (e.g. API out of credits) is resumed on the next run
    doc["reads"]["8.50"]["pass"], doc["reads"]["8.50"]["team"] = 1, 0
    (tmp_path / "shots.json").write_text(json.dumps(doc))
    shots.find_shots(config, sc, video, cal, events, W, H, FPS, det, tmp_path, reader=reader)
    assert len(prompts) == 5 and "says team B" in prompts[-1]
    prompts.pop()
    # cached: no new requests; a deleted crop is re-cut from the video
    (tmp_path / r["crop"]).unlink()
    shots.find_shots(config, sc, video, cal, events, W, H, FPS, det, tmp_path, reader=reader)
    assert len(prompts) == 4
    assert (tmp_path / r["crop"]).is_file()

    res = shots.apply_shots({"events": events}, json.loads((tmp_path / "shots.json").read_text()))
    assert res["colours"]["home"]["name"] == "pomarańczowe"
    assert all(e["shot"]["colour_ok"] for e in res["events"])


def test_shot_region_from_point_without_box():
    r = {"view": [1000, 500, 2000, 1000], "xy": [0.5, 0.5]}
    x0, y0, x1, y1 = shots.shot_region(r)
    assert (x0 + x1) / 2 == 1500 and (y0 + y1) / 2 == 750
    assert x1 - x0 == pytest.approx(240) and y1 - y0 == pytest.approx(280)
    assert shots.shot_region({"view": [0, 0, 10, 10]}) is None
    box = shots.shot_region({"shooter_box": [100, 100, 120, 160]})
    assert box[0] < 100 and box[3] > 160


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_shots_only_on_scored_video(clip, tmp_path):
    video, det = clip
    cal = {"hoops": [{"yaw": -60, "rim_pitch": -6, "board_top_pitch": -3},
                     {"yaw": 60, "rim_pitch": -6, "board_top_pitch": -3}]}
    config = {"detection": {"img_size": 360},
              "scoring": {"shooters": {"enabled": True, "window_sec": 5, "workers": 1}}}
    result = {"start": {}, "periods": [], "final": {"home": 2, "guest": 0},
              "events": [{"board_t": 7.0, "team": "home", "points": 2, "period": 1}]}
    scoring.write_outputs(tmp_path, result, scoring.match_info(None),
                          scoring.scoring_config(config))
    if det.parent != tmp_path:
        shutil.copy(det, tmp_path / "detections.jsonl")

    def reader(frames, prompt):
        return {"shot_frame": 1, "shooter_box": "A", "shooter_team": "A"}

    scoring.run_shots_only(config, video, cal, W, H, FPS, tmp_path, shot_reader=reader)
    doc = json.loads((tmp_path / "scoring.json").read_text())
    assert doc["events"][0]["shot_t"] == pytest.approx(3.0)
    assert doc["events"][0]["t"] == pytest.approx(0.0)  # 3 s - 3 s lead
    assert doc["colours"]["home"]["name"] == "pomarańczowe"
