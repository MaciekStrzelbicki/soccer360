"""Tests for scoreboard scoring (stage 1: team scores + YouTube text)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src import scoring
from src import calibration as cal


def _reads(seq):
    """[(t, left, right, period), ...] -> timeline dicts."""
    return [{"t": float(t), "left": l, "right": r, "period": p} for t, l, r, p in seq]


class TestParse:
    def test_parse_tile_answer(self):
        data = {"tiles": [
            {"i": 0, "left": 12, "right": "7", "period": 2},
            {"i": 2, "left": None, "right": 3, "period": 0},   # period 0 -> None
            {"i": 9, "left": 1, "right": 1, "period": 1},      # out of range
            {"i": 1, "left": True, "right": -4, "period": "x"},
            "junk",
        ]}
        out = scoring.parse_tile_answer(data, 3)
        assert out[0] == {"left": 12, "right": 7, "period": 2}
        assert out[1] == {"left": None, "right": None, "period": None}
        assert out[2] == {"left": None, "right": 3, "period": None}

    def test_parse_garbage(self):
        assert scoring.parse_tile_answer({}, 2) == [None, None]
        assert scoring.parse_tile_answer({"tiles": "x"}, 1) == [None]


class TestClean:
    def test_lone_misread_fixed(self):
        assert scoring._clean_series([10, 10, 18, 10, 12]) == [10, 10, 10, 10, 12]

    def test_out_of_range_single(self):
        # 31 between 12 and 14 cannot be right
        assert scoring._clean_series([12, 31, 14, 14]) == [12, 12, 14, 14]

    def test_none_skipped(self):
        assert scoring._clean_series([5, None, 9, None, 5]) == [5, None, 5, None, 5]


class TestEvents:
    def test_basic_events_and_periods(self):
        tl = _reads([
            (0, 0, 0, 1), (15, 0, 0, 1), (16, 2, 0, 1), (17, 2, 0, 1),
            (30, 2, 0, 1), (31, 2, 3, 1), (45, 2, 3, 1),
            (600, 2, 3, 2), (601, 3, 3, 2), (602, 4, 3, 2), (615, 4, 3, 2),
        ])
        res = scoring.derive_events(tl)
        ev = [(e["board_t"], e["team"], e["points"], e["home"], e["guest"]) for e in res["events"]]
        assert ev == [(16, "home", 2, 2, 0), (31, "guest", 3, 2, 3),
                      (601, "home", 1, 3, 3), (602, "home", 1, 4, 3)]
        assert res["final"] == {"home": 4, "guest": 3}
        assert [p["period"] for p in res["periods"]] == [1, 2]

    def test_home_on_right(self):
        tl = _reads([(0, 0, 0, 1), (10, 0, 0, 1), (11, 2, 0, 1), (12, 2, 0, 1)])
        res = scoring.derive_events(tl, home_side="right")
        assert res["events"][0]["team"] == "guest"
        assert res["final"] == {"home": 0, "guest": 2}

    def test_flags(self):
        tl = _reads([(0, 10, 0, 1), (1, 10, 0, 1), (2, 18, 0, 1), (3, 18, 0, 1),
                     (4, 16, 0, 1), (5, 16, 0, 1)])
        ev = scoring.derive_events(tl)["events"]
        assert ev[0]["flag"] == "check" and ev[0]["points"] == 8
        assert ev[1]["flag"] == "correction" and ev[1]["points"] == -2


def _board(left_on: bool, right_on: bool, clock: int, h=30, w=60):
    """Synthetic red-LED board: two score 'digits' + a changing clock."""
    img = np.zeros((h, w, 3), np.uint8)
    if left_on:
        img[6:12, 12:20, 0] = 220
    if right_on:
        img[6:12, 40:48, 0] = 220
    if clock % 2:
        img[18:24, 26:34, 0] = 220      # clock blinks every sample
    return img


class TestChanges:
    def test_detects_score_steps_not_clock(self):
        # left lights up at 100, right at 200; clock blinks every sample
        crops = [_board(i >= 100, i >= 200, i) for i in range(300)]
        crops[150] = np.zeros_like(crops[150])     # one dark frame (glitch)
        ch = scoring.detect_changes(crops, margin=0.0)
        assert ch == [100, 200]

    def test_too_short(self):
        assert scoring.detect_changes([_board(1, 1, 0)] * 5) == []

    def test_segments(self):
        crops = [_board(i >= 3, False, 0) for i in range(8)]
        st = scoring.segment_states(np.arange(8.0) * 2, crops, [3])
        assert [(s["t"], s["i0"], s["i1"]) for s in st] == [(0, 0, 3), (6, 3, 8)]
        assert st[0]["tile"][8, 15, 0] == 0 and st[1]["tile"][8, 15, 0] == 220


class TestReadStates:
    def test_batches_context_and_cache(self):
        states = [{"t": float(i), "tile": np.full((4, 4, 3), i, np.uint8)} for i in range(7)]
        prompts, asked = [], []

        def reader(tiles, prompt):
            prompts.append(prompt)
            ids = [int(t[0, 0, 0]) for t in tiles]
            asked.extend(ids)
            return [{"left": i, "right": 2 * i, "period": 1} for i in ids]

        cache = {}
        tl = scoring.read_states(states, reader, per_request=3, cache=cache)
        assert [(r["t"], r["left"], r["right"]) for r in tl][-1] == (6.0, 6, 12)
        assert asked == list(range(7)) and len(prompts) == 3
        assert "Just before" not in prompts[0]
        assert "left 2, right 4" in prompts[1] and "left 5, right 10" in prompts[2]
        asked.clear()
        scoring.read_states(states, reader, per_request=3, cache=cache)
        assert asked == []

    def test_verify_pass_rereads_impossible_steps(self):
        truth = [0, 2, 2, 4, 5, 7, 7, 9, 10, 12]
        states = [{"t": float(i), "tile": np.full((4, 4, 3), i, np.uint8)} for i in range(10)]
        calls = []

        def reader(tiles, prompt):
            ids = [int(t[0, 0, 0]) for t in tiles]
            calls.append(ids)
            first = len(calls) == 1
            # first pass misreads state 5 as 17 (impossible +12, then -10)
            return [{"left": 17 if (first and i == 5) else truth[i], "right": 0,
                     "period": 1} for i in ids]

        cache = {}
        tl = scoring.read_states(states, reader, per_request=20, cache=cache,
                                 verify_radius=1)
        assert [r["left"] for r in tl] == truth
        assert calls[1] == [4, 5, 6, 7]      # windows around 5 and 6 merged
        n = len(calls)
        scoring.read_states(states, reader, per_request=20, cache=cache)
        assert len(calls) == n                # verified once, cached

    def test_suspicious_states(self):
        reads = [{"left": 0, "right": 0}, None, {"left": 2, "right": 0},
                 {"left": 1, "right": 0}, {"left": 1, "right": 7}]
        assert scoring.suspicious_states(reads) == [3, 4]

    def test_failed_request_is_unreadable(self):
        def boom(tiles, prompt):
            raise RuntimeError("api down")

        states = [{"t": float(i), "tile": np.zeros((2, 2, 3), np.uint8)} for i in range(3)]
        tl = scoring.read_states(states, boom)
        assert all(r.get("left") is None for r in tl)


class TestYoutube:
    RESULT = {
        "start": {"home": 0, "guest": 0},
        "events": [
            {"board_t": 130.0, "team": "home", "points": 2, "home": 2, "guest": 0, "period": 1},
            {"board_t": 200.0, "team": "guest", "points": 3, "home": 2, "guest": 3, "period": 1},
            {"board_t": 3700.0, "team": "home", "points": -1, "home": 1, "guest": 3,
             "period": 4, "flag": "correction"},
        ],
        "periods": [{"board_t": 4.0, "period": 1}, {"board_t": 900.0, "period": 2},
                    {"board_t": 2000.0, "period": 3}, {"board_t": 3000.0, "period": 4}],
        "final": {"home": 1, "guest": 3},
    }
    MATCH = {"home": "Tygrysy", "guest": "Lwy", "home_side": "left"}

    def test_text(self):
        txt = scoring.youtube_text(self.RESULT, self.MATCH, lead_sec=6)
        lines = txt.splitlines()
        assert lines[0] == "Tygrysy – Lwy  1:3"
        assert "0:00 1. kwarta" in lines          # first chapter forced to 0:00
        assert "14:54 2. kwarta" in lines
        assert "2:04 2:0 Tygrysy +2" in lines
        assert "3:14 2:3 Lwy +3" in lines
        assert not any("-1" in l for l in lines)  # corrections not listed

    def test_fmt_ts(self):
        assert scoring.fmt_ts(5) == "0:05"
        assert scoring.fmt_ts(3725) == "1:02:05"
        assert scoring.fmt_ts(-3) == "0:00"

    def test_few_periods_no_chapters(self):
        res = {**self.RESULT, "periods": [{"board_t": 4.0, "period": 1}]}
        assert "Rozdziały" not in scoring.youtube_text(res, self.MATCH)

    def test_write_and_regenerate(self, tmp_path):
        res = json.loads(json.dumps(self.RESULT))
        paths = scoring.write_outputs(tmp_path, res, self.MATCH, scoring.scoring_config({}))
        doc = json.loads(Path(paths["json"]).read_text())
        assert doc["events"][0]["timestamp"] == "2:04"
        assert doc["events"][0]["team_name"] == "Tygrysy"
        csv_lines = Path(paths["csv"]).read_text().splitlines()
        assert csv_lines[0].startswith("timestamp,") and len(csv_lines) == 4
        scoring.regenerate_outputs({}, tmp_path, {"match": {"home": "A", "guest": "B"}})
        assert (tmp_path / "youtube.txt").read_text().startswith("A – B  1:3")


class TestMatchStorage:
    def test_match_survives_calibration_save(self, tmp_path):
        c = {"calibration": {"dir": str(tmp_path)}}
        v = tmp_path / "game.mp4"
        v.write_bytes(b"x")
        cal.save_match(c, v, {"home": " Tygrysy ", "guest": "Lwy", "home_side": "bogus"})
        assert cal.load_match(c, v) == {"home": "Tygrysy", "guest": "Lwy", "home_side": "left"}
        cal.save_manual(c, v, {"scoreboard": {"yaw": 1, "pitch": 1, "width_deg": 5,
                                              "height_deg": 2}})
        assert cal.load_match(c, v)["home"] == "Tygrysy"
        cal.register_video(c, v)
        assert cal.load_match(c, v)["guest"] == "Lwy"

    def test_match_info_defaults(self):
        m = scoring.match_info(None)
        assert m == {"home": "Gospodarze", "guest": "Goście", "home_side": "left"}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_extract_board_crops(tmp_path):
    """Keyframe crops follow the board, including after a camera shift."""
    w, h, fps, secs = 360, 180, 10, 4
    video = tmp_path / "v.mp4"
    frames = []
    for i in range(fps * secs):
        f = np.zeros((h, w, 3), np.uint8)
        x = 200 if i < 2 * fps else 210          # camera bump at 2 s: +10 px = +10 deg
        f[80:90, x:x + 10] = 255
        frames.append(f)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-g", str(fps),
         "-pix_fmt", "yuv444p", "-crf", "0", str(video)],
        input=b"".join(f.tobytes() for f in frames), check=True,
    )
    sb = {"yaw": 25.0, "pitch": 5.0, "width_deg": 10.0, "height_deg": 10.0}
    shifts = [{"from_sec": 0.0, "dyaw": 0.0, "dpitch": 0.0},
              {"from_sec": 2.0, "dyaw": 10.0, "dpitch": 0.0}]
    times, crops = scoring.extract_board_crops(video, sb, shifts, w, h, margin=0.0)
    assert list(np.round(times)) == [0, 1, 2, 3]
    for c in crops:
        assert c.shape[:2] == (10, 10)
        assert c.mean() > 200   # the white block is centred in every crop


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_extract_odd_sized_board(tmp_path):
    """Odd crop sizes/offsets (ffmpeg crops yuv420 on even pixels) stay aligned."""
    w, h, fps = 360, 180, 5
    video = tmp_path / "v.mp4"
    f = np.zeros((h, w, 3), np.uint8)
    f[81:90, 201:210] = 255
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-g", str(fps),
         "-pix_fmt", "yuv420p", "-crf", "0", str(video)],
        input=f.tobytes() * (fps * 3), check=True,
    )
    sb = {"yaw": 25.5, "pitch": 4.5, "width_deg": 9.0, "height_deg": 9.0}
    times, crops = scoring.extract_board_crops(video, sb, None, w, h, margin=0.0)
    assert len(times) == 3
    for c in crops:
        assert c.shape[:2] == (9, 9) and c.mean() > 150
