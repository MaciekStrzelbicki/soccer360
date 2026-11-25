"""Scoring stage 3: jersey numbers, roster, corrections."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import jerseys, scoring, shots
from tests.test_shots import FPS, H, W, clip  # noqa: F401  (fixture)


def _players(tracks: dict[int, list[list[float]]]) -> shots.Players:
    fr, bx = [], []
    for f, boxes in tracks.items():
        for b in boxes:
            fr.append(f)
            bx.append(b)
    return shots.Players(np.array(fr), np.array(bx, np.float32), 720, 360)


class TestTracking:
    def test_follows_moving_player_not_neighbour(self):
        tracks = {f: [[100 + f, 50, 110 + f, 80], [300, 50, 310, 80]] for f in range(0, 61)}
        p = _players(tracks)
        tr = jerseys.track_player(p, 30, [130, 50, 140, 80], span=30, step=6)
        assert sorted(tr) == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        assert all(abs(b[0] - (100 + f)) < 1e-3 for f, b in tr.items())

    def test_stops_when_lost(self):
        tracks = {f: [[100, 50, 110, 80]] for f in range(0, 31)}
        tracks.update({f: [[400, 50, 410, 80]] for f in range(31, 61)})
        tr = jerseys.track_player(_players(tracks), 30, [100, 50, 110, 80], span=30, step=6)
        assert max(tr) == 30

    def test_stops_on_size_jump(self):
        tracks = {f: [[100, 50, 110, 80]] for f in range(0, 31)}
        tracks.update({f: [[100, 30, 120, 90]] for f in range(31, 61)})  # 2x taller box
        tr = jerseys.track_player(_players(tracks), 30, [100, 50, 110, 80], span=30, step=6)
        assert max(tr) == 30

    def test_seed_from_point(self):
        # no box at the shot frame 30, the free-throw shooter is detected at 36
        p = _players({36: [[200, 100, 210, 130]], 30: [[400, 100, 410, 130]]})
        r = {"view": [0, 0, 720, 360], "xy": [205 / 720, 117 / 360]}
        assert jerseys.seed_from_point(r, p, 1.0, 1.0, 30, 30, 6) == (36, [200, 100, 210, 130])
        r["xy"] = [0.9, 0.9]
        assert jerseys.seed_from_point(r, p, 1.0, 1.0, 30, 30, 6) is None

    def test_samples_skip_overlaps_and_spread(self):
        tracks = {f: [[100, 50, 110, 80]] for f in range(0, 61, 6)}
        tracks[24].append([105, 50, 115, 80])  # someone in front at frame 24
        p = _players(tracks)
        tr = jerseys.track_player(p, 30, [100, 50, 110, 80], span=30, step=6)
        s = jerseys.pick_samples(tr, p, 4)
        assert 24 not in s and len(s) == 4 and s == sorted(s)

    def test_grid(self):
        g = jerseys.make_grid([np.full((50, 20, 3), 200, np.uint8)] * 5, cell_h=100, cols=4)
        assert g.shape == (2 * 124, 4 * 75, 3)


class TestParseAndPick:
    def test_parse(self):
        r = jerseys.parse_numbers({"reads": [
            {"image": 0, "number": 7, "confidence": "high"},
            {"image": 1, "number": 123, "confidence": "high"},
            {"image": 5, "number": 3}]}, 3)
        assert r[0] == {"number": 7, "confidence": "high"}
        assert r[1] == {"number": None, "confidence": None}
        assert r[2] is None

    def test_prompt_lists_shirts(self):
        p = jerseys.number_prompt(["orange", ""])
        assert "#0: shooter in a orange shirt" in p and "#1: shooter\n" in p

    @pytest.mark.parametrize("read,num,edit,want", [
        ({"number": 5}, {"number": 7, "confidence": "high"}, {"number": 9},
         {"number": 9, "number_src": "edit"}),
        ({"number": 5}, {"number": 7, "confidence": "high"}, {"number": None},
         {"number": None, "number_src": "edit"}),
        ({"number": 5}, {"number": 7, "confidence": "medium"}, None,
         {"number": 7, "number_src": "read", "number_check": True}),
        ({"number": 5}, {"number": 7, "confidence": "low"}, None,
         {"number": 5, "number_src": "shot", "number_check": True}),
        ({}, {"number": 7, "confidence": "low"}, None,
         {"number": 7, "number_src": "read_low"}),
        ({}, None, None, {"number": None, "number_src": None}),
    ])
    def test_pick_number(self, read, num, edit, want):
        assert shots.pick_number(read, num, edit) == want


class TestOutputs:
    MATCH = {"home": "Tygrysy", "guest": "Lwy", "home_side": "left",
             "roster": {"home": {"7": "Kowalski"}}}

    def test_youtube_players_and_scorers(self):
        res = {"final": {"home": 5, "guest": 2}, "periods": [], "events": [
            {"board_t": 10, "t": 7, "team": "home", "points": 2, "home": 2, "guest": 0,
             "shot": {"number": 7, "number_src": "read"}},
            {"board_t": 20, "t": 17, "team": "guest", "points": 2, "home": 2, "guest": 2,
             "shot": {"number": 4, "number_src": "edit"}},
            {"board_t": 30, "t": 27, "team": "home", "points": 3, "home": 5, "guest": 2,
             "shot": {"number": 7, "number_src": "shot"}},
            {"board_t": 40, "t": 37, "team": "home", "points": 1, "home": 6, "guest": 2,
             "shot": {"number": 11, "number_src": "read_low"}},
        ]}
        txt = scoring.youtube_text(res, self.MATCH)
        assert "0:07 2:0 Tygrysy +2 – #7 Kowalski" in txt
        assert "0:17 2:2 Lwy +2 – #4\n" in txt
        assert "0:37 6:2 Tygrysy +1\n" in txt          # uncertain number left out
        assert "Strzelcy:\nTygrysy: #7 Kowalski 5\nLwy: #4 2" in txt
        assert "#7" not in scoring.youtube_text(res, self.MATCH, players=False)

    def test_roster_cleaning_and_kept_on_team_save(self, tmp_path):
        from src import calibration as cal
        c = {"calibration": {"dir": str(tmp_path)}}
        v = tmp_path / "g.mp4"
        v.write_bytes(b"x")
        cal.save_match(c, v, {"home": "A", "roster": {"home": {"#7": " Kowalski ", "x": "?",
                                                               "150": "no"}}})
        assert cal.load_match(c, v)["roster"] == {"home": {"7": "Kowalski"}}
        cal.save_match(c, v, {"home": "B", "guest": "C"})   # calibrate page: no roster sent
        m = cal.load_match(c, v)
        assert m["home"] == "B" and m["roster"] == {"home": {"7": "Kowalski"}}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_numbers_end_to_end_and_review_api(clip, tmp_path):  # noqa: F811
    video, det = clip
    cal = {"hoops": [{"yaw": -60, "rim_pitch": -6, "board_top_pitch": -3},
                     {"yaw": 60, "rim_pitch": -6, "board_top_pitch": -3}]}
    config = {"detection": {"img_size": 360}, "scoring": {
        "shooters": {"enabled": True, "window_sec": 5, "workers": 1},
        "numbers": {"enabled": True, "track_sec": 1.0, "step_frames": 2, "samples": 4,
                    "per_request": 2}}}
    sc = scoring.scoring_config(config)
    events = [{"board_t": 7.0, "team": "home", "points": 2, "period": 1},
              {"board_t": 8.0, "team": "home", "points": 2, "period": 1},
              {"board_t": 8.5, "team": "home", "points": 3, "period": 1}]
    asked = []

    def reader(frames, prompt):
        if "JERSEY NUMBER" in prompt:
            asked.append(len(frames))
            assert "orange shirt" in prompt
            return {"reads": [{"image": i, "number": 23, "confidence": "high"}
                              for i in range(len(frames))]}
        return {"shot_frame": 2, "shooter_box": "A", "shooter_team": "A"}

    doc = shots.find_shots(config, sc, video, cal, events, W, H, FPS, det, tmp_path,
                           reader=reader)
    assert asked == [2, 1]                       # 3 baskets, 2 per request
    assert doc["numbers"]["8.50"]["number"] == 23
    grid = doc["reads"]["8.50"]["grid"]
    assert (tmp_path / grid).is_file()

    # outputs + review API on top of a "processed" dir
    proc = tmp_path / "processed" / "g"
    proc.mkdir(parents=True)
    for f in ("shots.json",):
        shutil.copy(tmp_path / f, proc / f)
    shutil.copytree(tmp_path / "shots", proc / "shots")
    res = {"start": {}, "periods": [], "final": {"home": 7, "guest": 0},
           "events": [dict(e) for e in events]}
    shots.apply_shots(res, doc)
    scoring.write_outputs(proc, res, scoring.match_info(None), sc)
    assert "– #23" in (proc / "youtube.txt").read_text()

    from fastapi.testclient import TestClient
    from src.dashboard import create_app
    c = TestClient(create_app({"paths": {"processed": str(tmp_path / "processed")},
                               "dashboard": {"db_path": str(tmp_path / "d.db")},
                               "calibration": {"dir": str(tmp_path / "cal")},
                               "scoring": config["scoring"], "detection": config["detection"]}))
    assert c.get("/scoring/g").status_code == 200
    v = c.get("/api/scoring/g").json()
    assert len(v["events"]) == 3 and v["events"][0]["key"] == "7.00"
    name = Path(v["events"][0]["shot"]["grid"]).name
    assert c.get(f"/api/scoring/g/file/{name}").headers["content-type"] == "image/jpeg"
    assert c.get("/api/scoring/g/file/..%2Fshots.json").status_code == 404

    v = c.post("/api/scoring/g/edits", json={"edits": {"7.00": {"number": 4}}}).json()
    assert v["events"][0]["shot"]["number"] == 4 and v["events"][0]["shot"]["number_src"] == "edit"
    assert c.post("/api/scoring/g/edits", json={"edits": {"7.00": {"number": 400}}}).status_code == 400
    v = c.post("/api/scoring/g/roster", json={"home": "4 Kowalski\n#23. Nowak", "guest": ""}).json()
    assert v["match"]["roster"]["home"] == {"4": "Kowalski", "23": "Nowak"}
    yt = (proc / "youtube.txt").read_text()
    assert "– #4 Kowalski" in yt and "#23 Nowak" in yt
    assert "Gospodarze: #23 Nowak 5, #4 Kowalski 2" in yt
    v = c.post("/api/scoring/g/edits", json={"edits": {"7.00": {"clear": ["number"]}}}).json()
    assert v["events"][0]["shot"]["number"] == 23

    # scorer's-table fixes from the review page
    v = c.post("/api/scoring/g/edits", json={"edits": {"8.00": {"cancel": True}}}).json()
    e = v["events"][1]
    assert e["points"] == 0 and e["flag"] == "edited" and e["edit"] == {"cancel": True}
    assert v["events"][2]["home"] == 5            # running score recounted
    assert "Gospodarze: #23 Nowak 5\n" in (proc / "youtube.txt").read_text()
    v = c.post("/api/scoring/g/edits", json={"edits": {"8.00": {"cancel": False},
                                                       "8.50": {"team": "guest"}}}).json()
    assert v["events"][1]["points"] == 2 and v["events"][2]["team"] == "guest"
    assert c.post("/api/scoring/g/edits", json={"edits": {"8.50": {"points": 5}}}).status_code == 400
    v = c.post("/api/scoring/g/edits", json={"edits": {"8.50": None}}).json()
    assert v["events"][2]["team"] == "home"


def test_reader_stops_after_no_credits():
    r = shots.OpenAIShotReader.__new__(shots.OpenAIShotReader)
    r.dead = None
    calls = []

    def boom(frames, prompt):
        calls.append(1)
        raise RuntimeError("Error code: 429 - insufficient_quota")

    r._call = boom
    for _ in range(3):
        with pytest.raises(RuntimeError):
            r([], "x")
    assert len(calls) == 1 and "no credits" in r.dead
