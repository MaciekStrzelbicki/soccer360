"""highlights.enabled switches Phase 6 on/off."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.utils import VideoMeta


@pytest.mark.parametrize("enabled", [True, False])
def test_highlights_phase_switch(test_config, tmp_path, monkeypatch, enabled):
    from src.pipeline import Pipeline

    config = deepcopy(test_config)
    config["paths"]["scratch"] = str(tmp_path / "scratch")
    config["highlights"]["enabled"] = enabled
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"dummy")
    meta = VideoMeta(width=640, height=320, fps=30.0, duration=2.0, total_frames=60, codec="h264")
    calls = {"highlights": 0}

    class _Det:
        def __init__(self, cfg): pass
        def run_streaming(self, video_path, meta_obj, output_path):
            Path(output_path).write_text("")
            return 60

    class _Stab:
        def __init__(self, cfg): pass
        def run(self, detections_path, tracks_path, fps, total_frames=None):
            Path(tracks_path).write_text("[]")
            return []

    class _AL:
        def __init__(self, cfg): pass
        def run(self, *a, **k): return None

    class _Cam:
        keep_in_frame: list = []
        def __init__(self, cfg): pass
        def generate(self, tracks_path, meta_obj, out_path, **kwargs):
            Path(out_path).write_text("[]")
        def generate_static(self, meta_obj, out_path):
            Path(out_path).write_text("[]")

    class _Ref:
        scoreboard_overlay = None
        def __init__(self, cfg): pass
        def render_broadcast(self, *a, **k): return None
        def render_tactical(self, *a, **k): return None

    class _HL:
        def __init__(self, cfg): pass
        def detect_and_export(self, *a, **k):
            calls["highlights"] += 1

    class _Exp:
        def __init__(self, cfg): pass
        def finalize(self, *a, **k): return tmp_path / "out"

    monkeypatch.setattr("src.pipeline.resolve_model_path_v1", lambda cfg, models_dir=None: ("d.pt", "normal"))
    monkeypatch.setattr("src.pipeline.Detector", _Det)
    monkeypatch.setattr("src.pipeline.BallStabilizer", _Stab)
    monkeypatch.setattr("src.pipeline.ActiveLearningExporter", _AL)
    monkeypatch.setattr("src.pipeline.CameraPathGenerator", _Cam)
    monkeypatch.setattr("src.pipeline.Reframer", _Ref)
    monkeypatch.setattr("src.pipeline.HighlightDetector", _HL)
    monkeypatch.setattr("src.pipeline.Exporter", _Exp)
    monkeypatch.setattr("src.pipeline.probe_video", lambda path: meta)

    Pipeline(config).run(input_path, cleanup=False)
    assert calls["highlights"] == (1 if enabled else 0)
