"""concat_segments keeps the source recording's audio."""

from __future__ import annotations

import subprocess

from src.utils import concat_segments


def _streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return sorted(out.stdout.split())


def test_concat_muxes_source_audio(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=64x32:rate=10:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)], check=True)
    segs = []
    for i in range(2):
        seg = tmp_path / f"seg{i}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(i), "-i", str(src),
                        "-t", "1", "-an", "-c:v", "libx264", str(seg)], check=True)
        segs.append(seg)
    out = tmp_path / "out.mp4"
    concat_segments(segs, out, audio_source=src)
    assert _streams(out) == ["audio", "video"]


def test_concat_source_without_audio(tmp_path):
    seg = tmp_path / "seg.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=64x32:rate=10:duration=1",
                    "-c:v", "libx264", str(seg)], check=True)
    out = tmp_path / "out.mp4"
    concat_segments([seg], out, audio_source=seg)
    assert _streams(out) == ["video"]


def test_concat_pcm_audio_is_reencoded(tmp_path):
    src = tmp_path / "src.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=64x32:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=duration=1",
                    "-c:v", "libx264", "-c:a", "pcm_s24le", "-shortest", str(src)], check=True)
    seg = tmp_path / "seg.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an",
                    "-c:v", "copy", str(seg)], check=True)
    out = tmp_path / "out.mp4"
    concat_segments([seg], out, audio_source=src)
    assert _streams(out) == ["audio", "video"]
