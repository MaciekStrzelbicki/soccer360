#!/usr/bin/env bash
# Local (non-Docker) setup for macOS (Apple Silicon -> MPS) or Linux.
# Usage: bash scripts/setup_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v ffmpeg >/dev/null || { echo "ffmpeg missing -> brew install ffmpeg"; exit 1; }

if command -v uv >/dev/null; then
  uv venv -p 3.11 .venv
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  PY=${PYTHON:-python3.11}
  command -v "$PY" >/dev/null || { echo "Need Python >= 3.11 (brew install python@3.11 or uv)"; exit 1; }
  "$PY" -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e ".[dev]"
fi

mkdir -p data/{ingest,models,state,logs}
( cd data/models && ../../.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo26m.pt')" )

.venv/bin/python -c "from src.utils import resolve_device; print('Inference device:', resolve_device('auto'))"
echo
echo "Ready. Run:"
echo "  source .venv/bin/activate"
echo "  soccer360 -c configs/basketball.yaml process data/ingest/<match>.mp4"
