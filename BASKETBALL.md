# Basketball360 (fork of soccer360)

Auto-follow, tactical view and highlights for indoor basketball recorded
with an Insta360 X5 (or any 360 camera), running locally without Docker.

## Quick start (Mac)

```bash
brew install ffmpeg            # once
bash scripts/setup_local.sh    # venv + deps + yolo26m.pt into data/models
source .venv/bin/activate
soccer360 -c configs/basketball.yaml process data/ingest/match.mp4 --no-cleanup
```

Outputs land in `data/processed/<match>/` (broadcast follow, tactical wide,
highlights, camera path, detections). `data/` is git-ignored.

## Input

Export the clip from Insta360 Studio as **360 equirectangular MP4**
(not reframed), lens facing the court centre so the court is centred
horizontally (yaw 0). Start with 5-10 minutes.

## What differs from upstream

- `configs/basketball.yaml` extends `pipeline.yaml` (new `extends:` support)
  with indoor-court geometry, faster camera pans, more weight on the player
  cluster than on the ball, hoop regions for highlights, local `./data` paths.
- `detection.device: auto` -> CUDA, then Apple MPS, then CPU.
- Relative paths in configs resolve against the repo (not `/app`, `/tank`).

## Calibrate first

Values marked in `configs/basketball.yaml` assume the camera at half-court on
the sideline. After the first run check `data/processed/<match>/` and tune:
`field_of_interest` (yaw/pitch window), `filters.min_y_frac/max_y_frac`,
`highlights.goal_box_regions` (hoop positions).
