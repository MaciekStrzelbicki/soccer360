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

## Scores and YouTube description

With `scoring.enabled: true` (on in `basketball.yaml`) processing reads the
score from the venue scoreboard (needs a calibrated scoreboard and
`OPENAI_API_KEY`) and writes to `data/processed/<match>/`:

- `scoring.json` / `scoring.csv` — every score change: time, team, points, score
- `youtube.txt` — final score, period chapters, list of baskets with timestamps

How it works: scoreboard crops from every keyframe (~1/s); board changes are
found locally (LED pixels, the running clock ignored); each constant board
state is read by the vision LLM in chronological batches (~10 requests per
game). Timestamps are the board update minus `scoring.lead_sec` (6 s), so a
click shows the play.

Team names: dashboard `/calibrate/<video>` → *Match (scoring)*, or the CLI.
Changing them after processing only rewrites the text files.

```bash
# score an already processed video (reuses data/processed/<stem>/calibration.json)
soccer360 -c configs/basketball.yaml score ~/Desktop/game.mp4 --home "Tygrysy" --guest "Lwy"
# only new team names / home side -> rewrite csv + youtube.txt
soccer360 -c configs/basketball.yaml score ~/Desktop/game.mp4 --home "Tygrysy" --text-only
```

The left scoreboard score is the home team; use `--home-side right` (or the
dashboard select) if the board is the other way round. Rows flagged `check`
(jump > 3 points) or `correction` (score went down) in `scoring.csv` are
worth a look; corrections are left out of `youtube.txt`.
