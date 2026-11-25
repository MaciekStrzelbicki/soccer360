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

### Shooter and shirt colours (stage 2)

With `scoring.shooters.enabled` (on in `basketball.yaml`; needs two calibrated
hoops and `detections.jsonl`) every basket is also located in the video:

1. The player detections before each board update tell which basket the play
   is at; one vote over all baskets fixes which basket each team attacks per
   half.
2. `window_sec` (15 s) before the update, one full-resolution frame per second
   around that basket, with lettered player boxes.
3. Shirt colours of all players in those frames are clustered: the two biggest
   clusters are the teams (referees fall into the third).
4. The vision LLM names the release frame, the shooter and his team colour
   (one request per basket). Which colour is home/guest is voted against the
   board; baskets where the shooter's colour disagrees are asked again with
   the scoring team's colour as a hint.

Results: `shots.json` (LLM answers + colours, cached — reruns cost nothing),
`shots/<board time>.jpg` crops of every shooter (for jersey numbers later),
and in `scoring.csv` the columns `shot_t_sec`, `shooter_colour`, `colour_ok`,
`number_guess`, `shot_crop`. YouTube timestamps then use the shot itself
minus `shot_lead_sec` (3 s) instead of the board update minus 6 s. The
dashboard *Match* section shows the team colours.

```bash
# stage 2 only, on an already scored video (also resumes reads that failed,
# e.g. when the API ran out of credits)
soccer360 -c configs/basketball.yaml score ~/Desktop/game.mp4 --shots-only
```

Accuracy on the first game (66 baskets): the shooter's shirt matched the
board in 56/66 baskets without the hint; most crops show the right player,
free-throw shooters the detector missed get a wider crop around the LLM's
point. Treat the shooter as a strong suggestion, not a fact.
