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
- `youtube.txt` — ready to paste into the YouTube description: final score,
  points per period, shirt colours, then ONE chronological list (period
  markers + every basket with the shooter's number) that YouTube turns into
  chapters — it starts at 0:00 and entries less than 10 s apart share a line
  (YouTube's minimum chapter length; e.g. two free throws)

How it works: scoreboard crops from every keyframe (~1/s); board changes are
found locally (LED pixels, the running clock ignored); each constant board
state is read by the vision LLM in chronological batches (~10 requests per
game). Timestamps are the board update minus `scoring.lead_sec` (6 s), so a
click shows the play.

Team names: dashboard `/calibrate/<video>` → *Match (scoring)*, or the CLI.
Set them **before processing**: upload to staging, click *Drużyny* next to
the file, enter the names, then *Send To Ingest* (names are kept per video
name and follow the file into ingest). Changing them after processing only
rewrites the text files.

```bash
# score an already processed video (reuses data/processed/<stem>/calibration.json)
soccer360 -c configs/basketball.yaml score ~/Desktop/game.mp4 --home "Tygrysy" --guest "Lwy"
# only new team names / home side -> rewrite csv + youtube.txt
soccer360 -c configs/basketball.yaml score ~/Desktop/game.mp4 --home "Tygrysy" --text-only
```

The left scoreboard score is the home team; use `--home-side right` (or the
dashboard select) if the board is the other way round. Rows flagged `check`
(jump > 3 points) in `scoring.csv` are worth a look.

**Scorer's table mistakes** are undone automatically when the board itself is
corrected later:

- points taken back (`+2`, later `-2`, within 3 min) → that basket is
  `cancelled` (kept in `scoring.csv`, left out of `youtube.txt`);
- wrong value (`+3`, later `-1`) → a 2-pointer at the original time
  (`corrected`, `board_points` = what the board first showed);
- wrong team (points removed from one team and given to the other within
  20 s, also in one step) → the basket goes to the right team at the time
  it was first entered (`moved`, `entered_t_sec` = when it was moved), so
  shooter search looks at the right moment;
- points shown for one board state only with no transfer → treated as a
  misread and ignored.

Running scores are recounted after these fixes. What the automation cannot
untangle (e.g. a correction minutes later, a basket that never reached the
board) is fixed on the review page `/scoring/<video>`: *anuluj* (no basket),
*↔ drużyna*, points 1/2/3, all undoable; stored in `edits.json` next to the
results, so re-runs keep them.

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

### Jersey numbers, roster and corrections (stage 3)

With `scoring.numbers.enabled` the shooter is followed through the player
detections ±2 s around the shot (crossing players are not followed: box
size jumps and crops in the opponent's shirt colour are dropped) and up to 8
crops go on one grid image per basket (`shots/<t>_n.jpg`). `gpt-5-mini`
reads the number, 6 baskets per request.

Review page: dashboard `/scoring/<video>` (linked from the calibration page)
— every basket with the shooter crop (click: the grid), number field, source
(`odczyt` automatic, `poprawka` your correction, `niepewny` low-confidence
read shown only as a hint) and flags (`kolor?` shirt vs board, `różne` the two
automatic reads disagree). Saved numbers win over automatic ones and
regenerate `youtube.txt`. The roster (one `number name` per line per team)
adds names. `youtube.txt` then has `– #7 Kowalski` on every basket with a
known number and a `Strzelcy:` summary per team.

### LLM cost per game (first game; scoreboard not measured, estimate)

| Step | Model (default) | Requests | Cost |
|---|---|---|---|
| Scoreboard | `gpt-5` | ~12 | ~$0.5 (est.) |
| Shooters | `gpt-5-mini` | ~76 | ~$0.5 (`gpt-5`: ~$3.3) |
| Numbers | `gpt-5-mini`, low effort | ~11 | ~$0.02 |

Models are set per step: `scoring.llm`, `scoring.shooters.llm`,
`scoring.numbers.llm` (`model`, `reasoning_effort`). When the API account
runs out of credits the remaining requests stop at once; rerun
`score --shots-only` later — everything already answered is cached.
