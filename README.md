# trajlint

[![tests](https://github.com/AI-WFox/trajlint/actions/workflows/tests.yml/badge.svg)](https://github.com/AI-WFox/trajlint/actions/workflows/tests.yml)
![license](https://img.shields.io/badge/license-Apache%202.0-blue)

**Find broken episodes in robot-learning datasets before you train on them.**

Teleoperated robot data is messy: cameras freeze, frames get dropped, a servo bus
stalls, an operator forgets to stop recording, two sessions get merged with
clashing episode numbers. A policy trained on that data quietly gets worse.
`trajlint` scans a [LeRobot](https://github.com/huggingface/lerobot) dataset
(local folder or Hugging Face Hub) and tells you which episodes to fix, trim or drop.

```bash
pip install git+https://github.com/AI-WFox/trajlint
trajlint check lerobot/svla_so100_stacking --no-video
```

```
lerobot/svla_so100_stacking  (56 episodes, 22956 frames, 30 fps)
  clean: 51   needs a look: 1   likely broken: 4

  problem                episodes
  idle_start                19   Robot sits still at the start: can be trimmed
  dead_episode               4   Robot state never changes: nothing was recorded
  too_short                  3   Episode much shorter than the others (often an aborted take)
  pause                      1   Robot holds still for a while mid-episode (often normal)

  worst episodes: 37 (45), 38 (45), 39 (60), 55 (60)
```

Yes, even curated datasets have them: episode 39 above is 14.5 seconds in which no
joint moves at all.

## What we found on the Hub

We checked 6,953 episodes in 65 popular SO-100, SO-101 and Koch datasets.
3% of episodes are likely broken, and 11 of the 65 datasets contain at least one:
recordings merged under the same episode number, episodes where nothing moves,
aborted takes. Almost 1 in 5 episodes starts with 1.5 s or more of the robot
sitting idle. Details: [docs/week1-findings.md](docs/week1-findings.md).

## Output

`trajlint check` writes three files to `trajlint-report/`:

| File | What it is |
| --- | --- |
| `report.html` | Readable report: what was found, per-episode problems, joint plots with the bad stretch shaded |
| `report.json` | Same results for scripts and dashboards |
| `keep_episodes.json` | Episodes that are not broken, ready for `LeRobotDataset(repo_id, episodes=[...])` |

## What it checks

| Check | Severity | What it means |
| --- | --- | --- |
| `index_conflict` | error | Two different recordings share one episode index (bad merge) |
| `dead_episode` | error | Robot state never changes: nothing was recorded |
| `nan_values` | error | NaN or infinite numbers in states or actions |
| `timestamp_order` | error | Timestamps go backwards or repeat |
| `dropped_frames` | warning / error | Gaps in timestamps: frames lost while recording |
| `frozen_state` | warning | Joint readings stuck while the action kept moving (sensor or bus dropout) |
| `action_spike` | warning | One-frame jumps in the action stream (teleop glitch) |
| `tracking_error` | warning | Action and state disagree far more than in other episodes (calibration?) |
| `duplicate` | warning | Same trajectory as another episode |
| `too_short` / `too_long` | warning / info | Much shorter or longer than the other episodes |
| `pause`, `idle_start`, `idle_end` | info | Robot holds still: usually fine, start/end can be trimmed |
| `frozen_camera` | warning / error | Camera image stops changing |
| `dark_frames` / `overexposed` | warning / error | Camera image almost black or white |
| `video_short`, `video_missing`, `video_unreadable` | error | Video does not cover the episode, is missing, or cannot be decoded |

Every episode is judged against the other episodes of the same dataset, so
thresholds adapt to the robot and the frame rate. An episode with any error is
graded **likely broken**, with a warning **needs a look**, otherwise **clean**.

## Clean a dataset

```bash
trajlint filter lerobot/svla_so100_stacking ./stacking-clean
```

```
  kept 51 of 56 episodes, 21116 of 22956 frames (8% removed)
```

`filter` drops broken episodes (`--drop review` drops those that need a look too),
trims idle time at the start and end of each episode (keeping 0.3 s of margin;
`--no-trim` to skip), and renumbers everything so the result loads like any
LeRobot v3.0 dataset. Videos are not re-encoded: only the per-episode video
timestamps move, and the original video files are hard-linked or copied.
Normalization statistics are kept from the source dataset. Checked with LeRobot 0.7.0:
the cleaned `svla_so100_stacking` loads with `LeRobotDataset`, and actions and
camera frames match the source frame for frame, including trimmed episodes.

## Options

```bash
trajlint check <repo_id or folder> [--no-video] [-o OUT] [--fail-on bad|review]
```

* `--no-video` skips camera checks: no video download, seconds instead of minutes.
* `--fail-on bad` exits with code 1 when a broken episode is found, so you can run it in CI
  before every training job.

Supports LeRobot dataset format v3.0 and v2.x.

## Roadmap

- [x] Table checks (states, actions, timestamps) and camera checks
- [x] HTML / JSON report and keep-list
- [x] `trajlint filter`: write a cleaned copy of a v3.0 dataset, with idle starts and ends trimmed
- [ ] Learned quality score, validated by training the same policy on raw vs. filtered data
- [ ] Public health board for datasets on the Hugging Face Hub

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests build small synthetic datasets with known faults (`tests/synth.py`), so they
run offline in about a second. `scripts/audit_hub.py` runs the checks over many
public datasets and summarizes what breaks most often.

## License

Apache 2.0
