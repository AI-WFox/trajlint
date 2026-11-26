# What breaks in public LeRobot datasets (first audit, 26 Nov 2026)

We ran `trajlint` (table checks only, no video) on the most-downloaded LeRobot
datasets found by searching the Hub for `so101`, `so100` and `koch`
(25 per term). Raw numbers: [`audit-2026-11-26.json`](audit-2026-11-26.json).
Reproduce with:

```bash
python scripts/audit_hub.py --search so101 --search so100 --search koch --limit 25
```

## Headline numbers

| | Count |
| --- | --- |
| Datasets checked | 65 (6 skipped as larger than 300k frames, 4 could not be read: gated, missing files, corrupt parquet, non-LeRobot layout) |
| Episodes checked | 6,953 |
| Episodes graded "likely broken" | 209 (3.0%) |
| Datasets with at least one broken episode | 11 of 65 (17%) |
| Episodes with an idle start worth trimming (≥1.5 s) | 1,314 (19%) |
| Episodes with an idle end worth trimming | 659 (9%) |

## Problems found, by type

| Problem | Episodes | Datasets | What we saw |
| --- | ---: | ---: | --- |
| `index_conflict` | 122 | 2 | Two different recordings stored under the same `episode_index`, in different data files, with two rows in `meta/episodes`. Typical result of merging two recording sessions without re-numbering. A loader that groups frames by episode index mixes two takes into one. |
| `dead_episode` | 87 | 9 | Every joint has exactly zero range for the whole episode. Includes 4 of 56 episodes in `lerobot/svla_so100_stacking` (one of them 14.5 s long) and 2 of 52 in `lerobot/svla_so100_sorting`. One community dataset has 69 dead episodes out of 995. |
| `too_short` | 53 | 14 | Aborted takes, often under one second. Usually overlaps with dead episodes. |
| `frozen_state` | 44 | 17 | Joint readings identical for ≥2 s while the commanded action kept moving. Needs manual review: some may be real sensor dropouts, some a joint held against a limit. |
| `tracking_error` | 42 | 12 | Action and measured state disagree far more than in the rest of the dataset. |
| `too_long` | 11 | 3 | Recording left running. |
| `duplicate` | 7 | 1 | Same trajectory stored twice. |

## Lessons for the tool

- A frozen camera image on its own is not a fault: when the robot is idle the
  scene is static and compressed video changes by less than one grey level.
  The first version flagged 18 of 56 episodes in `svla_so100_stacking` this way.
  The camera check now only counts a frozen image while the robot is moving,
  which removed all 18 false positives.
- In the same way, identical joint readings mid-episode are only a sensor
  problem when the action keeps moving; otherwise they are a normal pause.
- Idle starts and ends are by far the most common finding (19% and 9% of
  episodes). Trimming them is the most useful thing `trajlint filter` can do
  next, and it is a measurable claim for the week-4 training experiment.

## Not yet checked

- Camera checks over the whole set (video download is large; done for
  `lerobot/svla_so101_pickplace` and `lerobot/svla_so100_stacking` only).
- Whether removing these episodes changes policy success. That is the week-4
  experiment.
