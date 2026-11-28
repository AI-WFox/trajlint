# PushT experiment: does filtering bad episodes help the policy?

The question trajlint has to answer: if you clean a dataset, does the robot
policy trained on it actually get better? This experiment trains the same
Diffusion Policy on three versions of `lerobot/pusht` and scores each one in
the PushT simulator.

| arm | training data |
| --- | --- |
| `corrupted` | `lerobot/pusht` plus the faults trajlint found in real teleop data |
| `filtered` | the corrupted data after `trajlint filter` |
| `clean` | the original `lerobot/pusht` (reference) |

## The injected faults

`make_corrupted.py` copies `lerobot/pusht` and adds faults at rates close to
what the first audit of 65 Hub datasets measured (see `docs/week1-findings.md`).
The seed is fixed and every change is written to `corruption.json`.

| fault | episodes | what changes | trajlint found |
| --- | --- | --- | --- |
| idle start | 52 of 206 | robot waits 2-4 s before moving | 52 |
| idle end | 21 | robot waits 2-4 s after finishing | 21 |
| dead take | 10 extra | 10 s where nothing moves | 10 |
| action spike | 10 | two one-frame jumps of the command | 7 (the other 3 jumped to a point close to the current one) |
| frozen state | 10 | state stuck for 2.5 s while the action keeps moving | 10 |

No false alarms on the 127 untouched episodes, and the original `lerobot/pusht`
comes out with all 206 episodes clean.

`trajlint filter` (defaults: drop broken episodes, trim idle starts and ends)
keeps 206 of 216 episodes and 25,817 of 28,867 frames.

## Run it

Open [`pusht_experiment.ipynb`](pusht_experiment.ipynb) in Google Colab with a
T4 GPU and run all cells. Checkpoints are copied to Google Drive
(`MyDrive/trajlint-pusht`, about 0.9 GB per arm), so after a disconnect you
just run all cells again: finished arms are skipped, unfinished ones resume.
A speed test cell prints how long the full run will take before training starts.

Settings used for the run below: LeRobot 0.7.0, `diffusion` policy with a smaller U-Net
(`down_dims=[256,512,1024]`, 76M parameters), batch 64, 20k steps, seed 1000,
100 evaluation episodes per arm. All arms use exactly the same settings, so the
comparison is fair even though 20k steps is shorter than the LeRobot reference run.

The notebook is generated from `build_notebook.py`. To test the whole pipeline
on a CPU in a few minutes:

```bash
python build_notebook.py --smoke /tmp/smoke.py
SMOKE_REPO=$PWD/../.. python /tmp/smoke.py
```

## Results

First full run (2026-11-28, Colab T4, one seed, 100 evaluation episodes per arm):

| arm | episodes | success | 95% CI | mean max coverage |
| --- | --- | --- | --- | --- |
| `corrupted` | 216 | 30% (30/100) | 21.9-39.6% | 0.857 |
| `filtered` | 206 | **39% (39/100)** | 30.0-48.8% | **0.886** |
| `clean` | 206 | 37% (37/100) | 28.2-46.8% | 0.880 |

Training on the corrupted data costs about 7 points of success compared with
the original data, and filtering it with trajlint brings it back to the
level of the original (39% vs 37%).

Read this as a first signal, not a proof: it is one seed, and with 100
episodes per arm the gap between `corrupted` and `filtered` (9 points) is not
statistically significant (two-proportion test, p about 0.18). The next run
adds two more seeds for `corrupted` and `filtered`.

A 3,000-step pilot run (both arms near 1-2% success) is not reported: the
policy had not learned the task yet.
