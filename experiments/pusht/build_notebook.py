"""Build pusht_experiment.ipynb from the cells below (keeps the notebook diff-friendly).

    python build_notebook.py            # writes pusht_experiment.ipynb
    python build_notebook.py --smoke    # also writes smoke_test.py, the code cells as one CPU script
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

CELLS: list[tuple[str, str]] = [
("md", """# Does cleaning robot data help the policy? (PushT)

This notebook trains the same Diffusion Policy three times on PushT and compares success in simulation:

| arm | training data |
|---|---|
| `corrupted` | `lerobot/pusht` with faults injected at the rates trajlint measured on real Hub datasets (idle starts, idle ends, dead takes, action spikes, frozen sensor readings) |
| `filtered` | the same corrupted data after `trajlint filter` (bad episodes dropped, idle starts/ends trimmed) |
| `clean` | the original `lerobot/pusht` (reference) |

**How to run:** `Runtime > Change runtime type > T4 GPU`, then run the cells from top to bottom.
Checkpoints are copied to Google Drive (`MyDrive/trajlint-pusht`), so if Colab disconnects,
just run all cells again: finished arms are skipped and unfinished ones resume from the last checkpoint.
"""),
("code", """# 1. Settings. Change STEPS if the speed test (step 4) says it will take too long.
STEPS = 20_000          # training steps per arm (same for every arm)
BATCH = 64
SAVE_EVERY = 2_500      # checkpoint (and copy to Drive) every N steps
EVAL_EPISODES = 100     # simulation episodes per arm for the final score
SEED = 1000
ARMS = ["corrupted", "filtered", "clean"]   # run order: the first two are the main comparison
SMALL_MODEL = True      # smaller U-Net (76M instead of 263M params): about 2x faster, same for every arm
TRAJLINT_REF = "main"   # git branch or tag of trajlint to use
SMOKE = False           # True = tiny CPU run to test the notebook itself (no Drive, no GPU)
"""),
("code", """# 2. Check the GPU and connect Google Drive
import os, sys, subprocess, shutil, json, time, re
from pathlib import Path

def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r.returncode

IN_COLAB = "google.colab" in sys.modules
if SMOKE:
    WORK = Path(os.environ.get("SMOKE_WORK", "/tmp/trajlint-smoke"))
    DEVICE = "cpu"
else:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    import torch
    assert torch.cuda.is_available(), "No GPU: choose Runtime > Change runtime type > T4 GPU, then run again"
    DEVICE = "cuda"
    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = Path("/content/drive/MyDrive/trajlint-pusht")
    else:
        WORK = Path("trajlint-pusht").resolve()
LOCAL = Path(os.environ.get("SMOKE_LOCAL", "/content")) if (IN_COLAB or SMOKE) else Path("local-runs").resolve()
DATA, RUNS = LOCAL / "data", LOCAL / "runs"
for d in (WORK / "runs", WORK / "results", WORK / "reports", DATA, RUNS):
    d.mkdir(parents=True, exist_ok=True)
print("results go to", WORK)
"""),
("code", """# 3. Install trajlint and LeRobot (2-4 minutes). If Colab asks to restart the session, restart and run again from step 1.
if not SMOKE:
    sh(f'pip install -q "git+https://github.com/AI-WFox/trajlint@{TRAJLINT_REF}" "lerobot[training,pusht]==0.7.0"')
    if not (LOCAL / "trajlint").exists():
        sh(f"git clone -q --depth 1 -b {TRAJLINT_REF} https://github.com/AI-WFox/trajlint {LOCAL / 'trajlint'}")
EXP = Path(os.environ.get("SMOKE_REPO", LOCAL / "trajlint")) / "experiments/pusht"
sh("trajlint --help > /dev/null && lerobot-train --help > /dev/null")
print("ok")
"""),
("code", """# 4. Build the three datasets (about 2 minutes) and save trajlint's reports to Drive
from huggingface_hub import snapshot_download

clean = DATA / "pusht_clean"
if not (clean / "meta/info.json").exists():
    snapshot_download("lerobot/pusht", repo_type="dataset", local_dir=clean)
corrupted, filtered = DATA / "pusht_corrupted", DATA / "pusht_filtered"
if not (corrupted / "meta/info.json").exists():
    sh(f"python {EXP / 'make_corrupted.py'} --src {clean} --out {corrupted} --seed 0")
if not (filtered / "meta/info.json").exists():
    sh(f"trajlint filter {corrupted} {filtered}")
sh(f"trajlint check {corrupted} -o {WORK / 'reports/corrupted'}")
shutil.copy2(corrupted / "corruption.json", WORK / "reports/corruption.json")
shutil.copy2(filtered / "trajlint-filter.json", WORK / "reports/trajlint-filter.json")
for name in ("clean", "corrupted", "filtered"):
    info = json.loads((DATA / f"pusht_{name}/meta/info.json").read_text())
    print(f"{name:10s} {info['total_episodes']:4d} episodes {info['total_frames']:6d} frames")
"""),
("code", """# 5. Helpers: train with resume + copy checkpoints to Drive, evaluate in simulation
def ckpts(d):
    root = d / "checkpoints"
    return sorted(p for p in root.glob("[0-9]*") if p.is_dir() and not p.name.endswith(".tmp")) if root.exists() else []

def to_drive(arm):
    \"\"\"Copy the newest local checkpoint to Drive and keep only that one there (saves Drive space).\"\"\"
    local = ckpts(RUNS / arm)
    if not local:
        return
    src, dst_root = local[-1], WORK / "runs" / arm / "checkpoints"
    dst = dst_root / src.name
    if not dst.exists():
        tmp = dst_root / (src.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp)
        tmp.rename(dst)
        print(f"  [saved {arm} step {int(src.name)} to Drive]", flush=True)
    for old in dst_root.glob("*"):
        if old.name != dst.name:
            shutil.rmtree(old, ignore_errors=True)
    for old in local[:-1]:  # older local checkpoints are not needed either
        shutil.rmtree(old, ignore_errors=True)

def from_drive(arm):
    \"\"\"After a disconnect, bring the last checkpoint back from Drive.\"\"\"
    saved = ckpts(WORK / "runs" / arm)
    if saved and not ckpts(RUNS / arm):
        dst = RUNS / arm / "checkpoints" / saved[-1].name
        shutil.copytree(saved[-1], dst)
        print(f"  restored {arm} from Drive at step {int(saved[-1].name)}")
    c = ckpts(RUNS / arm)
    if c:
        last = RUNS / arm / "checkpoints/last"
        if last.is_symlink() or last.exists():
            last.unlink()
        last.symlink_to(c[-1].name)

def run_streaming(cmd, arm, on_line=None):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    pending = False
    for line in p.stdout:
        if "Checkpoint policy after step" in line:
            pending = True
        elif pending and ("step:" in line or "End of training" in line):
            to_drive(arm); pending = False
        if on_line:
            on_line(line)
        if any(k in line for k in ("step:", "Error", "error", "Traceback", "Checkpoint", "End of training", "num_learnable")):
            print(line.rstrip()[-220:], flush=True)
    p.wait()
    to_drive(arm)
    if p.returncode != 0:
        raise RuntimeError(f"training failed for {arm} (exit {p.returncode}); scroll up for the error")

def train(arm, steps=None, save_every=None):
    steps, save_every = steps or STEPS, save_every or SAVE_EVERY
    from_drive(arm)
    done = [c for c in ckpts(RUNS / arm) if int(c.name) >= steps]
    if done:
        print(f"{arm}: already trained to {int(done[-1].name)} steps, skipping")
        return done[-1]
    if ckpts(RUNS / arm):
        cfg = RUNS / arm / "checkpoints/last/pretrained_model/train_config.json"
        cmd = f"lerobot-train --config_path={cfg} --resume=true"
    else:
        shutil.rmtree(RUNS / arm, ignore_errors=True)
        cmd = (f"lerobot-train --dataset.repo_id=local/pusht_{arm} --dataset.root={DATA / f'pusht_{arm}'} "
               f"--policy.type=diffusion --policy.push_to_hub=false --policy.device={DEVICE} "
               + ("'--policy.down_dims=[256,512,1024]' " if SMALL_MODEL else "")
               + ("--policy.pretrained_backbone_weights=null " if SMOKE else "")
               + f"--env.type=pusht --steps={steps} --batch_size={BATCH} --save_freq={save_every} "
               f"--env_eval_freq=0 --log_freq={1 if SMOKE else 200} --num_workers={0 if SMOKE else 2} "
               f"--seed={SEED} --output_dir={RUNS / arm}")
    run_streaming(cmd, arm)
    return ckpts(RUNS / arm)[-1]

def evaluate(arm, episodes=None):
    episodes = episodes or EVAL_EPISODES
    out_json = WORK / "results" / f"{arm}.json"
    if out_json.exists():
        r = json.loads(out_json.read_text())
        if r.get("n_episodes") == episodes and r.get("steps") == STEPS:
            print(f"{arm}: already evaluated: {r['pc_success']:.1f}% success")
            return r
    from_drive(arm)
    ck = ckpts(RUNS / arm)[-1]
    out = LOCAL / "evals" / arm
    shutil.rmtree(out, ignore_errors=True)
    extra = "--policy.num_inference_steps=2 --eval.max_episode_steps=20 " if SMOKE else ""
    sh(f"lerobot-eval --policy.path={ck / 'pretrained_model'} --policy.device={DEVICE} --env.type=pusht "
       f"--eval.n_episodes={episodes} --eval.batch_size={min(10, episodes)} --seed={SEED} {extra}--output_dir={out}")
    o = json.loads((out / "eval_info.json").read_text())["overall"]
    r = {"arm": arm, "steps": int(ck.name), "n_episodes": o["n_episodes"], "n_success": o["n_success"],
         "pc_success": o["pc_success"], "ci95": o.get("pc_success_ci95"), "avg_max_reward": o["avg_max_reward"]}
    out_json.write_text(json.dumps(r, indent=1))
    vids = sorted(out.rglob("*.mp4"))[:2]
    for v in vids:
        shutil.copy2(v, WORK / "results" / f"{arm}_{v.name}")
    print(f"{arm}: {r['pc_success']:.1f}% success ({r['n_success']}/{r['n_episodes']}), "
          f"mean max coverage {r['avg_max_reward']:.3f}")
    return r
print("helpers ready")
"""),
("code", """# 6. Speed test (1-2 minutes): estimates how long the full run takes on this GPU
speeds = []
def grab(line):
    m = re.search(r"updt_s:([0-9.]+)", line)
    if m:
        speeds.append(float(m.group(1)))
test_steps = 3 if SMOKE else 60
cmd = (f"lerobot-train --dataset.repo_id=local/pusht_filtered --dataset.root={DATA / 'pusht_filtered'} "
       f"--policy.type=diffusion --policy.push_to_hub=false --policy.device={DEVICE} "
       + ("'--policy.down_dims=[256,512,1024]' " if SMALL_MODEL else "")
       + ("--policy.pretrained_backbone_weights=null " if SMOKE else "")
       + f"--env.type=pusht --steps={test_steps} --batch_size={BATCH} --save_checkpoint=false "
       f"--env_eval_freq=0 --log_freq=1 --num_workers={0 if SMOKE else 2} --output_dir={LOCAL / 'speedtest'}")
shutil.rmtree(LOCAL / "speedtest", ignore_errors=True)
run_streaming(cmd, "speedtest", on_line=grab)
shutil.rmtree(LOCAL / "speedtest", ignore_errors=True)
shutil.rmtree(WORK / "runs/speedtest", ignore_errors=True)
s = sorted(speeds[len(speeds) // 3:])[len(speeds[len(speeds) // 3:]) // 2] if speeds else float("nan")
per_arm_h = s * STEPS / 3600 * 1.1
print(f"\\n~{s:.2f} s/step -> about {per_arm_h:.1f} h per arm, {per_arm_h * len(ARMS):.1f} h for all {len(ARMS)} arms")
print("Free Colab sessions last a few hours: that is fine, just run all cells again after a disconnect.")
"""),
("code", """# 7. Train + evaluate the "corrupted" arm
train("corrupted")
evaluate("corrupted")
"""),
("code", """# 8. Train + evaluate the "filtered" arm (same data after trajlint filter)
train("filtered")
evaluate("filtered")
"""),
("code", """# 9. Train + evaluate the "clean" arm (reference; optional if time is short)
train("clean")
evaluate("clean")
"""),
("code", """# 10. Results table (also saved to Drive as results/summary.json)
rows = [json.loads((WORK / "results" / f"{a}.json").read_text()) for a in ARMS if (WORK / "results" / f"{a}.json").exists()]
print(f"{'arm':10s} {'episodes':>8s} {'success':>8s} {'95% CI':>15s} {'coverage':>9s}")
for r in rows:
    ci = r.get("ci95") or [float("nan")] * 2
    print(f"{r['arm']:10s} {r['n_episodes']:8d} {r['pc_success']:7.1f}% {ci[0]:6.1f}-{ci[1]:5.1f}% {r['avg_max_reward']:9.3f}")
summary = {"steps": STEPS, "batch": BATCH, "seed": SEED, "small_model": SMALL_MODEL, "results": rows}
(WORK / "results/summary.json").write_text(json.dumps(summary, indent=1))
print("\\nCopy this line and send it back:")
print(json.dumps(summary))
"""),
]


def notebook() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.strip("\n").splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    (HERE / "pusht_experiment.ipynb").write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n")
    if "--smoke" in sys.argv:
        code = "\n\n".join(src for kind, src in CELLS if kind == "code")
        code = code.replace("SMOKE = False", "SMOKE = True").replace("STEPS = 20_000", "STEPS = 4") \
                   .replace("SAVE_EVERY = 2_500", "SAVE_EVERY = 2").replace("EVAL_EPISODES = 100", "EVAL_EPISODES = 2") \
                   .replace("BATCH = 64", "BATCH = 2")
        Path(sys.argv[sys.argv.index("--smoke") + 1]).write_text(code)
    print("wrote", HERE / "pusht_experiment.ipynb")
