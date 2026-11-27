"""Make a copy of lerobot/pusht with the faults trajlint found in real teleop data.

Rates are close to what the first Hub audit measured (docs/week1-findings.md):
idle starts in about 1 of 5 episodes, a few dead takes, glitches and sensor
dropouts. Every change is recorded in corruption.json so results can be traced.

    python make_corrupted.py --src <path to lerobot/pusht> --out pusht_corrupted

Faults (fractions of the 206 source episodes, fixed seed):
  idle_start    25%  robot waits 2-4 s before moving (state, action and image frozen)
  idle_end      10%  robot waits 2-4 s after finishing
  dead           5%  extra takes where nothing moves (10 s each)
  action_spike   5%  two one-frame jumps of the commanded action to a random point
  frozen_state   5%  state reading stuck for 2.5 s while the action keeps moving
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd

RATES = {"idle_start": 0.25, "idle_end": 0.10, "dead": 0.05, "action_spike": 0.05, "frozen_state": 0.05}
VIDEO_KEY = "observation.image"


def decode_all(path: Path) -> list[np.ndarray]:
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
    return frames


def encode_all(frames: list[np.ndarray], path: Path, fps: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(path), "w") as c:
        s = c.add_stream(codec, rate=fps)
        h, w = frames[0].shape[:2]
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        if codec == "libx264":
            s.options = {"crf": "18", "g": "2", "preset": "fast"}
        for img in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return "h264" if codec == "libx264" else "mpeg4"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(args.seed)

    info = json.loads((src / "meta/info.json").read_text())
    fps = int(info["fps"])
    data = pd.concat([pd.read_parquet(f) for f in sorted((src / "data").rglob("*.parquet"))], ignore_index=True)
    meta = pd.concat([pd.read_parquet(f) for f in sorted((src / "meta/episodes").rglob("*.parquet"))],
                     ignore_index=True).set_index("episode_index")
    video_files = sorted((src / "videos" / VIDEO_KEY).rglob("*.mp4"))
    assert len(video_files) == 1, "expected the single-file pusht layout"
    all_frames = decode_all(video_files[0])

    n_src = len(meta)
    ids = np.arange(n_src)
    plan: dict[str, list[int]] = {}
    for fault in ("idle_start", "idle_end", "action_spike", "frozen_state"):
        plan[fault] = sorted(rng.choice(ids, size=round(RATES[fault] * n_src), replace=False).tolist())
    n_dead = round(RATES["dead"] * n_src)

    rows, metas, frames_out = [], [], []
    log = {"source": str(src), "seed": args.seed, "rates": RATES, "episodes": []}

    def add_episode(ep_df: pd.DataFrame, imgs: list[np.ndarray], tasks, faults: list[str], src_ep: int):
        new_idx = len(metas)
        n = len(ep_df)
        start_t = len(frames_out) / fps
        g = ep_df.copy()
        g["episode_index"] = new_idx
        g["frame_index"] = np.arange(n)
        g["timestamp"] = (np.arange(n) / fps).astype(np.float32)
        rows.append(g)
        frames_out.extend(imgs)
        metas.append({
            "episode_index": new_idx,
            "tasks": tasks,
            "length": n,
            f"videos/{VIDEO_KEY}/from_timestamp": start_t,
            f"videos/{VIDEO_KEY}/to_timestamp": start_t + n / fps,
        })
        log["episodes"].append({"new": new_idx, "source": src_ep, "faults": faults, "length": n})

    for e in range(n_src):
        m = meta.loc[e]
        ep = data[data["episode_index"] == e].sort_values("frame_index").reset_index(drop=True)
        t0 = float(m[f"videos/{VIDEO_KEY}/from_timestamp"])
        f0 = int(round(t0 * fps))
        imgs = all_frames[f0:f0 + len(ep)]
        assert len(imgs) == len(ep), (e, len(imgs), len(ep))
        faults = []
        state = np.stack(ep["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(ep["action"].to_numpy()).astype(np.float32)

        if e in plan["action_spike"]:
            for t in rng.choice(np.arange(5, len(ep) - 5), size=2, replace=False):
                action[t] = rng.uniform(0, 512, size=2)
            faults.append("action_spike")
        if e in plan["frozen_state"] and len(ep) > 40:
            a = int(rng.integers(5, len(ep) - 30))
            state[a:a + 25] = state[a]
            faults.append("frozen_state")
        ep["observation.state"] = list(state)
        ep["action"] = list(action)

        if e in plan["idle_start"]:
            k = int(rng.integers(2 * fps, 4 * fps + 1))
            pad = pd.concat([ep.iloc[[0]]] * k, ignore_index=True)
            pad["action"] = [state[0].copy() for _ in range(k)]  # holding still = command the current position
            ep = pd.concat([pad, ep], ignore_index=True)
            imgs = [imgs[0]] * k + imgs
            faults.append("idle_start")
        if e in plan["idle_end"]:
            k = int(rng.integers(2 * fps, 4 * fps + 1))
            last = ep.iloc[[-1]].copy()
            last["action"] = [np.asarray(ep["observation.state"].iloc[-1], dtype=np.float32)]
            pad = pd.concat([last] * k, ignore_index=True)
            ep = pd.concat([ep, pad], ignore_index=True)
            imgs = imgs + [imgs[-1]] * k
            faults.append("idle_end")
        add_episode(ep, imgs, m["tasks"], faults, e)

    for _ in range(n_dead):
        e = int(rng.integers(0, n_src))
        m = meta.loc[e]
        first = data[(data["episode_index"] == e) & (data["frame_index"] == 0)]
        k = 10 * fps
        ep = pd.concat([first] * k, ignore_index=True)
        ep["action"] = [np.asarray(first["observation.state"].iloc[0], dtype=np.float32) for _ in range(k)]
        f0 = int(round(float(m[f"videos/{VIDEO_KEY}/from_timestamp"]) * fps))
        add_episode(ep, [all_frames[f0]] * k, m["tasks"], ["dead"], e)

    frames = pd.concat(rows, ignore_index=True)
    frames["index"] = np.arange(len(frames))
    for c in ("observation.state", "action"):
        frames[c] = [np.asarray(v, dtype=np.float32) for v in frames[c]]
    (out / "data/chunk-000").mkdir(parents=True)
    frames.to_parquet(out / "data/chunk-000/file-000.parquet", index=False)

    em = pd.DataFrame(metas)
    lengths = em["length"].to_numpy()
    em["data/chunk_index"] = 0
    em["data/file_index"] = 0
    em["dataset_from_index"] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    em["dataset_to_index"] = np.cumsum(lengths)
    em[f"videos/{VIDEO_KEY}/chunk_index"] = 0
    em[f"videos/{VIDEO_KEY}/file_index"] = 0
    em["meta/episodes/chunk_index"] = 0
    em["meta/episodes/file_index"] = 0
    (out / "meta/episodes/chunk-000").mkdir(parents=True)
    em.to_parquet(out / "meta/episodes/chunk-000/file-000.parquet", index=False)

    codec = encode_all(frames_out, out / "videos" / VIDEO_KEY / "chunk-000/file-000.mp4", fps)
    for name in ("tasks.parquet", "stats.json"):
        shutil.copy2(src / "meta" / name, out / "meta" / name)
    info["total_episodes"] = len(em)
    info["total_frames"] = int(len(frames))
    info["splits"] = {"train": f"0:{len(em)}"}
    vinfo = info["features"][VIDEO_KEY].get("video_info") or info["features"][VIDEO_KEY].get("info") or {}
    vinfo["video.codec"] = codec
    (out / "meta/info.json").write_text(json.dumps(info, indent=2))
    (out / "corruption.json").write_text(json.dumps(log, indent=1))
    counts = {f: sum(f in e["faults"] for e in log["episodes"]) for f in RATES}
    print(f"wrote {out}: {len(em)} episodes ({n_src} source + {n_dead} dead), {len(frames)} frames; faults {counts}")


if __name__ == "__main__":
    main()
