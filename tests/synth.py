"""Build tiny LeRobot v3.0 datasets with known faults, for tests and demos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 10
NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "gripper.pos"]


def _clean_episode(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n)[:, None]
    phase = rng.uniform(0, 2 * np.pi, size=(1, len(NAMES)))
    amp = rng.uniform(20, 40, size=(1, len(NAMES)))
    action = amp * np.sin(2 * np.pi * t + phase) + rng.normal(0, 0.3, size=(n, len(NAMES)))
    state = np.vstack([action[:1], action[:-1]]) + rng.normal(0, 0.2, size=(n, len(NAMES)))
    return state, action


def make_dataset(root: Path, n_episodes: int = 12, length: int = 60, faults: dict | None = None,
                 video: bool = False, seed: int = 0) -> Path:
    """faults maps an episode index to one of: dead, frozen, pause, spike, gap, short, nan, conflict, duplicate_of:<i>,
    frozen_camera, dark_camera."""
    faults = faults or {}
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    rows, eps = [], []
    actions: dict[int, np.ndarray] = {}
    states: dict[int, np.ndarray] = {}
    global_index = 0
    video_t = 0.0
    lengths = {}
    for e in range(n_episodes):
        f = faults.get(e, "")
        n = 8 if f == "short" else length
        state, action = _clean_episode(rng, n)
        if f == "dead":
            state[:] = state[0]
            action[:] = state[0]
        if f == "frozen":  # sensor stream froze, operator kept moving
            state[20:45] = state[20]
        if f == "pause":  # robot really held still
            state[20:45] = state[20]
            action[20:45] = action[20]
        if f == "idle_start":  # robot waits 2 s before moving
            state[:20] = state[20]
            action[:20] = action[20]
        if f == "spike":
            action[30, 1] += 400.0
        if f == "nan":
            state[10, 0] = np.nan
        if f.startswith("duplicate_of:"):
            src = int(f.split(":")[1])
            action, state = actions[src].copy(), states[src].copy()
            n = len(action)
        actions[e], states[e] = action, state
        ts = np.arange(n) / FPS
        if f == "gap":
            ts[30:] += 1.0  # 10 frames lost
        lengths[e] = n
        for i in range(n):
            rows.append({
                "action": action[i].astype(np.float32),
                "observation.state": state[i].astype(np.float32),
                "timestamp": np.float32(ts[i]),
                "frame_index": i,
                "episode_index": e,
                "index": global_index,
                "task_index": 0,
            })
            global_index += 1
        ep = {
            "episode_index": e,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_index - n,
            "dataset_to_index": global_index,
            "tasks": ["pick the cube"],
            "length": n,
        }
        if video:
            ep.update({
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 0,
                "videos/observation.images.front/from_timestamp": video_t,
                "videos/observation.images.front/to_timestamp": video_t + n / FPS,
            })
            video_t += n / FPS
        eps.append(ep)

    # a second, different recording saved under an existing episode index (merge bug)
    for e, f in faults.items():
        if f == "conflict":
            state, action = _clean_episode(rng, length)
            for i in range(length):
                rows.append({
                    "action": action[i].astype(np.float32),
                    "observation.state": state[i].astype(np.float32),
                    "timestamp": np.float32(i / FPS),
                    "frame_index": i,
                    "episode_index": e,
                    "index": global_index,
                    "task_index": 0,
                })
                global_index += 1
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(eps).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick the cube"], name="task")).to_parquet(
        root / "meta" / "tasks.parquet")

    features = {
        "action": {"dtype": "float32", "shape": [len(NAMES)], "names": NAMES},
        "observation.state": {"dtype": "float32", "shape": [len(NAMES)], "names": NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if video:
        features["observation.images.front"] = {"dtype": "video", "shape": [48, 64, 3],
                                                "names": ["height", "width", "channels"]}
        _write_video(root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4",
                     lengths, faults, rng)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "total_episodes": n_episodes,
        "total_frames": global_index,
        "fps": FPS,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return root


def _write_video(path: Path, lengths: dict[int, int], faults: dict, rng: np.random.Generator) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.codec_context.qscale = 2
        for e in sorted(lengths):
            f = faults.get(e, "")
            frozen = rng.integers(40, 200, size=(48, 64, 3), dtype=np.uint8)
            for i in range(lengths[e]):
                if f == "frozen_camera":
                    img = frozen
                elif f == "dark_camera":
                    img = np.full((48, 64, 3), 3, np.uint8)
                else:
                    img = rng.integers(40, 200, size=(48, 64, 3), dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                for pkt in s.encode(frame):
                    c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
