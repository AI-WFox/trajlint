"""Read LeRobot datasets (v3.0 and v2.x) from a local folder or the Hugging Face Hub.

Only the parts the checks need are loaded: metadata, the per-frame table, and
(optionally) the camera videos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

META_PATTERNS = ["meta/*", "meta/**/*", "README.md"]
DATA_PATTERNS = ["data/**/*.parquet"]
VIDEO_PATTERNS = ["videos/**/*.mp4"]


@dataclass
class VideoSegment:
    """Where one episode's frames live inside a camera video file."""

    path: Path
    start: float  # seconds into the file
    end: float  # seconds into the file


@dataclass
class Episode:
    index: int
    length: int
    timestamps: np.ndarray  # shape (T,)
    state: np.ndarray | None  # shape (T, D)
    action: np.ndarray | None  # shape (T, D)
    task: str = ""
    videos: dict[str, VideoSegment] = field(default_factory=dict)
    # >1 when several different recordings share this episode_index (a merge bug)
    recordings: int = 1


@dataclass
class Dataset:
    root: Path
    repo_id: str
    version: str
    fps: float
    robot_type: str
    state_names: list[str]
    action_names: list[str]
    camera_keys: list[str]
    episodes: list[Episode]

    @property
    def total_frames(self) -> int:
        return int(sum(e.length for e in self.episodes))


def _feature_names(feature: dict | None) -> list[str]:
    if not feature:
        return []
    names = feature.get("names")
    if isinstance(names, dict):  # e.g. {"motors": [...]}
        for v in names.values():
            if isinstance(v, list):
                return [str(x) for x in v]
        return []
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def resolve(source: str, with_videos: bool = True, cache_dir: str | None = None) -> Path:
    """Return a local folder for `source`, downloading from the Hub if it is not a path."""
    p = Path(source).expanduser()
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    patterns = META_PATTERNS + DATA_PATTERNS + (VIDEO_PATTERNS if with_videos else [])
    local = snapshot_download(
        repo_id=source,
        repo_type="dataset",
        allow_patterns=patterns,
        cache_dir=cache_dir,
    )
    return Path(local)


def _stack(col: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in col.to_numpy()])


def load(source: str, with_videos: bool = True, cache_dir: str | None = None) -> Dataset:
    root = resolve(source, with_videos=with_videos, cache_dir=cache_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found: is this a LeRobot dataset?")
    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", "unknown"))
    features = info.get("features", {})
    fps = float(info.get("fps") or 30)
    camera_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if not with_videos:
        camera_keys = []

    frames = _read_frames(root)
    tasks = _read_tasks(root)
    if version.startswith("v3"):
        meta_eps = _read_v3_episodes(root)
    else:
        meta_eps = _read_v2_episodes(root)

    has_state = "observation.state" in frames.columns
    has_action = "action" in frames.columns
    episodes: list[Episode] = []
    for ep_idx, g in frames.groupby("episode_index", sort=True):
        recordings = 1
        if "frame_index" in g.columns:
            recordings = max(1, int((g["frame_index"].to_numpy() == 0).sum()))
            if recordings > 1:
                # keep the first recording only; the conflict is reported as its own issue
                order = g.sort_values("index") if "index" in g.columns else g
                starts = list(np.flatnonzero(order["frame_index"].to_numpy() == 0)) + [len(order)]
                g = order.iloc[starts[0]:starts[1]]
            else:
                g = g.sort_values("frame_index")
        ts = g["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in g.columns else np.arange(len(g)) / fps
        state = _stack(g["observation.state"]) if has_state else None
        action = _stack(g["action"]) if has_action else None
        if state is not None and state.ndim == 1:
            state = state[:, None]
        if action is not None and action.ndim == 1:
            action = action[:, None]
        meta = meta_eps.get(int(ep_idx), {})
        task = ""
        if "task_index" in g.columns and len(g):
            task = tasks.get(int(g["task_index"].iloc[0]), "")
        if not task and meta.get("tasks"):
            t = meta["tasks"]
            task = t[0] if isinstance(t, (list, np.ndarray)) and len(t) else str(t)
        videos = {}
        for key in camera_keys:
            seg = _video_segment(root, info, key, int(ep_idx), meta, version, fps, len(g))
            if seg is not None:
                videos[key] = seg
        episodes.append(
            Episode(
                index=int(ep_idx),
                length=len(g),
                timestamps=ts,
                state=state,
                action=action,
                task=str(task),
                videos=videos,
                recordings=recordings,
            )
        )

    return Dataset(
        root=root,
        repo_id=source,
        version=version,
        fps=fps,
        robot_type=str(info.get("robot_type") or "unknown"),
        state_names=_feature_names(features.get("observation.state")),
        action_names=_feature_names(features.get("action")),
        camera_keys=camera_keys,
        episodes=episodes,
    )


def _read_frames(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    wanted = {"observation.state", "action", "timestamp", "frame_index", "episode_index", "task_index", "index"}
    parts = []
    for f in files:
        import pyarrow.parquet as pq

        cols = [c for c in pq.read_schema(f).names if c in wanted]
        parts.append(pd.read_parquet(f, columns=cols))
    return pd.concat(parts, ignore_index=True)


def _read_tasks(root: Path) -> dict[int, str]:
    pq_path = root / "meta" / "tasks.parquet"
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        if "task_index" in df.columns:
            if "task" in df.columns:
                return {int(i): str(t) for i, t in zip(df["task_index"], df["task"])}
            return {int(i): str(t) for t, i in zip(df.index, df["task_index"])}
    jl = root / "meta" / "tasks.jsonl"
    if jl.exists():
        out = {}
        for line in jl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[int(row["task_index"])] = str(row.get("task", ""))
        return out
    return {}


def _read_v3_episodes(root: Path) -> dict[int, dict]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    out: dict[int, dict] = {}
    for f in files:
        df = pd.read_parquet(f, columns=None)
        keep = [c for c in df.columns if not c.startswith("stats/")]
        for row in df[keep].to_dict("records"):
            # if an index appears twice (merge bug) keep the first, matching the frames we keep
            out.setdefault(int(row["episode_index"]), row)
    return out


def _read_v2_episodes(root: Path) -> dict[int, dict]:
    jl = root / "meta" / "episodes.jsonl"
    out: dict[int, dict] = {}
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[int(row["episode_index"])] = row
    return out


def _video_segment(
    root: Path, info: dict, key: str, ep_idx: int, meta: dict, version: str, fps: float, n_frames: int
) -> VideoSegment | None:
    tmpl = info.get("video_path")
    if not tmpl:
        return None
    chunks_size = int(info.get("chunks_size") or 1000)
    if version.startswith("v3"):
        chunk = meta.get(f"videos/{key}/chunk_index")
        file_idx = meta.get(f"videos/{key}/file_index")
        start = meta.get(f"videos/{key}/from_timestamp")
        end = meta.get(f"videos/{key}/to_timestamp")
        if chunk is None or file_idx is None:
            return None
        rel = tmpl.format(video_key=key, chunk_index=int(chunk), file_index=int(file_idx))
        start = float(start or 0.0)
        end = float(end) if end is not None else start + n_frames / fps
    else:
        rel = tmpl.format(video_key=key, episode_chunk=ep_idx // chunks_size, episode_index=ep_idx)
        start, end = 0.0, n_frames / fps
    path = root / rel
    if not path.exists():
        return None
    return VideoSegment(path=path, start=start, end=end)
