"""Write a cleaned copy of a LeRobot v3.0 dataset.

* drops episodes by grade ("bad", or "bad" and "review")
* optionally trims idle time at the start and end of each episode
* renumbers episodes and frames so the result loads like any LeRobot dataset

Videos are never re-encoded. In v3.0 each episode points into a shared video
file with from/to timestamps, so trimming only moves those timestamps and the
original video files are copied (or hard-linked) as they are.

Per-episode statistics and meta/stats.json are kept from the source dataset.
They describe the full recordings, which is close enough for normalization;
recompute them with LeRobot if you need exact values.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import Dataset

TRIM_MARGIN_S = 0.3  # keep a little still time before the motion starts and after it ends


def plan(result: dict, fps: float, drop: tuple[str, ...] = ("bad",), trim_idle: bool = True) -> list[dict]:
    """Decide, per kept episode, which frame range [start, end) to keep."""
    out = []
    margin = int(round(TRIM_MARGIN_S * fps))
    for ep in sorted(result["episodes"], key=lambda e: e["episode"]):
        if ep["grade"] in drop:
            continue
        start, end = 0, ep["length"]
        if trim_idle:
            for i in ep["issues"]:
                if i["check"] == "idle_start" and "end_s" in i:
                    start = max(0, int(round(i["end_s"] * fps)) - margin)
                if i["check"] == "idle_end" and "start_s" in i:
                    end = min(ep["length"], int(round(i["start_s"] * fps)) + margin)
        if end - start < 2:
            continue
        out.append({"episode": ep["episode"], "start": start, "end": end, "length": ep["length"]})
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copy2(src.resolve(), dst)


def write(ds: Dataset, keep: list[dict], out: Path) -> dict:
    if not ds.version.startswith("v3"):
        raise NotImplementedError(f"filter supports LeRobot v3.0 datasets only (this one is {ds.version})")
    src = ds.root
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    info = json.loads((src / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    frames = pd.concat([pd.read_parquet(f) for f in sorted((src / "data").rglob("*.parquet"))], ignore_index=True)
    meta = pd.concat([pd.read_parquet(f) for f in sorted((src / "meta" / "episodes").rglob("*.parquet"))],
                     ignore_index=True)
    meta = meta.drop_duplicates("episode_index", keep="first").set_index("episode_index")
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    parts, rows = [], []
    gidx = 0
    for new_idx, k in enumerate(keep):
        e, a, b = k["episode"], k["start"], k["end"]
        g = frames[frames["episode_index"] == e].sort_values("frame_index")
        g = g[(g["frame_index"] >= a) & (g["frame_index"] < b)].copy()
        n = len(g)
        t0 = float(g["timestamp"].iloc[0])
        g["timestamp"] = (g["timestamp"] - t0).astype(g["timestamp"].dtype)
        g["frame_index"] = np.arange(n, dtype=g["frame_index"].dtype)
        g["episode_index"] = np.full(n, new_idx, dtype=g["episode_index"].dtype)
        if "index" in g.columns:
            g["index"] = np.arange(gidx, gidx + n, dtype=g["index"].dtype)
        parts.append(g)

        row = meta.loc[e].to_dict()
        row.update({
            "episode_index": new_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": gidx,
            "dataset_to_index": gidx + n,
            "length": n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        })
        for key in video_keys:
            col = f"videos/{key}/from_timestamp"
            if col in row and row[col] is not None:
                start = float(row[col]) + a / fps
                row[col] = start
                row[f"videos/{key}/to_timestamp"] = start + n / fps
        rows.append(row)
        gidx += n

    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out / "data" / "chunk-000" / "file-000.parquet", index=False)
    new_meta = pd.DataFrame(rows)
    new_meta = new_meta[["episode_index"] + [c for c in new_meta.columns if c != "episode_index"]]
    new_meta.to_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    for name in ("tasks.parquet", "tasks.jsonl", "stats.json"):
        if (src / "meta" / name).exists():
            shutil.copy2(src / "meta" / name, out / "meta" / name)
    for vid in sorted((src / "videos").rglob("*.mp4")) if (src / "videos").exists() else []:
        _link_or_copy(vid, out / vid.relative_to(src))

    info["total_episodes"] = len(keep)
    info["total_frames"] = gidx
    info["splits"] = {"train": f"0:{len(keep)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    summary = {
        "source": ds.repo_id,
        "episodes_before": len(ds.episodes),
        "episodes_after": len(keep),
        "frames_before": ds.total_frames,
        "frames_after": gidx,
        "kept": [{"new": i, **k} for i, k in enumerate(keep)],
    }
    (out / "trajlint-filter.json").write_text(json.dumps(summary, indent=2))
    return summary
