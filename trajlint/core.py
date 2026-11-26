from __future__ import annotations

import time
from collections import Counter

from . import __version__
from .checks_tabular import DatasetStats, check_duplicates, check_episode, check_tracking, moving_masks
from .checks_video import check_videos
from .issues import SEVERITY_ORDER, Issue, episode_score, grade
from .loader import Dataset

CHECK_HELP = {
    "index_conflict": "Two different recordings share one episode index (bad merge): training sees the wrong frames",
    "pause": "Robot holds still for a while mid-episode (often normal)",
    "nan_values": "NaN or infinite numbers in joint states or actions",
    "too_short": "Episode much shorter than the others (often an aborted take)",
    "too_long": "Episode much longer than the others (often forgot to stop recording)",
    "timestamp_order": "Timestamps go backwards or repeat",
    "dropped_frames": "Gaps in timestamps: frames were lost while recording",
    "dead_episode": "Robot state never changes: nothing was recorded",
    "frozen_state": "Joint readings stuck while the action kept moving (sensor or bus dropout)",
    "idle_start": "Robot sits still at the start: can be trimmed",
    "idle_end": "Robot sits still at the end: can be trimmed",
    "action_spike": "One-frame jumps in the action stream (teleop glitch)",
    "tracking_error": "Action and state disagree far more than in other episodes (calibration?)",
    "duplicate": "Same trajectory as another episode (recorded or merged twice)",
    "video_missing": "Video file for this camera is missing",
    "video_unreadable": "Video file cannot be decoded",
    "video_short": "Video ends before the episode does",
    "frozen_camera": "Camera image stops changing while the robot moves",
    "dark_frames": "Camera image almost black",
    "overexposed": "Camera image almost white",
}


def run(ds: Dataset, video: bool = True, progress=None) -> dict:
    t0 = time.time()
    st = DatasetStats(ds)
    issues: list[Issue] = []
    for ep in ds.episodes:
        issues.extend(check_episode(ep, ds, st))
    issues.extend(check_tracking(ds, st))
    issues.extend(check_duplicates(ds))
    if video and ds.camera_keys:
        issues.extend(check_videos(ds, progress=progress, moving=moving_masks(ds, st)))

    by_ep: dict[int, list[Issue]] = {ep.index: [] for ep in ds.episodes}
    for i in issues:
        by_ep.setdefault(i.episode, []).append(i)

    episodes = []
    for ep in ds.episodes:
        iss = sorted(by_ep[ep.index], key=lambda i: (SEVERITY_ORDER[i.severity], i.check))
        score = episode_score(iss)
        episodes.append({
            "episode": ep.index,
            "length": ep.length,
            "seconds": round(ep.length / ds.fps, 2),
            "task": ep.task,
            "score": score,
            "grade": grade(iss),
            "issues": [i.to_dict() for i in iss],
        })

    counts = Counter(i.check for i in issues)
    eps_per_check = {c: len({i.episode for i in issues if i.check == c}) for c in counts}
    grades = Counter(e["grade"] for e in episodes)
    return {
        "tool": {"name": "trajlint", "version": __version__},
        "dataset": {
            "id": ds.repo_id,
            "version": ds.version,
            "robot_type": ds.robot_type,
            "fps": ds.fps,
            "episodes": len(ds.episodes),
            "frames": ds.total_frames,
            "cameras": ds.camera_keys,
            "video_checked": bool(video and ds.camera_keys),
        },
        "summary": {
            "ok": grades.get("ok", 0),
            "review": grades.get("review", 0),
            "bad": grades.get("bad", 0),
            "checks": {
                c: {"episodes": eps_per_check[c], "help": CHECK_HELP.get(c, "")}
                for c in sorted(counts, key=lambda c: -eps_per_check[c])
            },
        },
        "keep": [e["episode"] for e in episodes if e["grade"] != "bad"],
        "episodes": episodes,
        "seconds_taken": round(time.time() - t0, 1),
    }
