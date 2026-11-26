"""Camera checks: frozen camera, black/overexposed frames, video shorter than the episode.

Each video file is decoded once. Frames are shrunk to 64x48 grayscale and
sampled at about SAMPLE_FPS, which keeps memory small even for long files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .issues import Issue
from .loader import Dataset

SAMPLE_FPS = 5.0
FROZEN_DIFF = 0.6  # mean absolute pixel change (0-255) below which two samples look identical
FROZEN_MIN_S = 2.0
DARK_LEVEL = 12.0
BRIGHT_LEVEL = 245.0
EXPOSURE_MIN_S = 1.0
SHORT_TOLERANCE_FRAMES = 3
MOVING_MIN = 0.5  # the robot must be moving for at least half of a frozen stretch


def _decode(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (times, frames[N,48,64] uint8, duration) for a video file."""
    import av

    times, frames = [], []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        if _keyframes_are_dense(path):
            # LeRobot encodes with a keyframe every few frames, so decoding only
            # keyframes still gives plenty of samples and is much faster.
            stream.codec_context.skip_frame = "NONKEY"
        next_t = -1.0
        last_t = 0.0
        for f in c.decode(stream):
            t = float(f.time) if f.time is not None else last_t
            last_t = t
            if t + 1e-9 < next_t:
                continue
            next_t = t + 1.0 / SAMPLE_FPS
            frames.append(f.reformat(width=64, height=48, format="gray").to_ndarray())
            times.append(t)
        duration = last_t
    if not frames:
        return np.zeros(0), np.zeros((0, 48, 64), np.uint8), 0.0
    return np.array(times), np.stack(frames), duration


def _keyframes_are_dense(path: Path, probe_packets: int = 300) -> bool:
    """True if keyframes come at least every 1/SAMPLE_FPS seconds (checked on the first packets)."""
    import av

    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        key_times = []
        for i, pkt in enumerate(c.demux(stream)):
            if i >= probe_packets:
                break
            if pkt.is_keyframe and pkt.pts is not None:
                key_times.append(float(pkt.pts * stream.time_base))
    if len(key_times) < 3:
        return False
    gaps = np.diff(sorted(key_times))
    return bool(np.median(gaps) <= 1.0 / SAMPLE_FPS)


def check_videos(ds: Dataset, progress=None, moving: dict[int, np.ndarray] | None = None) -> list[Issue]:
    """`moving[episode]` is a per-frame bool array: True where the robot was moving.

    A camera that shows no change while the robot sits still is just a static scene,
    so a frozen image only counts while the robot is moving.
    """
    moving = moving or {}
    out: list[Issue] = []
    if not ds.camera_keys:
        return out

    # group episodes by file so each file is decoded once
    by_file: dict[Path, list[tuple[int, str, float, float]]] = {}
    for ep in ds.episodes:
        for key in ds.camera_keys:
            seg = ep.videos.get(key)
            if seg is None:
                out.append(Issue(ep.index, "video_missing", "error", "no video found for this episode", camera=key))
                continue
            by_file.setdefault(seg.path, []).append((ep.index, key, seg.start, seg.end))

    lengths = {ep.index: ep.length for ep in ds.episodes}
    for n, (path, segs) in enumerate(sorted(by_file.items())):
        if progress:
            progress(n + 1, len(by_file), path.name)
        try:
            times, frames, duration = _decode(path)
        except Exception as exc:  # corrupt file, unsupported codec
            for e, key, _, _ in segs:
                out.append(Issue(e, "video_unreadable", "error", f"cannot decode video: {exc}", camera=key))
            continue
        for e, key, start, end in segs:
            out.extend(_check_segment(e, key, start, end, times, frames, duration, lengths[e], ds.fps,
                                      moving.get(e)))
    return out


def _moving_fraction(mov: np.ndarray | None, t0: float, t1: float, fps: float) -> float:
    if mov is None or len(mov) == 0:
        return 1.0  # no joint data: judge the image alone
    a = int(max(0, np.floor(t0 * fps)))
    b = int(min(len(mov), np.ceil(t1 * fps) + 1))
    return float(mov[a:b].mean()) if b > a else 0.0


def _check_segment(e, key, start, end, times, frames, duration, n_frames, fps, mov=None) -> list[Issue]:
    out: list[Issue] = []
    expected = n_frames / fps
    have = min(end, duration + 1.0 / fps) - start
    if have < expected - SHORT_TOLERANCE_FRAMES / fps:
        out.append(Issue(e, "video_short", "error",
                         f"video covers {max(have, 0):.1f}s but episode is {expected:.1f}s", camera=key))
    sel = (times >= start - 1e-6) & (times < end - 1e-6)
    t = times[sel] - start
    f = frames[sel].astype(np.float32)
    if len(f) < 2:
        return out

    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    frozen = np.concatenate([[False], diff < FROZEN_DIFF])
    span = t[-1] - t[0]
    runs = _runs(frozen)
    whole = any(a <= 1 and b >= len(f) for a, b in runs)
    if whole and span >= FROZEN_MIN_S and _moving_fraction(mov, 0, span, fps) > MOVING_MIN:
        out.append(Issue(e, "frozen_camera", "error",
                         "camera image never changes although the robot moves", camera=key))
    elif not whole:
        for a, b in runs:
            t0, t1 = t[max(a - 1, 0)], t[b - 1]
            if t1 - t0 >= FROZEN_MIN_S and _moving_fraction(mov, t0, t1, fps) > MOVING_MIN:
                out.append(Issue(e, "frozen_camera", "warning",
                                 f"camera image frozen for {t1 - t0:.1f}s while the robot moves",
                                 start_s=float(t0), end_s=float(t1), camera=key))
                break

    level = f.mean(axis=(1, 2))
    for name, mask, sev_msg in (
        ("dark_frames", level < DARK_LEVEL, "image almost black"),
        ("overexposed", level > BRIGHT_LEVEL, "image almost white"),
    ):
        for a, b in _runs(mask):
            t0, t1 = t[a], t[b - 1] + 1.0 / SAMPLE_FPS
            if t1 - t0 >= EXPOSURE_MIN_S:
                sev = "error" if (b - a) >= 0.9 * len(f) else "warning"
                out.append(Issue(e, name, sev, f"{sev_msg} for {t1 - t0:.1f}s",
                                 start_s=float(t0), end_s=float(t1), camera=key))
                break
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
