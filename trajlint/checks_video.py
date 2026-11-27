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
DISPLACEMENT_MIN = 0.10  # ...and some joint must travel >10% of its range, enough to show in the image
STILL_STEP = 0.002  # per-frame change (fraction of range) below which a joint counts as still
CHANGE_LEVEL = 25.0  # a pixel "changed" if it moved by more than this (0-255)
CHANGED_MIN = 0.001  # ...and a frame changed if more than 0.1% of pixels did (small objects count)
BLANK_STD = 6.0  # an almost black/white frame is also almost uniform (a white background alone is fine)


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


def check_videos(ds: Dataset, progress=None, motion: dict[int, np.ndarray] | None = None) -> list[Issue]:
    """`motion[episode]` holds joint states scaled to their dataset range (shape T x D).

    A camera that shows no change while the robot sits still, or barely moves, is
    just a static scene, so a frozen image only counts while the robot clearly moves.
    """
    motion = motion or {}
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
                                      motion.get(e)))
    return out


def _robot_moves(track: np.ndarray | None, t0: float, t1: float, fps: float) -> bool:
    """True if the robot was clearly moving between t0 and t1 (seconds from episode start)."""
    if track is None or len(track) < 2:
        return True  # no joint data: judge the image alone
    a = int(max(0, np.floor(t0 * fps)))
    b = int(min(len(track), np.ceil(t1 * fps) + 1))
    if b - a < 2:
        return False
    seg = track[a:b]
    moving = np.any(np.abs(np.diff(seg, axis=0)) > STILL_STEP, axis=1)
    travel = float(np.max(seg.max(axis=0) - seg.min(axis=0)))
    return bool(moving.mean() > MOVING_MIN and travel > DISPLACEMENT_MIN)


def _check_segment(e, key, start, end, times, frames, duration, n_frames, fps, track=None) -> list[Issue]:
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

    absdiff = np.abs(np.diff(f, axis=0))
    # mean change misses a small object moving on a still background, so also count changed pixels
    still = (absdiff.mean(axis=(1, 2)) < FROZEN_DIFF) & ((absdiff > CHANGE_LEVEL).mean(axis=(1, 2)) <= CHANGED_MIN)
    frozen = np.concatenate([[False], still])
    span = t[-1] - t[0]
    runs = _runs(frozen)
    whole = any(a <= 1 and b >= len(f) for a, b in runs)
    if whole and span >= FROZEN_MIN_S and _robot_moves(track, 0, span, fps):
        out.append(Issue(e, "frozen_camera", "error",
                         "camera image never changes although the robot moves", camera=key))
    elif not whole:
        for a, b in runs:
            t0, t1 = t[max(a - 1, 0)], t[b - 1]
            if t1 - t0 >= FROZEN_MIN_S and _robot_moves(track, t0, t1, fps):
                out.append(Issue(e, "frozen_camera", "warning",
                                 f"camera image frozen for {t1 - t0:.1f}s while the robot moves",
                                 start_s=float(t0), end_s=float(t1), camera=key))
                break

    level = f.mean(axis=(1, 2))
    uniform = f.std(axis=(1, 2)) < BLANK_STD
    for name, mask, sev_msg in (
        ("dark_frames", (level < DARK_LEVEL) & uniform, "image almost black"),
        ("overexposed", (level > BRIGHT_LEVEL) & uniform, "image almost white"),
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
