"""Checks that only need the per-frame table (joint states, actions, timestamps)."""

from __future__ import annotations

import numpy as np

from .issues import Issue
from .loader import Dataset, Episode

# Thresholds. Kept in one place so they are easy to tune and document.
STILL_EPS_FRACTION = 0.002  # joint "did not move" if change < 0.2% of its range in the dataset
FROZEN_MIN_S = 2.0  # a frozen stretch in the middle of an episode must last this long
IDLE_EDGE_MIN_S = 1.5  # idle time at the start/end worth trimming
SHORT_FRACTION = 0.25  # shorter than 25% of the median episode
LONG_FACTOR = 3.0  # longer than 3x the median episode
GAP_FACTOR = 1.5  # a timestamp step > 1.5 frame periods means dropped frames
SPIKE_K = 12.0  # spike if the jump is 12x the typical step for that joint
TRACKING_Z = 4.0  # robust z-score for action/state disagreement
DUPLICATE_TOL = 1e-3  # normalized distance under which two episodes are duplicates
ACTION_MOVED_FRACTION = 0.05  # action moved >5% of its range while the state stayed frozen


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of consecutive True values."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


class DatasetStats:
    """Dataset-wide reference values, so each episode is judged against its peers."""

    def __init__(self, ds: Dataset):
        lengths = np.array([e.length for e in ds.episodes]) if ds.episodes else np.array([0])
        self.median_length = float(np.median(lengths))
        self.state_range = self._range([e.state for e in ds.episodes])
        self.action_range = self._range([e.action for e in ds.episodes])
        self.action_step = self._typical_step([e.action for e in ds.episodes])

    @staticmethod
    def _range(arrays: list[np.ndarray | None]) -> np.ndarray | None:
        arrs = [a for a in arrays if a is not None and len(a)]
        if not arrs:
            return None
        allv = np.concatenate(arrs)
        lo = np.nanpercentile(allv, 1, axis=0)
        hi = np.nanpercentile(allv, 99, axis=0)
        return np.maximum(hi - lo, 1e-6)

    @staticmethod
    def _typical_step(arrays: list[np.ndarray | None]) -> np.ndarray | None:
        diffs = [np.abs(np.diff(a, axis=0)) for a in arrays if a is not None and len(a) > 1]
        if not diffs:
            return None
        d = np.concatenate(diffs)
        out = np.zeros(d.shape[1])
        for j in range(d.shape[1]):
            col = d[:, j]
            col = col[np.isfinite(col) & (col > 0)]
            out[j] = np.percentile(col, 90) if col.size else 0.0
        return np.maximum(out, 1e-6)


def _still_mask(x: np.ndarray, rng: np.ndarray) -> np.ndarray:
    """mask[t] = True when no joint moved between frame t-1 and t."""
    step = np.abs(np.diff(x, axis=0)) / rng
    still = np.all(step < STILL_EPS_FRACTION, axis=1)
    return np.concatenate([[True], still])


def check_episode(ep: Episode, ds: Dataset, st: DatasetStats) -> list[Issue]:
    out: list[Issue] = []
    fps = ds.fps
    e = ep.index

    # 0. Two different recordings saved under the same episode index (usually a bad merge)
    if ep.recordings > 1:
        out.append(Issue(e, "index_conflict", "error",
                         f"{ep.recordings} different recordings share episode index {e}; "
                         "only the first was checked"))

    # 1. Missing values
    for name, arr in (("observation.state", ep.state), ("action", ep.action)):
        if arr is not None and not np.all(np.isfinite(arr)):
            n = int((~np.all(np.isfinite(arr), axis=1)).sum())
            out.append(Issue(e, "nan_values", "error", f"{n} frames have NaN/inf in {name}"))

    # 2. Length
    if st.median_length > 0 and ep.length < max(SHORT_FRACTION * st.median_length, fps):
        out.append(
            Issue(e, "too_short", "warning",
                  f"{ep.length} frames ({ep.length / fps:.1f}s) vs median {st.median_length:.0f}")
        )
    elif st.median_length > 0 and ep.length > LONG_FACTOR * st.median_length:
        out.append(
            Issue(e, "too_long", "info",
                  f"{ep.length} frames ({ep.length / fps:.1f}s) vs median {st.median_length:.0f}")
        )

    # 3. Timestamps: going backwards, or gaps (dropped frames)
    ts = ep.timestamps
    if len(ts) > 1:
        dt = np.diff(ts)
        if np.any(dt <= 0):
            out.append(Issue(e, "timestamp_order", "error", f"{int((dt <= 0).sum())} timestamps do not increase"))
        gaps = np.flatnonzero(dt > GAP_FACTOR / fps)
        if gaps.size:
            missing = int(np.round(dt[gaps] * fps - 1).sum())
            sev = "error" if missing > 0.05 * ep.length else "warning"
            first = int(gaps[0])
            out.append(
                Issue(e, "dropped_frames", sev,
                      f"{gaps.size} gaps, about {missing} frames missing",
                      start_s=float(ts[first] - ts[0]), end_s=float(ts[first + 1] - ts[0]))
            )

    # 4. Robot never moved / frozen in the middle / idle at the edges
    if ep.state is not None and st.state_range is not None and ep.length > 1 and np.all(np.isfinite(ep.state)):
        still = _still_mask(ep.state, st.state_range)
        moved = (np.nanmax(ep.state, axis=0) - np.nanmin(ep.state, axis=0)) / st.state_range
        if np.all(moved < 0.01):
            out.append(Issue(e, "dead_episode", "error", "robot state never changes: nothing was recorded"))
        else:
            runs = _runs(still)
            n = ep.length
            for a, b in runs:
                dur = (b - a) / fps
                if a == 0 and dur >= IDLE_EDGE_MIN_S:
                    out.append(Issue(e, "idle_start", "info", f"robot idle for first {dur:.1f}s (trim)",
                                     start_s=0.0, end_s=b / fps))
                elif b == n and dur >= IDLE_EDGE_MIN_S:
                    out.append(Issue(e, "idle_end", "info", f"robot idle for last {dur:.1f}s (trim)",
                                     start_s=a / fps, end_s=n / fps))
                elif a > 0 and b < n and dur >= FROZEN_MIN_S:
                    # If the operator kept moving (action changes) while the readings stayed
                    # identical, the sensor stream froze. If both are still, it is just a pause.
                    commanded = 0.0
                    if ep.action is not None and st.action_range is not None and ep.action.shape[0] == n:
                        seg = ep.action[a:b]
                        commanded = float(np.max((seg.max(axis=0) - seg.min(axis=0)) / st.action_range))
                    if commanded > ACTION_MOVED_FRACTION:
                        out.append(Issue(e, "frozen_state", "warning",
                                         f"joint readings frozen for {dur:.1f}s while the action kept moving",
                                         start_s=a / fps, end_s=b / fps))
                    else:
                        out.append(Issue(e, "pause", "info", f"robot paused for {dur:.1f}s mid-episode",
                                         start_s=a / fps, end_s=b / fps))

    # 5. Teleop glitches: one-frame spikes in the action stream
    if ep.action is not None and st.action_step is not None and ep.length > 2 and np.all(np.isfinite(ep.action)):
        a = ep.action
        d1 = np.abs(a[1:-1] - a[:-2])
        d2 = np.abs(a[2:] - a[1:-1])
        back = np.abs(a[2:] - a[:-2])
        thr = SPIKE_K * st.action_step
        spikes = (np.minimum(d1, d2) > thr) & (back < 0.5 * np.minimum(d1, d2))
        rows = np.flatnonzero(spikes.any(axis=1))
        if rows.size:
            joints = sorted({int(j) for j in np.flatnonzero(spikes.any(axis=0))})
            names = [ds.action_names[j] if j < len(ds.action_names) else f"joint {j}" for j in joints]
            t0 = (rows[0] + 1) / fps
            out.append(Issue(e, "action_spike", "warning",
                             f"{rows.size} one-frame spikes in action ({', '.join(names)})",
                             start_s=t0, end_s=t0 + 1 / fps))

    return out


def motion_tracks(ds: Dataset, st: DatasetStats) -> dict[int, np.ndarray]:
    """Per-episode joint states scaled by each joint's range in the dataset (1.0 = full range)."""
    out = {}
    if st.state_range is None:
        return out
    for ep in ds.episodes:
        if ep.state is not None and ep.length > 1 and np.all(np.isfinite(ep.state)):
            out[ep.index] = ep.state / st.state_range
    return out


def check_tracking(ds: Dataset, st: DatasetStats) -> list[Issue]:
    """Flag episodes where the commanded action and the measured state disagree far more than usual.

    Typical cause: a calibration problem, a loose motor, or action/state columns swapped.
    """
    if not ds.state_names or ds.state_names != ds.action_names or st.state_range is None:
        return []
    errs, idx = [], []
    for ep in ds.episodes:
        if ep.state is None or ep.action is None or ep.length < 3:
            continue
        if ep.state.shape != ep.action.shape or not (np.all(np.isfinite(ep.state)) and np.all(np.isfinite(ep.action))):
            continue
        # compare the command at t with the state a few frames later
        lag = min(3, ep.length - 1)
        err = np.abs(ep.action[:-lag] - ep.state[lag:]) / st.state_range
        errs.append(float(np.median(err.mean(axis=1))))
        idx.append(ep.index)
    if len(errs) < 5:
        return []
    errs_a = np.array(errs)
    med = np.median(errs_a)
    mad = np.median(np.abs(errs_a - med)) * 1.4826 + 1e-9
    out = []
    for e, v in zip(idx, errs_a):
        z = (v - med) / mad
        if z > TRACKING_Z and v > 2 * med:
            out.append(Issue(e, "tracking_error", "warning",
                             f"action and state disagree {v / med:.1f}x more than usual"))
    return out


def check_duplicates(ds: Dataset) -> list[Issue]:
    """Find episodes recorded twice (same action trajectory)."""
    sigs, idx = [], []
    for ep in ds.episodes:
        a = ep.action if ep.action is not None else ep.state
        if a is None or ep.length < 4 or not np.all(np.isfinite(a)):
            continue
        t = np.linspace(0, ep.length - 1, 32)
        res = np.stack([np.interp(t, np.arange(ep.length), a[:, j]) for j in range(a.shape[1])], axis=1)
        sigs.append(np.concatenate([res.ravel(), [ep.length]]))
        idx.append(ep.index)
    if len(sigs) < 2:
        return []
    S = np.stack(sigs)
    scale = np.maximum(S.std(axis=0), 1e-6)
    Z = S / scale
    out = []
    seen = set()
    for i in range(len(Z)):
        d = np.sqrt(((Z[i + 1:] - Z[i]) ** 2).mean(axis=1))
        for k in np.flatnonzero(d < DUPLICATE_TOL):
            j = i + 1 + int(k)
            if idx[j] in seen:
                continue
            seen.add(idx[j])
            out.append(Issue(idx[j], "duplicate", "warning", f"same trajectory as episode {idx[i]}"))
    return out
