import json

import numpy as np

import pytest

from trajlint.cli import main
from trajlint.core import run
from trajlint.loader import load
from trajlint.report import to_html

from .synth import make_dataset


def _checks_by_episode(result):
    return {e["episode"]: {i["check"] for i in e["issues"]} for e in result["episodes"]}


def test_clean_dataset_has_no_errors(tmp_path):
    ds = load(str(make_dataset(tmp_path / "clean")))
    result = run(ds, video=False)
    assert result["summary"]["bad"] == 0
    assert len(result["keep"]) == 12
    for e in result["episodes"]:
        assert all(i["severity"] != "error" for i in e["issues"]), e


@pytest.mark.parametrize(
    "fault,check",
    [
        ("dead", "dead_episode"),
        ("frozen", "frozen_state"),
        ("spike", "action_spike"),
        ("gap", "dropped_frames"),
        ("short", "too_short"),
        ("nan", "nan_values"),
        ("pause", "pause"),
        ("conflict", "index_conflict"),
    ],
)
def test_fault_is_detected(tmp_path, fault, check):
    ds = load(str(make_dataset(tmp_path / fault, faults={5: fault})))
    found = _checks_by_episode(run(ds, video=False))
    assert check in found[5]
    # the fault must not leak into other episodes
    assert all(check not in found[e] for e in found if e != 5)


def test_pause_is_not_a_sensor_freeze(tmp_path):
    ds = load(str(make_dataset(tmp_path / "p", faults={5: "pause"})))
    found = _checks_by_episode(run(ds, video=False))
    assert "frozen_state" not in found[5]


def test_duplicate_episode(tmp_path):
    ds = load(str(make_dataset(tmp_path / "dup", faults={7: "duplicate_of:2"})))
    found = _checks_by_episode(run(ds, video=False))
    assert "duplicate" in found[7]
    assert "duplicate" not in found[2]


def test_dead_episode_is_not_kept(tmp_path):
    ds = load(str(make_dataset(tmp_path / "d", faults={3: "dead"})))
    result = run(ds, video=False)
    assert 3 not in result["keep"]
    assert len(result["keep"]) == 11


def test_video_checks(tmp_path):
    pytest.importorskip("av")
    root = make_dataset(tmp_path / "v", video=True, faults={4: "frozen_camera", 8: "dark_camera"})
    ds = load(str(root))
    found = _checks_by_episode(run(ds, video=True))
    assert "frozen_camera" in found[4]
    assert "dark_frames" in found[8]
    assert "frozen_camera" not in found[0]
    assert "video_short" not in set().union(*found.values())


def test_cli_writes_reports(tmp_path):
    root = make_dataset(tmp_path / "c", faults={1: "dead"})
    out = tmp_path / "out"
    code = main(["check", str(root), "--no-video", "-o", str(out), "--fail-on", "bad"])
    assert code == 1
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["bad"] == 1
    assert json.loads((out / "keep_episodes.json").read_text()) == report["keep"]
    assert "trajlint report" in (out / "report.html").read_text()


def test_html_escapes_task_text(tmp_path):
    ds = load(str(make_dataset(tmp_path / "h", faults={2: "spike"})))
    result = run(ds, video=False)
    result["dataset"]["id"] = "<script>x</script>"
    assert "<script>x" not in to_html(result, ds)


def test_small_moving_object_is_not_a_frozen_camera():
    # a 3x3 dot moving on a still background barely changes the mean image (like PushT)
    from trajlint.checks_video import _check_segment

    n = 40
    frames = np.full((n, 48, 64), 200, np.uint8)
    for i in range(n):
        x = 5 + i
        frames[i, 20:23, x:x + 3] = 40
    times = np.arange(n) / 5.0
    track = np.linspace(0, 1, n * 2)[:, None]  # robot moves the whole time (10 fps)
    issues = _check_segment(0, "cam", 0.0, n / 5.0, times, frames, n / 5.0, n * 2, 10.0, track)
    assert not [i for i in issues if i.check == "frozen_camera"]

    still = np.repeat(frames[:1], n, axis=0)
    issues = _check_segment(0, "cam", 0.0, n / 5.0, times, still, n / 5.0, n * 2, 10.0, track)
    assert [i for i in issues if i.check == "frozen_camera"]
