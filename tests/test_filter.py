import json

import pandas as pd
import pytest

from trajlint.cli import main
from trajlint.core import run
from trajlint.filter import plan, write
from trajlint.loader import load

from .synth import FPS, make_dataset


def test_filter_drops_bad_and_trims_idle(tmp_path):
    pytest.importorskip("av")
    src = make_dataset(tmp_path / "src", video=True, faults={2: "dead", 4: "idle_start"})
    ds = load(str(src))
    keep = plan(run(ds, video=False), ds.fps)
    assert [k["episode"] for k in keep] == [e for e in range(12) if e != 2]
    trimmed = next(k for k in keep if k["episode"] == 4)
    assert 15 <= trimmed["start"] <= 18  # 2 s idle minus a 0.3 s margin

    out = tmp_path / "out"
    summary = write(ds, keep, out)
    assert summary["episodes_after"] == 11

    clean = load(str(out))
    assert [e.index for e in clean.episodes] == list(range(11))
    result = run(clean, video=True)
    assert result["summary"]["bad"] == 0
    assert all(i["check"] != "idle_start" for e in result["episodes"] for i in e["issues"])

    # episode 4 became episode 3; its video window moved forward by the trimmed frames
    src_meta = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    new_meta = pd.read_parquet(out / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    col = "videos/observation.images.front/from_timestamp"
    assert new_meta.loc[3, col] == pytest.approx(src_meta.loc[4, col] + trimmed["start"] / FPS)
    frames = pd.read_parquet(out / "data/chunk-000/file-000.parquet")
    assert frames["index"].tolist() == list(range(len(frames)))
    info = json.loads((out / "meta/info.json").read_text())
    assert info["total_episodes"] == 11 and info["total_frames"] == len(frames)


def test_filter_cli_refuses_non_empty_output(tmp_path):
    src = make_dataset(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("keep me")
    with pytest.raises(FileExistsError):
        main(["filter", str(src), str(out), "--no-video"])
