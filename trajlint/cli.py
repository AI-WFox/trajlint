from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core import run
from .loader import load
from .report import to_html


def _progress(i: int, n: int, name: str) -> None:
    print(f"\r  decoding video {i}/{n} ({name})", end="", file=sys.stderr, flush=True)
    if i == n:
        print(file=sys.stderr)


def _print_summary(result: dict) -> None:
    d, s = result["dataset"], result["summary"]
    print(f"\n{d['id']}  ({d['episodes']} episodes, {d['frames']} frames, {d['fps']:g} fps)")
    print(f"  clean: {s['ok']}   needs a look: {s['review']}   likely broken: {s['bad']}")
    if s["checks"]:
        print("\n  problem                episodes")
        for c, v in s["checks"].items():
            print(f"  {c:<22} {v['episodes']:>5}   {v['help']}")
    worst = [e for e in sorted(result["episodes"], key=lambda e: e["score"]) if e["grade"] == "bad"][:10]
    if worst:
        print("\n  worst episodes: " + ", ".join(f"{e['episode']} ({e['score']})" for e in worst))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="trajlint",
        description="Find broken episodes in LeRobot datasets before you train on them.",
    )
    p.add_argument("--version", action="version", version=f"trajlint {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="check a dataset and write a report")
    c.add_argument("dataset", help="Hugging Face repo id (user/name) or a local folder")
    c.add_argument("--no-video", action="store_true", help="skip camera checks (much faster, no video download)")
    c.add_argument("-o", "--out", default="trajlint-report", help="output folder (default: trajlint-report)")
    c.add_argument("--cache-dir", default=None, help="where to cache Hub downloads")
    c.add_argument("--fail-on", choices=["never", "bad", "review"], default="never",
                   help="exit with code 1 if any episode has this grade (for CI)")

    f = sub.add_parser("filter", help="write a cleaned copy of a v3.0 dataset")
    f.add_argument("dataset", help="Hugging Face repo id (user/name) or a local folder")
    f.add_argument("out", help="folder for the cleaned dataset (must be empty or missing)")
    f.add_argument("--drop", choices=["bad", "review"], default="bad",
                   help="drop 'bad' episodes (default) or also those that 'need a look'")
    f.add_argument("--no-trim", action="store_true", help="do not trim idle time at episode start/end")
    f.add_argument("--no-video", action="store_true", help="skip camera checks when deciding what to drop")
    f.add_argument("--cache-dir", default=None)

    args = p.parse_args(argv)
    if args.cmd == "filter":
        from .filter import plan, write

        print(f"loading {args.dataset} ...", file=sys.stderr)
        ds = load(args.dataset, with_videos=True, cache_dir=args.cache_dir)
        result = run(ds, video=not args.no_video, progress=_progress)
        drop = ("bad",) if args.drop == "bad" else ("bad", "review")
        keep = plan(result, ds.fps, drop=drop, trim_idle=not args.no_trim)
        summary = write(ds, keep, Path(args.out))
        cut = summary["frames_before"] - summary["frames_after"]
        print(f"\n  kept {summary['episodes_after']} of {summary['episodes_before']} episodes, "
              f"{summary['frames_after']} of {summary['frames_before']} frames "
              f"({cut / max(summary['frames_before'], 1):.0%} removed)")
        print(f"  cleaned dataset: {args.out}")
        return 0
    if args.cmd == "check":
        print(f"loading {args.dataset} ...", file=sys.stderr)
        ds = load(args.dataset, with_videos=not args.no_video, cache_dir=args.cache_dir)
        result = run(ds, video=not args.no_video, progress=_progress)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(result, indent=2))
        (out / "report.html").write_text(to_html(result, ds))
        (out / "keep_episodes.json").write_text(json.dumps(result["keep"]))
        _print_summary(result)
        print(f"\n  report: {out / 'report.html'}")
        s = result["summary"]
        if args.fail_on == "bad" and s["bad"]:
            return 1
        if args.fail_on == "review" and (s["bad"] or s["review"]):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
