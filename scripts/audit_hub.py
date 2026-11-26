"""Audit many public LeRobot datasets on the Hugging Face Hub and summarize what breaks.

Example:
    python scripts/audit_hub.py --search so101 --limit 50 --out audit/
Runs table checks only (no video download) unless --video is given.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from trajlint.core import run
from trajlint.loader import load


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--search", action="append", default=[], help="Hub search term (repeatable)")
    p.add_argument("--limit", type=int, default=50, help="datasets per search term")
    p.add_argument("--max-frames", type=int, default=300_000, help="skip bigger datasets")
    p.add_argument("--video", action="store_true")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out", default="audit")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    ids: list[str] = []
    for term in args.search or [None]:
        for d in api.list_datasets(filter="LeRobot", search=term, sort="downloads", limit=args.limit):
            if d.id not in ids:
                ids.append(d.id)

    rows = []
    for rid in ids:
        row = {"id": rid}
        try:
            info = json.loads(Path(hf_hub_download(rid, "meta/info.json", repo_type="dataset",
                                                   cache_dir=args.cache_dir)).read_text())
            row["version"] = info.get("codebase_version")
            row["frames"] = info.get("total_frames")
            if (info.get("total_frames") or 0) > args.max_frames:
                row["skipped"] = "too big"
                rows.append(row)
                continue
            ds = load(rid, with_videos=args.video, cache_dir=args.cache_dir)
            res = run(ds, video=args.video)
            (out / (rid.replace("/", "__") + ".json")).write_text(json.dumps(res))
            row.update({
                "episodes": res["dataset"]["episodes"],
                "ok": res["summary"]["ok"],
                "review": res["summary"]["review"],
                "bad": res["summary"]["bad"],
                "checks": {c: v["episodes"] for c, v in res["summary"]["checks"].items()},
            })
            print(f"{rid}: {row['episodes']} eps, bad {row['bad']}, review {row['review']}", flush=True)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            (out / "errors.log").open("a").write(f"== {rid}\n{traceback.format_exc()}\n")
            print(f"{rid}: ERROR {row['error'][:120]}", flush=True)
        rows.append(row)

    done = [r for r in rows if "episodes" in r]
    total_eps = sum(r["episodes"] for r in done)
    per_check = Counter()
    ds_with_check = Counter()
    for r in done:
        for c, n in r["checks"].items():
            per_check[c] += n
            ds_with_check[c] += 1
    summary = {
        "datasets_checked": len(done),
        "datasets_skipped": sum(1 for r in rows if "skipped" in r),
        "datasets_failed": sum(1 for r in rows if "error" in r),
        "episodes": total_eps,
        "episodes_bad": sum(r["bad"] for r in done),
        "episodes_review": sum(r["review"] for r in done),
        "datasets_with_any_bad": sum(1 for r in done if r["bad"]),
        "checks": {c: {"episodes": per_check[c], "datasets": ds_with_check[c]} for c, _ in per_check.most_common()},
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
