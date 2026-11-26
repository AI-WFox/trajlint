"""Self-contained HTML report (no external files, works offline)."""

from __future__ import annotations

import html
import json

import numpy as np

from .loader import Dataset

MAX_PLOTS = 40

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3df;--card:#fff;
--ok:#2f7d4f;--review:#a86a00;--bad:#b3261e;--accent:#3557c5;--mark:rgba(179,38,30,.13)}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--fg:#ecebe7;--muted:#a3a29c;--line:#2e2e2b;
--card:#1d1d1b;--ok:#5cc489;--review:#e0a73a;--bad:#f2766d;--accent:#8ea6ff;--mark:rgba(242,118,109,.18)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 10px}
.muted{color:var(--muted)}code{font:13px ui-monospace,Menlo,monospace}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{display:block;font-size:26px}.ok{color:var(--ok)}.review{color:var(--review)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:13px;color:var(--muted);font-weight:600}
.pill{display:inline-block;padding:0 8px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid currentColor}
.sev-error{color:var(--bad)}.sev-warning{color:var(--review)}.sev-info{color:var(--muted)}
ul.iss{margin:0;padding-left:18px}svg{display:block;width:100%;height:auto}
.plot{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:10px}
.wrap{overflow-x:auto}
"""


def _sparkline(ep, issues, fps, width=900, height=90) -> str:
    a = ep.action if ep.action is not None else ep.state
    if a is None or len(a) < 2:
        return ""
    a = np.asarray(a, dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    lo = np.nanmin(a, axis=0)
    hi = np.nanmax(a, axis=0)
    norm = (a - lo) / np.maximum(hi - lo, 1e-9)
    T = len(a)
    xs = np.linspace(0, width, T)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="joint trajectories">']
    dur = T / fps
    for i in issues:
        if "start_s" in i and "end_s" in i and dur > 0:
            x0 = width * i["start_s"] / dur
            x1 = max(width * i["end_s"] / dur, x0 + 2)
            parts.append(f'<rect x="{x0:.1f}" y="0" width="{x1 - x0:.1f}" height="{height}" fill="var(--mark)"/>')
    step = max(1, T // 400)
    for j in range(norm.shape[1]):
        ys = height - 6 - norm[::step, j] * (height - 12)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[::step], ys) if np.isfinite(y))
        op = 0.35 + 0.6 * (j + 1) / norm.shape[1]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-opacity="{op:.2f}" stroke-width="1.3"/>')
    parts.append("</svg>")
    return "".join(parts)


def to_html(result: dict, ds: Dataset | None = None) -> str:
    d = result["dataset"]
    s = result["summary"]
    e = html.escape
    rows_checks = "".join(
        f"<tr><td><code>{e(c)}</code></td><td>{v['episodes']}</td><td>{e(v['help'])}</td></tr>"
        for c, v in s["checks"].items()
    ) or '<tr><td colspan="3">No problems found.</td></tr>'

    eps = sorted(result["episodes"], key=lambda x: (x["score"], x["episode"]))
    flagged = [x for x in eps if x["issues"]]
    rows_eps = []
    for x in flagged:
        items = "".join(
            f'<li><span class="sev-{i["severity"]}">{e(i["check"])}</span>'
            f'{" [" + e(i["camera"]) + "]" if i.get("camera") else ""}: {e(i["message"])}'
            f'{" at %.1fs" % i["start_s"] if "start_s" in i else ""}</li>'
            for i in x["issues"]
        )
        rows_eps.append(
            f'<tr><td>{x["episode"]}</td><td><span class="pill {x["grade"]}">{x["score"]}</span></td>'
            f'<td>{x["seconds"]}s</td><td><ul class="iss">{items}</ul></td></tr>'
        )
    rows_eps_html = "".join(rows_eps) or '<tr><td colspan="4">Every episode passed.</td></tr>'

    plots = ""
    if ds is not None and flagged:
        by_idx = {ep.index: ep for ep in ds.episodes}
        blocks = []
        for x in flagged[:MAX_PLOTS]:
            ep = by_idx.get(x["episode"])
            if ep is None:
                continue
            svg = _sparkline(ep, x["issues"], ds.fps)
            if svg:
                blocks.append(
                    f'<div class="plot"><div class="muted">Episode {x["episode"]} · score {x["score"]} · '
                    f'{e(", ".join(sorted({i["check"] for i in x["issues"]})))}</div>{svg}</div>'
                )
        if blocks:
            plots = ("<h2>Trajectories of flagged episodes</h2><p class='muted'>Each line is one joint "
                     "(action, scaled to its own range). Shaded areas mark where a problem was found.</p>"
                     + "".join(blocks))

    keep = result["keep"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>trajlint report: {e(d['id'])}</title><style>{CSS}</style></head><body><main>
<h1>trajlint report</h1>
<div class="muted"><code>{e(d['id'])}</code> · LeRobot {e(d['version'])} · {e(d['robot_type'])} · {d['fps']:g} fps ·
{d['episodes']} episodes · {d['frames']} frames · cameras: {e(', '.join(d['cameras']) or 'none')}
{'' if d['video_checked'] else ' · video not checked'}</div>
<div class="cards">
<div class="card"><span class="muted">Clean</span><b class="ok">{s['ok']}</b></div>
<div class="card"><span class="muted">Needs a look</span><b class="review">{s['review']}</b></div>
<div class="card"><span class="muted">Likely broken</span><b class="bad">{s['bad']}</b></div>
<div class="card"><span class="muted">Keep (not broken)</span><b>{len(keep)}</b></div>
</div>
<h2>What was found</h2>
<div class="wrap"><table><thead><tr><th>Check</th><th>Episodes</th><th>Meaning</th></tr></thead>
<tbody>{rows_checks}</tbody></table></div>
<h2>Flagged episodes</h2>
<div class="wrap"><table><thead><tr><th>Episode</th><th>Score</th><th>Length</th><th>Problems</th></tr></thead>
<tbody>{rows_eps_html}</tbody></table></div>
{plots}
<h2>Episodes to keep</h2>
<p class="muted">Pass this list to <code>LeRobotDataset(repo_id, episodes=[...])</code>.</p>
<pre><code>{e(json.dumps(keep))}</code></pre>
<p class="muted">Generated by trajlint {e(result['tool']['version'])} in {result['seconds_taken']}s.</p>
</main></body></html>"""
