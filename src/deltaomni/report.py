from __future__ import annotations

import argparse
import html
import json
import os
import uuid
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else html.escape(str(value))


def render_report(
    summary: dict[str, Any],
    verification: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    initial = summary["initial_validation_losses"]
    final = summary["final_validation_losses"]
    metrics = verification["metrics"]
    checks = verification["checks"]
    loss_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{_metric(initial[name])}</td>"
        f"<td>{_metric(final[name])}</td></tr>"
        for name in ("total", "reconstruction", "trigger", "caption", "length")
    )
    metric_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{_metric(value)}</td></tr>"
        for name, value in metrics.items()
        if not isinstance(value, dict)
    )
    check_items = "".join(
        f"<li class='{('pass' if passed else 'fail')}'>{html.escape(name)}: "
        f"{('PASS' if passed else 'FAIL')}</li>"
        for name, passed in checks.items()
    )
    approved = ", ".join(provenance["approved"])
    blocked = ", ".join(provenance["blocked"])
    interleaving = html.escape(summary["interleaving"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DeltaOmni sanity report</title><style>
:root{{--bg:#0b0f14;--panel:#121923;--line:#273445;--text:#e7edf5;--muted:#91a0b4;
--blue:#68a7ff;--green:#52d18c;--red:#ff7b7b;--amber:#f4c15d}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:44px 24px 80px}}h1{{font-size:38px;margin:0 0 8px}}
h2{{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:8px}}p{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}
.card strong{{display:block;color:var(--blue);font-size:24px}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}
code,pre{{background:#070a0e;border:1px solid var(--line);border-radius:9px}}pre{{padding:15px;
overflow:auto;white-space:pre-wrap}}.pass{{color:var(--green)}}.fail{{color:var(--red)}}
.warn{{border-left:4px solid var(--amber)}}small{{color:var(--muted)}}
</style></head><body><main>
<small>Independent embedding-level functional verification</small><h1>DeltaOmni sanity report</h1>
<p>Run <code>{html.escape(summary['run_id'])}</code>.
Synthetic fixture only; no real-media claim.</p>
<div class="grid"><div class="card"><span>Verification</span>
<strong>{'PASS' if verification['passed'] else 'FAIL'}</strong></div>
<div class="card"><span>Caption exact</span><strong>{metrics['caption_exact']:.1%}</strong></div>
<div class="card"><span>Trigger F1</span><strong>{metrics['trigger']['f1']:.1%}</strong></div>
<div class="card"><span>Full + caption QA</span>
<strong>{metrics['full_plus_caption_qa_accuracy']:.1%}</strong></div></div>
<h2>Architecture</h2><pre>FULL anchor A_k + [delta(previous, current)]* → bounded slots
→ reconstruct current → commit → scoped caption → reset slots → refreshed FULL anchor</pre>
<h2>Loss reduction</h2><table><thead><tr><th>Loss</th><th>Initial</th><th>Final</th></tr></thead>
<tbody>{loss_rows}</tbody></table><h2>Causal and downstream metrics</h2>
<table><tbody>{metric_rows}</tbody></table><h2>Checks</h2><ul>{check_items}</ul>
<h2>Interleaving trace</h2><pre>{interleaving}</pre><h2>Final QA contract</h2>
<div class="card warn"><p>{html.escape(verification['qa_contract']['question'])}</p>
<p>The relational answer never appears in a caption. Full-only QA:
{metrics['final_full_only_qa_accuracy']:.1%}; caption-history QA:
{metrics['caption_history_qa_accuracy']:.1%}; combined:
{metrics['full_plus_caption_qa_accuracy']:.1%}.</p></div><h2>Provenance gate</h2>
<p><b>Approved:</b> {html.escape(approved)}</p><p><b>Blocked:</b> {html.escape(blocked)}</p>
</main></body></html>"""


def generate(run_root: Path, provenance_path: Path, output_path: Path) -> Path:
    runs = sorted(
        path
        for path in run_root.glob("delta-sanity-*")
        if (path / "verification.json").is_file()
    )
    if not runs:
        raise FileNotFoundError("No verified sanity run is available")
    run = runs[-1]
    content = render_report(
        _load(run / "summary.json"),
        _load(run / "verification.json"),
        _load(provenance_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".html.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a dark-mode DeltaOmni sanity report")
    parser.add_argument("--run-root", type=Path, default=Path("outputs/sanity"))
    parser.add_argument("--provenance", type=Path, default=Path("outputs/reports/provenance.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/sanity_report.html"))
    args = parser.parse_args()
    print(generate(args.run_root, args.provenance, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
