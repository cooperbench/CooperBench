"""Generate a self-contained HTML report of the team-harness ablation +
solo/coop/coop-git comparison experiments.

Reads the run logs under logs/ (summary.json + per-task result.json) and
emits docs/team_harness_ablation_report.html with all numbers embedded
inline (no external assets), so the file is reviewable standalone in a PR.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
OUT = REPO / "docs" / "team_harness_ablation_report.html"

# (run_name, setting_subdir, label, group)
RUNS = [
    ("compare-solo-flash", "solo", "solo (1 agent)", "compare"),
    ("compare-coop-flash", "coop", "coop (messaging only)", "compare"),
    ("compare-coop-git-flash", "coop", "coop + git", "compare"),
    ("ablate-flash-11111", "team", "team — all features (baseline)", "ablation"),
    ("ablate-flash-01111", "team", "team — no task_list", "ablation"),
    ("ablate-flash-10111", "team", "team — no scratchpad", "ablation"),
    ("ablate-flash-11011", "team", "team — no mcp", "ablation"),
    ("ablate-flash-11101", "team", "team — no auto_refresh", "ablation"),
    ("ablate-flash-11110", "team", "team — no protocol", "ablation"),
]

FEATURE_BITS = {  # run -> (task_list, scratchpad, mcp, auto_refresh, protocol)
    "ablate-flash-11111": (1, 1, 1, 1, 1),
    "ablate-flash-01111": (0, 1, 1, 1, 1),
    "ablate-flash-10111": (1, 0, 1, 1, 1),
    "ablate-flash-11011": (1, 1, 0, 1, 1),
    "ablate-flash-11101": (1, 1, 1, 0, 1),
    "ablate-flash-11110": (1, 1, 1, 1, 0),
}
FEATURES = ["task_list", "scratchpad", "mcp", "auto_refresh", "protocol"]


def fmt_dur(s: float | None) -> str:
    if not s:
        return "—"
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s"


def collect(run: str, setting: str) -> dict:
    base = LOGS / run / setting
    durations = []
    if base.exists():
        for rj in base.rglob("result.json"):
            try:
                ds = json.loads(rj.read_text()).get("duration_seconds")
                if ds:
                    durations.append(ds)
            except Exception:
                pass
    passed = total = run_wall = None
    summ = LOGS / run / "summary.json"
    if summ.exists():
        sj = json.loads(summ.read_text())
        e = sj.get("eval") or {}
        passed, total = e.get("passed"), e.get("total_evaluated")
        run_wall = sj.get("total_time_seconds")
    return {
        "n_tasks": len(durations),
        "passed": passed,
        "total": total,
        "run_wall": run_wall,
        "dur_min": min(durations) if durations else None,
        "dur_med": statistics.median(durations) if durations else None,
        "dur_max": max(durations) if durations else None,
    }


def bar(pct: float, color: str) -> str:
    return (
        f'<div class="bar-wrap"><div class="bar" style="width:{pct:.0f}%;background:{color}">'
        f"</div><span>{pct:.0f}%</span></div>"
    )


def main() -> None:
    data = {run: collect(run, setting) for run, setting, _, _ in RUNS}

    base = data["ablate-flash-11111"]
    base_pass = base["passed"] or 0
    base_total = base["total"] or 50

    # ---- comparison rows (sorted by pass rate) ----
    comp_rows = []
    everything = [(run, setting, label) for run, setting, label, _ in RUNS]
    for run, setting, label in everything:
        d = data[run]
        p, t = d["passed"], d["total"]
        rate = (p / t * 100) if (p is not None and t) else None
        comp_rows.append((label, run, p, t, rate, d))
    comp_rows.sort(key=lambda r: (r[4] is None, r[4] or 0))

    rows_html = []
    for label, run, p, t, rate, d in comp_rows:
        is_base = run == "ablate-flash-11111"
        pass_str = f"{p}/{t}" if p is not None else "running"
        rate_cell = bar(rate, "#2563eb" if not is_base else "#16a34a") if rate is not None else "—"
        cls = ' class="baseline"' if is_base else ""
        rows_html.append(
            f"<tr{cls}><td>{label}</td><td class='num'>{pass_str}</td><td>{rate_cell}</td>"
            f"<td class='num'>{fmt_dur(d['dur_med'])}</td><td class='num'>{fmt_dur(d['run_wall'])}</td></tr>"
        )
    comparison_table = "\n".join(rows_html)

    # ---- ablation matrix (one feature off per row) ----
    abl_rows = []
    for run in ["ablate-flash-11111", "ablate-flash-01111", "ablate-flash-10111",
                "ablate-flash-11011", "ablate-flash-11101", "ablate-flash-11110"]:
        d = data[run]
        bits = FEATURE_BITS[run]
        p = d["passed"]
        off = [FEATURES[i] for i, b in enumerate(bits) if b == 0]
        off_label = off[0] if off else "(baseline — none off)"
        delta = (p - base_pass) if p is not None else None
        cells = "".join(
            f"<td class='feat {'on' if b else 'off'}'>{'on' if b else 'OFF'}</td>" for b in bits
        )
        delta_str = "—" if delta is None else (f"+{delta}" if delta > 0 else str(delta))
        delta_cls = "pos" if (delta or 0) > 0 else ("neg" if (delta or 0) < 0 else "zero")
        pass_str = f"{p}/{d['total']}" if p is not None else "running"
        is_base = run == "ablate-flash-11111"
        cls = ' class="baseline"' if is_base else ""
        abl_rows.append(
            f"<tr{cls}><td>{off_label}</td>{cells}<td class='num'>{pass_str}</td>"
            f"<td class='num {delta_cls}'>{delta_str}</td></tr>"
        )
    ablation_table = "\n".join(abl_rows)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Team-Harness Ablation Report</title>
<style>
  :root {{ --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--fg); max-width:980px; margin:0 auto; padding:2rem 1.25rem; line-height:1.55; }}
  h1 {{ font-size:1.7rem; margin-bottom:.2rem; }}
  h2 {{ font-size:1.25rem; margin-top:2.2rem; border-bottom:2px solid var(--line); padding-bottom:.3rem; }}
  .meta {{ color:var(--muted); font-size:.9rem; margin-bottom:1.5rem; }}
  table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:.92rem; }}
  th,td {{ border:1px solid var(--line); padding:.45rem .6rem; text-align:left; }}
  th {{ background:#f9fafb; font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  tr.baseline {{ background:#f0fdf4; font-weight:600; }}
  .feat {{ text-align:center; font-size:.78rem; font-weight:600; }}
  .feat.on {{ color:#16a34a; }}
  .feat.off {{ color:#dc2626; background:#fef2f2; }}
  .pos {{ color:#16a34a; }} .neg {{ color:#dc2626; }} .zero {{ color:var(--muted); }}
  .bar-wrap {{ position:relative; background:#f3f4f6; border-radius:4px; height:20px; min-width:120px; }}
  .bar {{ height:100%; border-radius:4px; }}
  .bar-wrap span {{ position:absolute; left:6px; top:0; font-size:.78rem; line-height:20px; color:#111; }}
  .findings li {{ margin-bottom:.5rem; }}
  .caveat {{ background:#fffbeb; border:1px solid #fcd34d; border-radius:6px; padding:.8rem 1rem; font-size:.9rem; }}
  code {{ background:#f3f4f6; padding:.1rem .35rem; border-radius:3px; font-size:.85em; }}
  .key {{ font-size:.85rem; color:var(--muted); }}
</style>
</head>
<body>

<h1>Team-Harness Ablation &amp; Multi-Agent Comparison</h1>
<div class="meta">
  Generated {generated} · agent <code>codex</code> · model <code>gpt-5.5</code> ·
  subset <code>flash</code> (50 task pairs) · backend <code>docker</code> · 1 seed
</div>

<p>
  Measures (a) how N-agent settings compare — <strong>solo</strong> (1 agent),
  <strong>coop</strong> (messaging only), <strong>coop+git</strong> (shared git remote),
  and <strong>team</strong> (lead/member + shared task list + scratchpad) — and
  (b) the marginal contribution of each of the five team-harness coordination
  features via one-feature-off ablation.
</p>

<h2>1. Setting comparison</h2>
<p>Each row is the same 50 task pairs. A pair "passes" only if <em>both</em> features'
held-out test suites pass against one merged tree (see Methodology).</p>
<table>
  <thead><tr><th>configuration</th><th>passed</th><th>pass rate</th>
    <th>median task time</th><th>run wall</th></tr></thead>
  <tbody>
{comparison_table}
  </tbody>
</table>
<p class="key">Green row = team baseline (all features on). coop/coop+git ran at lower
concurrency alongside the ablation sweep, inflating their "run wall" — the median task
time is the cleaner cross-run comparison.</p>

<h2>2. Feature ablation (one feature off per row)</h2>
<p>All rows are team mode on the same 50 pairs; Δ is the change in passed count vs the
all-on baseline. The lead/member role split stays on in every row — it is the defining
property of team mode, not a toggle.</p>
<table>
  <thead><tr><th>feature removed</th>
    <th>task_list</th><th>scratchpad</th><th>mcp</th><th>auto_refresh</th><th>protocol</th>
    <th>passed</th><th>Δ</th></tr></thead>
  <tbody>
{ablation_table}
  </tbody>
</table>

<h2>3. Key findings</h2>
<ul class="findings">
  <li><strong>Code-sharing is the load-bearing mechanism.</strong> The two features that
    let agents see each other's work — <code>scratchpad</code> (−16) and
    <code>task_list</code> (−11) — account for nearly all of team mode's value. Remove
    either and team drops <em>below solo</em>: two uncoordinated agents are worse than one.</li>
  <li><strong>mcp, auto_refresh, protocol have no positive effect</strong> for codex.
    mcp and auto_refresh land within noise of baseline (−1 each); protocol-off actually
    scored <em>higher</em> (+4), suggesting the typed request/respond verbs add mild
    overhead without payoff here. <code>auto_refresh</code> is expected to be a no-op — it
    only fires in Python-loop adapters, and codex is a CLI adapter.</li>
  <li><strong>Most multi-agent value = a shared code substrate, not the orchestration.</strong>
    coop+git (git remote) reaches 56% and team (scratchpad+task_list) reaches 62% — both
    far above messaging-only coop (26%), which is the worst configuration of all, below solo.</li>
</ul>

<h2>4. Methodology</h2>
<p><strong>Eval protocol</strong> (per task pair <code>repo/task [f_a, f_b]</code>):</p>
<ul>
  <li><strong>solo</strong>: apply the single patch, both feature suites must pass against it.</li>
  <li><strong>coop / team</strong>: apply each agent's patch to its own branch, then:
    (1) if patches are byte-identical, use one; (2) else attempt naive 3-way merge — if
    clean, the merged tree is authoritative and both suites must pass; (3) if the merge
    conflicts or a patch fails to apply, fall back to the lead's patch alone. No union
    merge, no member fallback.</li>
  <li>A pass requires <strong>both</strong> feature suites green against the same tree.
    Eval runs in a fresh container from the task's frozen image, after the agent
    containers are torn down.</li>
</ul>
<p><strong>Step budget</strong>: codex <code>exec</code> ran unbounded (no <code>--max-turns</code>),
capped only by a 2-hour wall-clock timeout; agents self-terminated after ~50–95 tool calls each.</p>

<div class="caveat">
  <strong>Caveats.</strong> Single seed, n=50, codex/gpt-5.5 only. Effective discriminating
  n is smaller than 50 — many pairs pass or fail regardless of coordination. Costs/model
  field show $0 because codex's <code>--json</code> stream omits a cost field (real spend was
  nonzero). Team runs used the <strong>scratchpad</strong> for code-sharing, <em>not</em> a
  git server (<code>--git</code> was off) — so "team vs coop+git" compares two different
  sharing substrates, not "team = coop+git plus extras". The untested cell
  <code>team --git</code> (both substrates) is a follow-up.
</div>

</body>
</html>
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
