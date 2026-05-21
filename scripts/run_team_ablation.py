"""Drive a full team-harness ablation sweep over the `core` subset.

For each of the 2**5 = 32 feature combinations, run `cooperbench run -s core
--setting team` with the matching `--team-no-*` flags, evaluate inline,
and append a one-line summary to a CSV.

Run encoding: `ablate-<bits>` where bits = t s m a p (task_list, scratchpad,
mcp, auto_refresh, protocol).  1 = on, 0 = off.  E.g.:
  - ablate-11111  (baseline, all on)
  - ablate-01111  (task_list off, rest on)
  - ablate-00000  (only the lead/member role split)

Logs are written under logs/<run-name>/ as usual.  This driver is
gitignored (lives under scripts/) and is not part of PR #58 itself.
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = REPO_ROOT / "logs"
RESULTS_CSV = REPO_ROOT / "ablation_matrix_flash.csv"

FEATURES = ["task_list", "scratchpad", "mcp", "auto_refresh", "protocol"]
SUBSET = "flash"  # 50 task pairs vs core's 10
RUN_NAME_PREFIX = "ablate-flash-"


def name_for(bits: tuple[int, ...]) -> str:
    return RUN_NAME_PREFIX + "".join(str(b) for b in bits)


def flags_for(bits: tuple[int, ...]) -> list[str]:
    """Return the --team-no-* flags for the features that are OFF."""
    out = []
    for bit, feature in zip(bits, FEATURES):
        if bit == 0:
            out.append(f"--team-no-{feature.replace('_', '-')}")
    return out


def read_summary(run_name: str) -> dict:
    """Return the summary.json for ``run_name`` or {} if missing."""
    p = LOGS_ROOT / run_name / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def run_one(bits: tuple[int, ...], *, model: str, concurrency: int) -> dict:
    name = name_for(bits)
    flags = flags_for(bits)
    cmd = [
        "uv",
        "run",
        "cooperbench",
        "run",
        "-n",
        name,
        "-s",
        SUBSET,
        "-a",
        "codex",
        "-m",
        model,
        "--setting",
        "team",
        "--backend",
        "docker",
        "--concurrency",
        str(concurrency),
        "--force",
        *flags,
    ]
    started = time.time()
    print(f"\n=== {name} ({sum(bits)}/5 features on) ===", flush=True)
    print("  cmd:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    dur = time.time() - started
    if proc.returncode != 0:
        print(f"  [exit {proc.returncode}] stderr tail:", proc.stderr[-500:], flush=True)

    summary = read_summary(name)
    eval_stats = summary.get("eval") or {}
    return {
        "name": name,
        **{f"{f}_on": bits[i] for i, f in enumerate(FEATURES)},
        "n_features_on": sum(bits),
        "completed": summary.get("completed"),
        "failed": summary.get("failed"),
        "skipped": summary.get("skipped"),
        "total_cost": summary.get("total_cost"),
        "wall_seconds": round(dur, 1),
        "eval_passed": eval_stats.get("passed"),
        "eval_failed": eval_stats.get("failed"),
        "eval_errors": eval_stats.get("errors"),
        "eval_total": eval_stats.get("total_evaluated"),
        "pass_rate": eval_stats.get("pass_rate"),
        "exit_code": proc.returncode,
    }


def main() -> int:
    # Default codex model fallback rejects whatever we pass and falls
    # back to its own default — explicitly pass an OpenAI model so the
    # adapter's --model flag doesn't waste the round-trip.
    model = "gpt-5.5"
    concurrency = 5  # 10 tasks at concurrency=5 → 10 docker containers in flight

    # Marginal-effect design: baseline (all 5 on) + 5 one-feature-off
    # configs (4 on, 1 off).  Answers "what is each feature's marginal
    # contribution to pass rate" for ~1/5 the cost of the full 2^5
    # interaction matrix.  Interaction effects (e.g. "task_list and
    # scratchpad together") are not estimable from this design.
    all_combos = list(itertools.product([0, 1], repeat=len(FEATURES)))
    bit_combos = [b for b in all_combos if sum(b) in (5, 4)]
    bit_combos.sort(key=lambda b: -sum(b))  # baseline first

    rows = []
    csv_path = RESULTS_CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                *[f"{x}_on" for x in FEATURES],
                "n_features_on",
                "completed",
                "failed",
                "skipped",
                "total_cost",
                "wall_seconds",
                "eval_passed",
                "eval_failed",
                "eval_errors",
                "eval_total",
                "pass_rate",
                "exit_code",
            ],
        )
        writer.writeheader()
        for i, bits in enumerate(bit_combos, start=1):
            row = run_one(bits, model=model, concurrency=concurrency)
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(
                f"  -> {row['name']}: pass={row.get('eval_passed')}/{row.get('eval_total')} "
                f"cost=${row.get('total_cost') or 0:.2f} wall={row.get('wall_seconds')}s "
                f"[{i}/{len(bit_combos)}]",
                flush=True,
            )

    print(f"\nWrote {len(rows)} rows to {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
