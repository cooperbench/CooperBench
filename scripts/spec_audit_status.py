"""Regenerate the status table in dataset/SPEC_AUDIT.md from the recorded verdicts.

The audit is per-feature and slow, so the verdicts live in a JSON file
(`dataset/.spec_audit_verdicts.json`) rather than being hand-maintained inside the markdown —
otherwise the table drifts from the prose every time a feature is judged.

    python scripts/spec_audit_status.py

Keys are "<repo>/<task>|<feature>". Values are [verdict, note], where verdict is one of:

    OK    spec is sufficient, nothing changed
    SPEC  feature.md amended, because a hidden assertion was not derivable
    TEST  tests.patch amended, because the test itself was the defect
    RUN?  defect identified, but the fix needs a container run to verify gold still passes
"""

from __future__ import annotations

import json
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "dataset"
VERDICTS = DATASET / ".spec_audit_verdicts.json"
AUDIT = DATASET / "SPEC_AUDIT.md"
START, END = "## Status", "## Separately: gradeability"


def main() -> None:
    verdicts = json.loads(VERDICTS.read_text())
    features = sorted(
        (f"{p.parent.parent.name}/{p.parent.name}", p.name)
        for p in DATASET.glob("*/task*/feature*")
        if p.is_dir() and (p / "feature.md").is_file()
    )

    rows, prev = [], None
    for task, feat in features:
        verdict, note = verdicts.get(f"{task}|{feat}", ["-", ""])
        rows.append(f"| {task if task != prev else ''} | {feat.replace('feature', 'f')} "
                    f"| `{verdict:4s}` | {note} |")
        prev = task

    done = sum(1 for t, f in features if f"{t}|{f}" in verdicts)
    counts: dict[str, int] = {}
    for v, _ in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    tally = " · ".join(f"`{k}` {v}" for k, v in sorted(counts.items()))

    table = "\n".join(["| task | feature | verdict | note |", "|---|---|---|---|", *rows])
    header = (
        f"## Status — {done}/{len(features)} features audited\n\n"
        f"{tally}\n\n"
        "`OK` sufficient, no change · `SPEC` feature.md amended · `TEST` tests.patch amended · "
        "`RUN?` defect identified, fix needs a container run · `-` not yet audited\n\n"
    )

    text = AUDIT.read_text()
    AUDIT.write_text(text[: text.index(START)] + header + table + "\n\n" + text[text.index(END):])
    print(f"{done}/{len(features)} audited — {tally}")


if __name__ == "__main__":
    main()
