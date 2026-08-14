"""Surface, for one task, everything the spec audit needs to judge each feature.

The audit question is whether a strong model could derive every hidden assertion from
`feature.md` plus the repo. Three things repeatedly decide it, so this prints them side by side:

    symbols   public names the reference introduces AND the test uses AND the spec never mentions
    messages  error text the test matches on, with whether the reference is where it comes from
              (a message absent from feature.patch is usually pre-existing or raised by the
              test's own stub, and is then not a requirement at all)
    raises    exception types the test expects that the spec does not name

    python scripts/spec_audit_survey.py pallets_jinja_task/task1465

Everything here is a *candidate*, never a verdict — the judgement is manual, and roughly half of
what this flags turns out to be derivable on inspection.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "dataset"

SYMBOL_PATTERNS = (
    r"^\+\s*(?:async )?def\s+(\w+)",
    r"^\+\s*class\s+(\w+)",
    r"^\+func\s+(?:\([^)]*\)\s*)?([A-Z]\w*)",
    r"^\+type\s+([A-Z]\w*)",
    r"^\+\s*([A-Z][A-Za-z0-9]*)\s*=\s",
    r"^\+\s*(?:pub\s+)?fn\s+([a-z_]\w*)",
    r"^\+\s*(?:export\s+)?(?:const|function)\s+(\w+)",
)
MESSAGE_PATTERNS = (
    r'match=["\']([^"\']{8,})["\']',
    r"// Error: [\d\-]+ (.{8,})",
    r'toThrow\(["\']([^"\']{8,})["\']\)',
)
RAISE_PATTERNS = (
    r"(?:raises|assertRaises)\(\s*([A-Z]\w*)",
    r"toThrow\(\s*([A-Z]\w*)\b",
)


def added(patch: Path) -> str:
    return "\n".join(ln for ln in patch.read_text(errors="replace").splitlines() if ln.startswith("+"))


def main() -> None:
    task = DATASET / sys.argv[1]
    if not task.is_dir():
        raise SystemExit(f"no such task: {task}")

    for fd in sorted(task.glob("feature*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        md, tp, fp = fd / "feature.md", fd / "tests.patch", fd / "feature.patch"
        if not (md.is_file() and tp.is_file() and fp.is_file()):
            continue
        spec, test, gold = md.read_text(errors="replace"), added(tp), fp.read_text(errors="replace")
        title = next((ln for ln in spec.splitlines() if ln.strip()), "")[:70]

        symbols = set()
        for pat in SYMBOL_PATTERNS:
            symbols |= set(re.findall(pat, gold, re.M))
        symbols = {s for s in symbols if len(s) > 3 and not s.startswith("_")}
        missing_sym = sorted(
            s for s in symbols
            if re.search(rf"\b{re.escape(s)}\b", test) and not re.search(rf"\b{re.escape(s)}\b", spec)
        )

        msgs = set()
        for pat in MESSAGE_PATTERNS:
            msgs |= set(re.findall(pat, test))
        msg_rows = [
            (m, "GOLD" if m.split("{")[0].strip()[:25] in gold else "elsewhere")
            for m in sorted(msgs) if m.strip(" .")[:35] not in spec
        ]

        raises = set()
        for pat in RAISE_PATTERNS:
            raises |= set(re.findall(pat, test))
        missing_raise = sorted(r for r in raises if not re.search(rf"\b{r}\b", spec))

        print(f"\n--- {fd.name}: {title}")
        if missing_sym:
            print(f"    symbols  : {', '.join(missing_sym[:8])}")
        for m, origin in msg_rows[:4]:
            print(f"    message  : [{origin}] {m[:80]}")
        if missing_raise:
            print(f"    raises   : {', '.join(missing_raise)}")
        if not (missing_sym or msg_rows or missing_raise):
            print("    (nothing flagged)")


if __name__ == "__main__":
    main()
