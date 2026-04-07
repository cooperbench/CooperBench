"""Deterministic verification of candidate features.

Runs three sequential checks against existing features in a task:

1. **Tests pass** -- the candidate's own tests pass when its patch is applied.
2. **Conflicts exist** -- the candidate's patch creates git merge conflicts
   with at least one existing feature.
3. **Joint solvability** -- for each conflicting pair, the gold patches can
   be merged (naive then union fallback) and *both* test suites still pass.

Usage::

    # Full verification (all 3 checks)
    python -m cooperbench.generation.verify \\
        --task pallets_click_task/task2068 \\
        --candidate-feature 1 \\
        --against 2 \\
        --backend docker

    # Individual checks
    python -m cooperbench.generation.verify \\
        --task pallets_click_task/task2068 \\
        --candidate-feature 1 \\
        --against 2 \\
        --check tests --check conflicts \\
        --backend docker

    # From raw patch files instead of feature ID
    python -m cooperbench.generation.verify \\
        --task pallets_click_task/task2068 \\
        --feature-patch /path/to/feature.patch \\
        --tests-patch /path/to/tests.patch \\
        --against 2 3 4 \\
        --backend docker

Output is structured JSON to stdout so an agent or script can parse it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cooperbench.eval.backends import get_backend
from cooperbench.eval.sandbox import (
    _filter_test_files,
    _parse_results,
    _write_patch,
    merge_and_test,
)
from cooperbench.generation.validator import (
    _get_existing_feature_ids,
    check_conflicts_in_sandbox,
    run_tests_in_sandbox,
)
from cooperbench.utils import get_image_name

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of running verification checks on a candidate feature."""

    tests_ok: bool = False
    tests_output: str = ""
    tests_passed: int = 0
    tests_failed: int = 0

    conflicts: list[int] = field(default_factory=list)
    clean: list[int] = field(default_factory=list)

    solvable_pairs: list[int] = field(default_factory=list)
    solvability_details: dict[int, dict] = field(default_factory=dict)

    overall_ok: bool = False
    failure_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Check 1: Tests Pass
# ---------------------------------------------------------------------------

def check_tests(
    repo_name: str,
    task_id: int,
    feature_patch: str,
    tests_patch: str,
    timeout: int = 600,
    backend: str = "docker",
) -> dict:
    """Verify that a candidate feature's tests pass in a sandbox.

    Delegates to :func:`validator.run_tests_in_sandbox`.

    Returns dict with: passed, tests_passed, tests_failed, output, error.
    """
    return run_tests_in_sandbox(
        repo_name=repo_name,
        task_id=task_id,
        feature_patch=feature_patch,
        tests_patch=tests_patch,
        timeout=timeout,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Check 2: Conflicts Exist
# ---------------------------------------------------------------------------

def check_conflicts(
    repo_name: str,
    task_id: int,
    feature_patch: str,
    existing_feature_ids: list[int] | None = None,
    timeout: int = 300,
    backend: str = "docker",
) -> dict:
    """Check which existing features create merge conflicts with the candidate.

    If *existing_feature_ids* is ``None``, all features found in the task
    directory are checked.

    Delegates to :func:`validator.check_conflicts_in_sandbox`.

    Returns dict with: conflicts, clean, errors, output.
    """
    if existing_feature_ids is None:
        task_dir = Path("dataset") / repo_name / f"task{task_id}"
        existing_feature_ids = _get_existing_feature_ids(task_dir)

    if not existing_feature_ids:
        return {
            "conflicts": [],
            "clean": [],
            "errors": ["No existing features to check against"],
            "output": "",
        }

    return check_conflicts_in_sandbox(
        repo_name=repo_name,
        task_id=task_id,
        new_feature_patch=feature_patch,
        existing_feature_ids=existing_feature_ids,
        timeout=timeout,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Check 3: Joint Solvability
# ---------------------------------------------------------------------------

def check_solvability(
    repo_name: str,
    task_id: int,
    candidate_patch: str,
    candidate_tests: str,
    conflicting_feature_id: int,
    timeout: int = 600,
    backend: str = "docker",
) -> dict:
    """Test whether a candidate + existing feature are jointly solvable.

    Applies the existing feature's gold patch as *patch1* and the candidate
    patch as *patch2*, then uses :func:`sandbox.merge_and_test` to attempt
    naive merge (then union fallback) and runs both test suites.

    Returns dict matching the shape of :func:`sandbox.test_merged` output,
    plus ``solvable: bool``.
    """
    task_dir = Path("dataset") / repo_name / f"task{task_id}"
    existing_dir = task_dir / f"feature{conflicting_feature_id}"

    existing_patch_path = existing_dir / "feature.patch"
    existing_tests_path = existing_dir / "tests.patch"

    if not existing_patch_path.exists():
        return {"solvable": False, "error": f"Patch not found: {existing_patch_path}"}
    if not existing_tests_path.exists():
        return {"solvable": False, "error": f"Tests not found: {existing_tests_path}"}

    existing_patch = existing_patch_path.read_text()
    existing_tests = existing_tests_path.read_text()

    existing_patch = _filter_test_files(existing_patch)
    candidate_patch_filtered = _filter_test_files(candidate_patch)

    image = get_image_name(repo_name, task_id)
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)

    try:
        result = merge_and_test(
            sb,
            patch1_content=existing_patch,
            patch2_content=candidate_patch_filtered,
            tests1_content=existing_tests,
            tests2_content=candidate_tests,
        )
        result["solvable"] = result.get("both_passed", False)
        return result
    except Exception as e:
        return {"solvable": False, "error": str(e)}
    finally:
        sb.terminate()


# ---------------------------------------------------------------------------
# Main entry point: verify_candidate
# ---------------------------------------------------------------------------

def verify_candidate(
    repo_name: str,
    task_id: int,
    feature_patch: str,
    tests_patch: str,
    existing_feature_ids: list[int] | None = None,
    checks: list[str] | None = None,
    timeout: int = 600,
    backend: str = "docker",
) -> VerificationResult:
    """Run all (or selected) verification checks on a candidate feature.

    Parameters
    ----------
    repo_name : str
        Repository / task family name (e.g. ``"pallets_click_task"``).
    task_id : int
        Numeric task identifier.
    feature_patch : str
        The candidate's implementation patch content.
    tests_patch : str
        The candidate's test patch content.
    existing_feature_ids : list[int] | None
        Feature IDs to check against.  ``None`` means auto-discover from the
        task directory.
    checks : list[str] | None
        Subset of ``["tests", "conflicts", "solvability"]`` to run.
        ``None`` runs all three sequentially, short-circuiting on failure.
    backend : str
        Sandbox backend (``"docker"`` or ``"modal"``).

    Returns
    -------
    VerificationResult
        Structured result with per-check details and an ``overall_ok`` flag.
    """
    run_all = checks is None
    run_tests = run_all or "tests" in checks
    run_conflicts = run_all or "conflicts" in checks
    run_solvability = run_all or "solvability" in checks

    result = VerificationResult()

    # -- Check 1: Tests pass ------------------------------------------------
    if run_tests:
        logger.info("Check 1/3: running candidate tests …")
        t = check_tests(
            repo_name, task_id, feature_patch, tests_patch,
            timeout=timeout, backend=backend,
        )
        result.tests_ok = t["passed"]
        result.tests_output = t.get("output", "")
        result.tests_passed = t.get("tests_passed", 0)
        result.tests_failed = t.get("tests_failed", 0)

        if t.get("error"):
            result.error = t["error"]
            result.failure_reason = "tests_error"
            return result

        if not result.tests_ok:
            result.failure_reason = "tests_failed"
            if run_all:
                return result

    # -- Check 2: Conflicts exist -------------------------------------------
    if run_conflicts:
        logger.info("Check 2/3: checking merge conflicts …")
        c = check_conflicts(
            repo_name, task_id, feature_patch,
            existing_feature_ids=existing_feature_ids,
            timeout=timeout, backend=backend,
        )
        result.conflicts = c.get("conflicts", [])
        result.clean = c.get("clean", [])

        if c.get("errors"):
            result.error = "; ".join(c["errors"])

        if not result.conflicts:
            result.failure_reason = "no_conflicts"
            if run_all:
                return result

    # -- Check 3: Joint solvability -----------------------------------------
    if run_solvability:
        pairs_to_check = result.conflicts if result.conflicts else []
        if not pairs_to_check:
            if existing_feature_ids is None:
                task_dir = Path("dataset") / repo_name / f"task{task_id}"
                pairs_to_check = _get_existing_feature_ids(task_dir)
            else:
                pairs_to_check = existing_feature_ids

        if not pairs_to_check:
            result.failure_reason = "no_features_for_solvability"
            return result

        logger.info(
            "Check 3/3: testing joint solvability against features %s …",
            pairs_to_check,
        )
        for fid in pairs_to_check:
            logger.info("  solvability check: candidate vs feature %d", fid)
            s = check_solvability(
                repo_name, task_id,
                candidate_patch=feature_patch,
                candidate_tests=tests_patch,
                conflicting_feature_id=fid,
                timeout=timeout, backend=backend,
            )
            result.solvability_details[fid] = {
                "both_passed": s.get("both_passed", False),
                "solvable": s.get("solvable", False),
                "merge_strategy": (s.get("merge", {}) or {}).get("strategy"),
                "merge_status": (s.get("merge", {}) or {}).get("status"),
                "error": s.get("error"),
            }
            if s.get("solvable"):
                result.solvable_pairs.append(fid)

        if not result.solvable_pairs:
            result.failure_reason = "not_solvable"
            if run_all:
                return result

    # -- Overall verdict ----------------------------------------------------
    if run_all:
        result.overall_ok = (
            result.tests_ok
            and len(result.conflicts) > 0
            and len(result.solvable_pairs) > 0
        )
    else:
        ok_parts = []
        if run_tests:
            ok_parts.append(result.tests_ok)
        if run_conflicts:
            ok_parts.append(len(result.conflicts) > 0)
        if run_solvability:
            ok_parts.append(len(result.solvable_pairs) > 0)
        result.overall_ok = all(ok_parts) if ok_parts else False

    if result.overall_ok:
        result.failure_reason = None

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic verification of a candidate feature.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True,
        help="Task identifier as repo_name/taskN (e.g. pallets_click_task/task2068)",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--candidate-feature", type=int, metavar="FID",
        help="Existing feature ID to use as the candidate (reads patches from dataset)",
    )
    src.add_argument(
        "--feature-patch", type=Path, metavar="PATH",
        help="Path to candidate feature.patch file",
    )

    parser.add_argument(
        "--tests-patch", type=Path, metavar="PATH",
        help="Path to candidate tests.patch file (required with --feature-patch)",
    )

    parser.add_argument(
        "--against", type=int, nargs="+", metavar="FID",
        help="Feature IDs to check against (default: all features in task)",
    )
    parser.add_argument(
        "--check", action="append", dest="checks",
        choices=["tests", "conflicts", "solvability"],
        help="Run only specific check(s). Repeat for multiple. Default: all.",
    )
    parser.add_argument(
        "--backend", default="docker", choices=["docker", "modal"],
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Parse task identifier
    parts = args.task.strip("/").split("/")
    if len(parts) != 2 or not parts[1].startswith("task"):
        parser.error("--task must be repo_name/taskN, e.g. pallets_click_task/task2068")
    repo_name = parts[0]
    task_id = int(parts[1].replace("task", ""))

    # Resolve candidate patches
    if args.candidate_feature is not None:
        fdir = Path("dataset") / repo_name / f"task{task_id}" / f"feature{args.candidate_feature}"
        fp = fdir / "feature.patch"
        tp = fdir / "tests.patch"
        if not fp.exists():
            parser.error(f"Feature patch not found: {fp}")
        if not tp.exists():
            parser.error(f"Tests patch not found: {tp}")
        feature_patch = fp.read_text()
        tests_patch = tp.read_text()

        # When checking a feature against others, exclude itself by default
        against = args.against
        if against is None:
            task_dir = Path("dataset") / repo_name / f"task{task_id}"
            against = [
                fid for fid in _get_existing_feature_ids(task_dir)
                if fid != args.candidate_feature
            ]
    else:
        if args.tests_patch is None:
            parser.error("--tests-patch is required when using --feature-patch")
        if not args.feature_patch.exists():
            parser.error(f"Feature patch not found: {args.feature_patch}")
        if not args.tests_patch.exists():
            parser.error(f"Tests patch not found: {args.tests_patch}")
        feature_patch = args.feature_patch.read_text()
        tests_patch = args.tests_patch.read_text()
        against = args.against

    result = verify_candidate(
        repo_name=repo_name,
        task_id=task_id,
        feature_patch=feature_patch,
        tests_patch=tests_patch,
        existing_feature_ids=against,
        checks=args.checks,
        timeout=args.timeout,
        backend=args.backend,
    )

    # Structured JSON output
    out = result.to_dict()
    # Truncate long test output for readability
    if len(out.get("tests_output", "")) > 2000:
        out["tests_output"] = out["tests_output"][-2000:]
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")

    sys.exit(0 if result.overall_ok else 1)


if __name__ == "__main__":
    main()
