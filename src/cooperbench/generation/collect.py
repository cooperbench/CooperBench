"""PR collection wrapper -- fetches GitHub PRs via SWE-smith and filters
them for use as CooperBench seed features.

Usage::

    # Collect PRs from a repo, build instances, and filter
    python -m cooperbench.generation.collect \\
        --repo pallets/flask --output logs/pr_collection --max-pulls 50

    # Filter an existing instances JSONL only
    python -m cooperbench.generation.collect \\
        --filter logs/pr_collection/tasks/pallets-flask-insts.jsonl \\
        --min-test-lines 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Fetch raw PRs
# ---------------------------------------------------------------------------

def collect_prs(
    repo: str,
    output_dir: Path,
    max_pulls: int = 200,
    token: str | None = None,
) -> Path:
    """Fetch merged PRs from a GitHub repo and write them to JSONL.

    Wraps ``swesmith.bug_gen.mirror.collect.print_pulls``.

    Returns the path to the raw PRs JSONL file.
    """
    from swesmith.bug_gen.mirror.collect.print_pulls import main as print_pulls

    token = token or os.environ.get("GITHUB_TOKEN", "")
    prs_dir = Path(output_dir) / "prs"
    prs_dir.mkdir(parents=True, exist_ok=True)

    repo_slug = repo.replace("/", "-")
    prs_path = prs_dir / f"{repo_slug}-prs.jsonl"

    if prs_path.exists():
        logger.info("Raw PRs file already exists: %s — skipping collection", prs_path)
        return prs_path

    logger.info("Collecting PRs from %s (max %d)…", repo, max_pulls)
    print_pulls(repo, str(prs_path), token=token, max_pulls=max_pulls)
    logger.info("Saved raw PRs to %s", prs_path)
    return prs_path


# ---------------------------------------------------------------------------
# Stage 2: Build structured instances from raw PRs
# ---------------------------------------------------------------------------

def build_instances(
    prs_path: Path,
    output_path: Path,
    token: str | None = None,
    sleep_interval: int = 2,
) -> Path:
    """Convert raw PR JSONL into SWE-bench-style instances.

    Re-implements the core loop of
    ``swesmith.bug_gen.mirror.collect.build_dataset`` but with a configurable
    *sleep_interval* (the upstream hard-codes ``time.sleep(60)``).

    Returns the path to the instances JSONL file.
    """
    from swesmith.bug_gen.mirror.collect.utils import (
        Repo,
        extract_patches,
        extract_problem_statement_and_hints,
    )

    token = token or os.environ.get("GITHUB_TOKEN", "")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-processed instances
    seen: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        logger.info("Resuming — %d instances already processed", len(seen))

    prs_path = Path(prs_path)
    with open(prs_path) as pf:
        pulls = [json.loads(line) for line in pf]

    # Infer owner/name from the first PR
    first_pr = pulls[0]
    repo_full = first_pr.get("base", {}).get("repo", {}).get("full_name", "")
    if not repo_full:
        repo_full = "/".join(prs_path.stem.replace("-prs", "").split("-", 1))
    owner, name = repo_full.split("/", 1) if "/" in repo_full else (repo_full, "")

    repo = Repo(owner, name, token=token)

    write_mode = "a" if seen else "w"
    total = 0
    written = 0

    with open(output_path, write_mode) as out:
        for ix, pull in enumerate(pulls):
            if pull.get("merged_at") is None:
                continue

            instance_id = (
                f"{owner}__{name}-{pull['number']}"
            )
            if instance_id in seen:
                continue

            total += 1
            try:
                patch, test_patch = extract_patches(pull, repo)
                problem_statement, hints = extract_problem_statement_and_hints(pull, repo)
            except Exception as exc:
                logger.warning("Skipping PR #%s: %s", pull.get("number"), exc)
                time.sleep(sleep_interval)
                continue

            if not patch:
                time.sleep(sleep_interval)
                continue

            instance = {
                "repo": f"{owner}/{name}",
                "pull_number": pull["number"],
                "instance_id": instance_id,
                "issue_numbers": pull.get("resolved_issues", []),
                "base_commit": pull["base"]["sha"],
                "patch": patch,
                "test_patch": test_patch,
                "problem_statement": problem_statement,
                "hints_text": hints,
                "created_at": pull.get("created_at", ""),
            }
            print(json.dumps(instance), file=out, flush=True)
            written += 1

            if ix % 20 == 0:
                logger.info(
                    "Progress: %d/%d PRs processed, %d instances written",
                    ix + 1, len(pulls), written,
                )

            time.sleep(sleep_interval)

    logger.info(
        "Done — %d merged PRs processed, %d instances written to %s",
        total, written, output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# Stage 3: Filter instances for CooperBench quality
# ---------------------------------------------------------------------------

_NON_TEST_CODE_EXTENSIONS = {
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".md",
    ".rst", ".txt", ".lock", ".xml", ".html", ".css",
}


def _has_runnable_test_files(test_patch: str) -> bool:
    """Return True if the test_patch touches at least one runnable test source
    file (e.g. ``.py``), as opposed to only CI configs or docs."""
    import re as _re

    for m in _re.finditer(r"^\+\+\+ b/(.+)$", test_patch, _re.MULTILINE):
        path = m.group(1)
        if not path or path.startswith("/dev/null"):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in _NON_TEST_CODE_EXTENSIONS:
            continue
        if path.startswith(".github/"):
            continue
        return True
    return False


def filter_instances(
    instances_path: Path,
    min_test_lines: int = 5,
    max_patch_lines: int = 500,
    max_patch_files: int = 10,
    require_description: bool = False,
) -> list[dict]:
    """Read an instances JSONL and return only those suitable as seed features.

    Filters applied:
    - Must have a non-empty ``test_patch``
    - ``test_patch`` must contain at least one runnable test source file
    - ``test_patch`` must have >= *min_test_lines* lines
    - ``patch`` must have <= *max_patch_lines* lines
    - ``patch`` must touch <= *max_patch_files* files
    - Optionally must have a non-empty ``problem_statement``
    """
    instances_path = Path(instances_path)
    candidates: list[dict] = []

    with open(instances_path) as f:
        for line in f:
            inst = json.loads(line)
            test_patch = inst.get("test_patch", "").strip()
            patch = inst.get("patch", "").strip()

            if not test_patch:
                continue
            if not _has_runnable_test_files(test_patch):
                continue
            if len(test_patch.splitlines()) < min_test_lines:
                continue
            if len(patch.splitlines()) > max_patch_lines:
                continue

            # Count distinct files in patch (diff --git a/... b/...)
            file_count = sum(
                1 for l in patch.splitlines() if l.startswith("diff --git")
            )
            if file_count > max_patch_files:
                continue

            if require_description and not inst.get("problem_statement", "").strip():
                continue

            candidates.append(inst)

    logger.info(
        "Filtered %s → %d candidates (min_test=%d, max_patch=%d, max_files=%d%s)",
        instances_path.name,
        len(candidates),
        min_test_lines,
        max_patch_lines,
        max_patch_files,
        ", require_desc" if require_description else "",
    )
    return candidates


# ---------------------------------------------------------------------------
# Convenience: run the full pipeline
# ---------------------------------------------------------------------------

def collect_and_filter(
    repo: str,
    output_dir: str | Path,
    max_pulls: int = 200,
    sleep_interval: int = 2,
    **filter_kwargs,
) -> list[dict]:
    """Collect PRs, build instances, and filter — all in one call."""
    output_dir = Path(output_dir)
    repo_slug = repo.replace("/", "-")

    prs_path = collect_prs(repo, output_dir, max_pulls=max_pulls)
    insts_path = output_dir / "tasks" / f"{repo_slug}-insts.jsonl"
    build_instances(prs_path, insts_path, sleep_interval=sleep_interval)
    return filter_instances(insts_path, **filter_kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Collect GitHub PRs and filter for CooperBench seed features",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--repo",
        help="GitHub owner/name to collect PRs from (e.g. pallets/flask)",
    )
    group.add_argument(
        "--filter",
        dest="filter_path",
        help="Path to an existing instances JSONL to filter only",
    )

    parser.add_argument("--output", default="logs/pr_collection", help="Output directory (default: logs/pr_collection)")
    parser.add_argument("--max-pulls", type=int, default=200, help="Max PRs to fetch (default: 200)")
    parser.add_argument("--sleep", type=int, default=2, help="Seconds between GitHub API calls (default: 2)")
    parser.add_argument("--min-test-lines", type=int, default=5, help="Min lines in test_patch (default: 5)")
    parser.add_argument("--max-patch-lines", type=int, default=500, help="Max lines in patch (default: 500)")
    parser.add_argument("--max-patch-files", type=int, default=10, help="Max files touched by patch (default: 10)")
    parser.add_argument("--require-description", action="store_true", help="Only keep PRs with a problem_statement")

    args = parser.parse_args()

    filter_kwargs = dict(
        min_test_lines=args.min_test_lines,
        max_patch_lines=args.max_patch_lines,
        max_patch_files=args.max_patch_files,
        require_description=args.require_description,
    )

    if args.filter_path:
        candidates = filter_instances(Path(args.filter_path), **filter_kwargs)
    else:
        candidates = collect_and_filter(
            args.repo,
            args.output,
            max_pulls=args.max_pulls,
            sleep_interval=args.sleep,
            **filter_kwargs,
        )

    print(f"\n{'=' * 60}")
    print(f"Candidates: {len(candidates)}")
    for c in candidates:
        has_desc = bool(c.get("problem_statement", "").strip())
        test_lines = len(c.get("test_patch", "").splitlines())
        patch_lines = len(c.get("patch", "").splitlines())
        print(
            f"  PR #{c['pull_number']:5d}  "
            f"patch={patch_lines:3d}L  test={test_lines:3d}L  "
            f"desc={'yes' if has_desc else 'no ':3s}"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
