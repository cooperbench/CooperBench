"""End-to-end pipeline for CooperBench data generation.

Runs the full PR-to-features pipeline in stages, saving outputs to a
timestamped run directory.  Each stage can be run independently; when
starting from stage N, outputs from stages 1..(N-1) are forked from a
previous run.

Stages:
  1. collect   – Fetch merged PRs from GitHub and build SWE-bench instances
  2. filter    – Apply seed-quality filters to instances
  3. onboard   – Convert PRs to task dirs, optionally build/validate Docker images
  4. controller – Run the LLM controller to expand/decompose features per task

Usage::

    # Full pipeline
    python -m cooperbench.generation.pipeline \\
        --repo pallets/flask --repo-name flask_task \\
        --repo-url https://github.com/pallets/flask.git

    # Single stage, forking from a previous run
    python -m cooperbench.generation.pipeline \\
        --repo pallets/flask --repo-name flask_task \\
        --repo-url https://github.com/pallets/flask.git \\
        --stage controller --from-run logs/pipeline_runs/flask_task_20260409_134500
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

STAGES = ["collect", "filter", "onboard", "controller"]


def _run_dir(repo_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "pipeline_runs" / f"{repo_name}_{ts}"


def _save_config(run_dir: Path, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline_config.json").write_text(
        json.dumps(config, indent=2, default=str)
    )


def _fork_stages(from_dir: Path, to_dir: Path, up_to_stage: str) -> None:
    """Copy stage outputs from a previous run up to (not including) the given stage."""
    stage_idx = STAGES.index(up_to_stage)
    for i in range(stage_idx):
        stage_name = STAGES[i]
        src = from_dir / f"stage{i + 1}_{stage_name}"
        dst = to_dir / f"stage{i + 1}_{stage_name}"
        if src.exists():
            logger.info("Forking stage %d (%s) from %s", i + 1, stage_name, from_dir)
            shutil.copytree(src, dst)
        else:
            logger.warning("Stage %d (%s) not found in %s", i + 1, stage_name, from_dir)


# ---------------------------------------------------------------------------
# Stage 1: Collect
# ---------------------------------------------------------------------------

def stage_collect(run_dir: Path, config: dict) -> dict:
    """Collect PRs from GitHub and build instances."""
    from cooperbench.generation.collect import build_instances, collect_prs

    stage_dir = run_dir / "stage1_collect"
    stage_dir.mkdir(parents=True, exist_ok=True)

    repo = config["repo"]
    max_pulls = config.get("max_pulls", 200)
    sleep_interval = config.get("sleep_interval", 2)

    logger.info("=== Stage 1: COLLECT PRs from %s (max=%d) ===", repo, max_pulls)

    prs_path = collect_prs(repo, stage_dir, max_pulls=max_pulls)
    prs_copy = stage_dir / "prs.jsonl"
    if prs_path != prs_copy and prs_path.exists():
        shutil.copy2(prs_path, prs_copy)

    repo_slug = repo.replace("/", "-")
    insts_path = stage_dir / f"{repo_slug}-insts.jsonl"
    build_instances(prs_path, insts_path, sleep_interval=sleep_interval)

    n_prs = sum(1 for _ in open(prs_path)) if prs_path.exists() else 0
    n_insts = sum(1 for _ in open(insts_path)) if insts_path.exists() else 0

    result = {
        "prs_path": str(prs_path),
        "instances_path": str(insts_path),
        "total_prs": n_prs,
        "total_instances": n_insts,
    }
    (stage_dir / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("Collect done: %d PRs → %d instances", n_prs, n_insts)
    return result


# ---------------------------------------------------------------------------
# Stage 2: Filter
# ---------------------------------------------------------------------------

def stage_filter(run_dir: Path, config: dict) -> dict:
    """Filter instances for seed quality."""
    from cooperbench.generation.collect import filter_instances

    stage_dir = run_dir / "stage2_filter"
    stage_dir.mkdir(parents=True, exist_ok=True)

    collect_dir = run_dir / "stage1_collect"
    repo_slug = config["repo"].replace("/", "-")
    insts_path = collect_dir / f"{repo_slug}-insts.jsonl"
    if not insts_path.exists():
        insts_files = list(collect_dir.glob("*-insts.jsonl"))
        if insts_files:
            insts_path = insts_files[0]
        else:
            raise FileNotFoundError(f"No instances JSONL in {collect_dir}")

    logger.info("=== Stage 2: FILTER instances from %s ===", insts_path)

    candidates = filter_instances(
        insts_path,
        min_test_lines=config.get("min_test_lines", 5),
        max_patch_lines=config.get("max_patch_lines", 500),
        max_patch_files=config.get("max_patch_files", 10),
    )

    # Save candidates
    cand_path = stage_dir / "candidates.jsonl"
    with open(cand_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, default=str) + "\n")

    # If we have image constraints, note which task IDs have images
    available_images = config.get("available_images", [])
    if available_images:
        matched = [c for c in candidates
                   if c.get("pull_number") in available_images]
        unmatched = [c for c in candidates
                     if c.get("pull_number") not in available_images]
        matched_path = stage_dir / "candidates_with_images.jsonl"
        with open(matched_path, "w") as f:
            for c in matched:
                f.write(json.dumps(c, default=str) + "\n")
        logger.info(
            "Filter: %d candidates total, %d have Docker images, %d do not",
            len(candidates), len(matched), len(unmatched),
        )
    else:
        matched = candidates

    result = {
        "instances_path": str(insts_path),
        "total_candidates": len(candidates),
        "candidates_with_images": len(matched),
        "candidate_pull_numbers": [c.get("pull_number") for c in candidates],
        "matched_pull_numbers": [c.get("pull_number") for c in matched],
    }
    (stage_dir / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("Filter done: %d candidates (%d with images)",
                len(candidates), len(matched))
    return result


# ---------------------------------------------------------------------------
# Stage 3: Onboard
# ---------------------------------------------------------------------------

def _check_image_exists(image_name: str) -> bool:
    """Return True if a Docker image exists locally."""
    import subprocess
    r = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True, timeout=15,
    )
    return r.returncode == 0


def stage_onboard(run_dir: Path, config: dict) -> dict:
    """Convert filtered PRs to task directories and optionally validate.

    When ``use_swesmith`` is set in *config*, Tier 2 image conversion is
    used instead of building from a Dockerfile, and the separate Docker
    build step is skipped.

    Processing order: candidates with existing Docker images first, then
    the rest (building images as needed).  Stops after *max_tasks* validated
    tasks if the option is set.
    """
    from cooperbench.generation.onboard import (
        build_task_image,
        convert_pr_to_task,
        validate_task,
    )
    from cooperbench.utils import get_image_name

    stage_dir = run_dir / "stage3_onboard"
    stage_dir.mkdir(parents=True, exist_ok=True)

    filter_dir = run_dir / "stage2_filter"
    use_swesmith = config.get("use_swesmith", False)
    skip_build = config.get("skip_build", False) or use_swesmith
    skip_validate = config.get("skip_validate", False)
    max_tasks = config.get("max_tasks", 0)  # 0 = unlimited
    repo_name = config["repo_name"]
    repo_url = config["repo_url"]
    model = config.get("model", "gemini/gemini-3-flash-preview")
    dataset_dir = Path(config.get("dataset_dir", "dataset"))

    if use_swesmith:
        from cooperbench.utils import get_base_image_name
        base_img = get_base_image_name(repo_name)
        if not _check_image_exists(base_img):
            raise RuntimeError(
                f"SWE-smith base image not found: {base_img}\n"
                f"Run 'python -m cooperbench.generation.convert_swesmith base' first."
            )

    cand_path = filter_dir / "candidates.jsonl"
    if not cand_path.exists():
        raise FileNotFoundError(f"No candidates JSONL in {filter_dir}")

    candidates = []
    with open(cand_path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    # Sort: candidates with existing images first
    def _has_image(c: dict) -> bool:
        img = get_image_name(repo_name, c.get("pull_number", 0))
        return _check_image_exists(img)

    with_img = [c for c in candidates if _has_image(c)]
    without_img = [c for c in candidates if c not in with_img]
    ordered = with_img + without_img

    logger.info(
        "=== Stage 3: ONBOARD %d candidates as %s "
        "(%d with existing images, max_tasks=%s) ===",
        len(candidates), repo_name, len(with_img),
        max_tasks or "unlimited",
    )

    repo_dir = dataset_dir / repo_name
    if config.get("clear_dataset") and repo_dir.exists():
        logger.info("Clearing existing dataset dir: %s", repo_dir)
        shutil.rmtree(repo_dir)

    results = []
    n_validated = 0

    for i, inst in enumerate(ordered):
        if max_tasks and n_validated >= max_tasks:
            logger.info("Reached max_tasks=%d, stopping onboarding", max_tasks)
            break

        pull_num = inst.get("pull_number", 0)
        has_existing_image = inst in with_img
        logger.info(
            "--- Onboarding PR #%d (%d/%d) [image=%s] ---",
            pull_num, i + 1, len(ordered),
            "exists" if has_existing_image else "needs build",
        )

        try:
            task_dir = convert_pr_to_task(
                inst, repo_name, repo_url,
                dataset_dir=dataset_dir,
                model_name=model,
                install_cmd=config.get("install_cmd",
                    'pip install -e ".[test]" || pip install -e . && pip install pytest'),
                reinstall_cmd=config.get("reinstall_cmd",
                    'pip install -e ".[test]" || pip install -e .\npip install pytest pytest-xdist pytest_mock'),
                base_image=config.get("base_image", "python:3.11-slim"),
                system_deps=config.get("system_deps", "git curl build-essential"),
                use_swesmith=use_swesmith,
            )
        except Exception as e:
            logger.error("Convert failed for PR #%d: %s", pull_num, e)
            results.append({"pull_number": pull_num, "status": "convert_failed",
                            "error": str(e)})
            continue

        entry = {"pull_number": pull_num, "task_dir": str(task_dir),
                 "had_existing_image": has_existing_image}

        need_build = not skip_build and not has_existing_image and not use_swesmith
        if need_build:
            ok, err = build_task_image(repo_name, pull_num, dataset_dir)
            if not ok:
                logger.error("Build failed for PR #%d: %s", pull_num, err)
                entry["status"] = "build_failed"
                entry["error"] = err
                results.append(entry)
                continue

        if not skip_validate:
            val = validate_task(repo_name, pull_num, dataset_dir,
                                auto_build=False)
            entry["validation"] = val
            if val.get("passed"):
                entry["status"] = "validated"
                n_validated += 1
                logger.info("PR #%d: VALIDATED (%d/%s)",
                            pull_num, n_validated, max_tasks or "∞")
            else:
                entry["status"] = "validation_failed"
                logger.warning("PR #%d: validation failed: %s",
                               pull_num, val.get("error", val.get("output", "")[:200]))
        else:
            entry["status"] = "onboarded"
            n_validated += 1

        results.append(entry)

    validated = [r for r in results if r["status"] in ("validated", "onboarded")]
    failed = [r for r in results if r["status"] not in ("validated", "onboarded")]

    result = {
        "total_candidates": len(candidates),
        "processed": len(results),
        "validated": len(validated),
        "failed": len(failed),
        "tasks": results,
        "validated_task_ids": [r["pull_number"] for r in validated],
    }
    (stage_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    logger.info("Onboard done: %d validated, %d failed out of %d processed",
                len(validated), len(failed), len(results))
    return result


# ---------------------------------------------------------------------------
# Stage 4: Controller
# ---------------------------------------------------------------------------

def _run_single_task(
    tid: int,
    repo_name: str,
    dataset_dir: Path,
    ctrl_config,
    stage_dir: Path,
) -> dict:
    """Run controller on a single task (designed for parallel execution)."""
    from cooperbench.generation.controller import run_controller

    task_dir = dataset_dir / repo_name / f"task{tid}"
    if not task_dir.exists():
        logger.warning("Task dir not found: %s", task_dir)
        return {"task_id": tid, "status": "missing"}

    logger.info("--- Controller for task%d STARTING ---", tid)

    try:
        state = run_controller(task_dir, ctrl_config)
        hard = sum(1 for e in state.entanglement
                   if e.has_conflict and e.is_solvable
                   and e.resolution_strategy not in (None, "auto"))
        trivial = sum(1 for e in state.entanglement
                     if e.has_conflict and e.is_solvable
                     and e.resolution_strategy in (None, "auto"))
        entry = {
            "task_id": tid,
            "status": "completed",
            "features": len(state.features),
            "hard_datapoints": hard,
            "trivial_pairs": trivial,
            "total_cost": state.total_cost,
            "actions": len(state.history),
        }
    except Exception as e:
        logger.error("Controller failed for task%d: %s", tid, e, exc_info=True)
        entry = {"task_id": tid, "status": "failed", "error": str(e)}

    (stage_dir / f"task{tid}_result.json").write_text(
        json.dumps(entry, indent=2, default=str)
    )
    logger.info("--- Controller for task%d DONE: %s ---", tid, entry.get("status"))
    return entry


def stage_controller(run_dir: Path, config: dict) -> dict:
    """Run the controller agent on each validated task.

    Tasks are processed in parallel up to *concurrency* workers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from cooperbench.generation.controller import ControllerConfig

    stage_dir = run_dir / "stage4_controller"
    stage_dir.mkdir(parents=True, exist_ok=True)

    onboard_dir = run_dir / "stage3_onboard"
    onboard_result_path = onboard_dir / "result.json"
    if not onboard_result_path.exists():
        raise FileNotFoundError(f"No onboard result in {onboard_dir}")

    onboard_result = json.loads(onboard_result_path.read_text())
    task_ids = onboard_result.get("validated_task_ids", [])

    repo_name = config["repo_name"]
    dataset_dir = Path(config.get("dataset_dir", "dataset"))
    model = config.get("model", "gemini/gemini-3-flash-preview")
    concurrency = config.get("concurrency", 1)

    logger.info("=== Stage 4: CONTROLLER on %d tasks in %s (concurrency=%d) ===",
                len(task_ids), repo_name, concurrency)

    ctrl_config = ControllerConfig(
        model_name=model,
        target_features=config.get("target_features", 8),
        max_features=config.get("max_features", 12),
        max_cost=config.get("max_cost", 3.0),
        max_consecutive_failures=config.get("max_failures", 3),
        expand_cost_limit=config.get("expand_cost_limit", 0.50),
        decompose_cost_limit=config.get("decompose_cost_limit", 0.50),
        resolve_cost_limit=config.get("resolve_cost_limit", 0.50),
        timeout=config.get("timeout", 600),
        backend=config.get("backend", "docker"),
    )

    results = []
    if concurrency <= 1:
        for i, tid in enumerate(task_ids):
            logger.info("--- Task %d/%d ---", i + 1, len(task_ids))
            entry = _run_single_task(tid, repo_name, dataset_dir, ctrl_config, stage_dir)
            results.append(entry)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    _run_single_task, tid, repo_name, dataset_dir, ctrl_config, stage_dir,
                ): tid
                for tid in task_ids
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    entry = future.result()
                except Exception as e:
                    logger.error("Controller thread for task%d crashed: %s", tid, e)
                    entry = {"task_id": tid, "status": "failed", "error": str(e)}
                results.append(entry)

    total_hard = sum(r.get("hard_datapoints", 0) for r in results)
    total_features = sum(r.get("features", 0) for r in results)
    total_cost = sum(r.get("total_cost", 0) for r in results)

    result = {
        "tasks": results,
        "summary": {
            "total_tasks": len(task_ids),
            "completed": sum(1 for r in results if r["status"] == "completed"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "total_features": total_features,
            "total_hard_datapoints": total_hard,
            "total_cost": total_cost,
        },
    }
    (stage_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    logger.info(
        "Controller done: %d tasks, %d features, %d hard datapoints, $%.2f",
        len(task_ids), total_features, total_hard, total_cost,
    )
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

STAGE_FUNCS = {
    "collect": stage_collect,
    "filter": stage_filter,
    "onboard": stage_onboard,
    "controller": stage_controller,
}


def run_pipeline(
    config: dict,
    stages: list[str] | None = None,
    from_run: str | None = None,
) -> Path:
    """Run the pipeline (or a subset of stages).

    Returns the run directory.
    """
    stages = stages or STAGES
    repo_name = config["repo_name"]

    run_dir = _run_dir(repo_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(run_dir, config)

    # Fork previous stages if starting mid-pipeline
    if from_run:
        first_stage = stages[0]
        _fork_stages(Path(from_run), run_dir, first_stage)

    logger.info("Pipeline run dir: %s", run_dir)
    logger.info("Stages to run: %s", stages)

    for stage_name in stages:
        logger.info("=" * 60)
        logger.info("Starting stage: %s", stage_name)
        logger.info("=" * 60)
        t0 = time.time()

        result = STAGE_FUNCS[stage_name](run_dir, config)

        elapsed = time.time() - t0
        logger.info("Stage %s completed in %.1fs", stage_name, elapsed)

    logger.info("Pipeline complete. Run dir: %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end CooperBench data generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--repo", required=True,
                        help="GitHub owner/repo (e.g. pallets/flask)")
    parser.add_argument("--repo-name", required=True,
                        help="CooperBench dataset name (e.g. flask_task)")
    parser.add_argument("--repo-url", required=True,
                        help="Git clone URL")

    parser.add_argument("--stage", choices=STAGES,
                        help="Run only this stage (default: all stages)")
    parser.add_argument("--from-run",
                        help="Fork earlier stages from this run directory")

    parser.add_argument("--max-pulls", type=int, default=200)
    parser.add_argument("--model", default="gemini/gemini-3-flash-preview")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Docker image building (use existing images)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip Docker validation of onboarded tasks")
    parser.add_argument("--clear-dataset", action="store_true",
                        help="Clear existing dataset dir before onboarding")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Stop onboarding after this many validated tasks (0=unlimited)")
    parser.add_argument("--available-images", type=int, nargs="*",
                        help="Task IDs (PR numbers) with existing Docker images (deprecated, auto-detected)")

    parser.add_argument("--target-features", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--max-cost", type=float, default=3.0)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of tasks to run in parallel in controller stage")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--backend", default="docker")

    # SWE-smith integration
    parser.add_argument("--use-swesmith", action="store_true",
                        help="Use SWE-smith converted base images (Tier 2) instead of building from Dockerfile")
    parser.add_argument("--swesmith-image",
                        help="SWE-smith source image name (for 'convert base' reference only, "
                             "not used at pipeline runtime)")

    parser.add_argument("--install-cmd")
    parser.add_argument("--reinstall-cmd")
    parser.add_argument("--base-image", default="python:3.11-slim")
    parser.add_argument("--system-deps", default="git curl build-essential")

    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("cooperbench", "__main__"):
        logging.getLogger(name).setLevel(level)
    for name in ("cooperbench.generation",):
        logging.getLogger(name).setLevel(level)
    if args.verbose:
        for name in ("cooperbench.generation.expand.agent",
                     "cooperbench.generation.resolve.agent",
                     "cooperbench.generation.decompose.agent",
                     "agent", "minisweagent"):
            logging.getLogger(name).setLevel(level)
    for name in ("LiteLLM", "litellm", "urllib3", "docker", "httpcore",
                 "httpx", "openai", "google", "grpc"):
        logging.getLogger(name).setLevel(logging.WARNING)

    config = {
        "repo": args.repo,
        "repo_name": args.repo_name,
        "repo_url": args.repo_url,
        "max_pulls": args.max_pulls,
        "model": args.model,
        "skip_build": args.skip_build,
        "skip_validate": args.skip_validate,
        "max_tasks": args.max_tasks,
        "clear_dataset": args.clear_dataset,
        "available_images": args.available_images or [],
        "use_swesmith": args.use_swesmith,
        "swesmith_image": args.swesmith_image,
        "target_features": args.target_features,
        "max_features": args.max_features,
        "max_cost": args.max_cost,
        "max_failures": args.max_failures,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "backend": args.backend,
        "base_image": args.base_image,
        "system_deps": args.system_deps,
    }
    if args.install_cmd:
        config["install_cmd"] = args.install_cmd
    if args.reinstall_cmd:
        config["reinstall_cmd"] = args.reinstall_cmd

    if args.stage:
        stages = [args.stage]
    else:
        stages = STAGES

    run_dir = run_pipeline(config, stages=stages, from_run=args.from_run)

    # Print final summary
    print(f"\nRun directory: {run_dir}")
    for i, stage_name in enumerate(STAGES):
        result_path = run_dir / f"stage{i + 1}_{stage_name}" / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            print(f"  Stage {i + 1} ({stage_name}): {json.dumps({k: v for k, v in result.items() if k != 'tasks'}, default=str)}")


if __name__ == "__main__":
    main()
