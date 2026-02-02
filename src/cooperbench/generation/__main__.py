"""Direct execution entry point for generation module.

Usage:
    python -m cooperbench.generation --task dspy_task/task8394 --model gpt-4o
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate new features for CooperBench tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate a single feature
    python -m cooperbench.generation --task dspy_task/task8394

    # Generate multiple attempts
    python -m cooperbench.generation --task dspy_task/task8394 --attempts 5 --output ./generated

    # Just build and print the prompt (no agent run)
    python -m cooperbench.generation --task dspy_task/task8394 --prompt-only

    # Validate an existing patch
    python -m cooperbench.generation --task dspy_task/task8394 --validate feature.patch tests.patch
""",
    )

    parser.add_argument(
        "--task",
        required=True,
        help="Task path relative to dataset/ (e.g., dspy_task/task8394)",
    )
    parser.add_argument(
        "--model",
        default="gemini/gemini-3-flash-preview",
        help="LLM model to use (default: gemini/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--backend",
        choices=["modal", "docker"],
        default="modal",
        help="Execution backend (default: modal)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of generation attempts (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for generated features",
    )
    parser.add_argument(
        "--feature",
        type=int,
        help="Target a specific feature ID for conflicts (default: first feature)",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Just print the prompt without running the agent",
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List all feature IDs in the task and exit",
    )
    parser.add_argument(
        "--validate",
        nargs=2,
        metavar=("FEATURE_PATCH", "TESTS_PATCH"),
        help="Validate existing patches instead of generating",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=75,
        help="Maximum agent steps (default: 75)",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=2.0,
        help="Maximum cost in USD (default: 2.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save full agent trajectory for debugging",
    )

    args = parser.parse_args()

    # Resolve task directory
    task_dir = Path("dataset") / args.task
    if not task_dir.exists():
        logger.error(f"Task directory not found: {task_dir}")
        sys.exit(1)

    # Parse repo_name and task_id from path
    parts = args.task.split("/")
    if len(parts) != 2:
        logger.error(f"Invalid task path format. Expected: repo_name/taskID (e.g., dspy_task/task8394)")
        sys.exit(1)

    repo_name = parts[0]
    task_id = int(parts[1].replace("task", ""))

    # Mode 0: List features
    if args.list_features:
        from cooperbench.generation.prompt import list_features
        features = list_features(task_dir)
        print(f"Features in {args.task}: {features}")
        return

    # Mode 1: Prompt only
    if args.prompt_only:
        from cooperbench.generation.prompt import build_prompt
        prompt = build_prompt(task_dir, feature_id=args.feature)
        print(prompt)
        return

    # Mode 2: Validate existing patches
    if args.validate:
        feature_patch_path, tests_patch_path = args.validate
        feature_patch = Path(feature_patch_path).read_text()
        tests_patch = Path(tests_patch_path).read_text()

        from cooperbench.generation.validator import validate_generated_feature

        logger.info(f"Validating patches for {args.task}")
        result = validate_generated_feature(
            repo_name=repo_name,
            task_id=task_id,
            feature_patch=feature_patch,
            tests_patch=tests_patch,
            backend=args.backend,
        )

        print(json.dumps(result, indent=2, default=str))
        sys.exit(0 if result["valid"] else 1)

    # Mode 3: Generate features
    from cooperbench.generation.generator import generate_feature, generate_features_batch
    import hashlib
    import re as re_module

    def make_output_dir(base_dir: Path, feature_md: str | None, fallback_hash: str) -> Path:
        """Create output directory named after the feature title + short hash."""
        # Extract title from feature_md if available
        title_slug = "unknown"
        if feature_md:
            # Look for **Title**: ... pattern
            match = re_module.search(r"\*\*Title\*\*:\s*(.+?)(?:\n|$)", feature_md)
            if match:
                title = match.group(1).strip()
                # Convert to slug: lowercase, replace spaces with underscores, remove special chars
                title_slug = re_module.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]

        # Add short hash for uniqueness
        content_hash = hashlib.md5((feature_md or fallback_hash).encode()).hexdigest()[:5]
        folder_name = f"{title_slug}_{content_hash}"

        output_dir = base_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Base directory for this task
    base_output_dir = args.output or (Path("generated") / repo_name / f"task{task_id}")

    if args.attempts == 1:
        logger.info(f"Generating feature for {args.task} with {args.model} (target: feature {args.feature or 'first'})")
        logger.info(f"Limits: {args.step_limit} steps, ${args.cost_limit} cost")

        # Generate first (to temp location for trajectory)
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())

        result = generate_feature(
            task_dir=task_dir,
            feature_id=args.feature,
            model_name=args.model,
            backend=args.backend,
            step_limit=args.step_limit,
            cost_limit=args.cost_limit,
            debug=args.debug,
            output_dir=temp_dir,
        )

        # Create final output dir based on feature title
        from datetime import datetime
        fallback = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = make_output_dir(base_output_dir, result.feature_md, fallback)
        logger.info(f"Output directory: {output_dir}")

        # Move trajectory files from temp to final
        import shutil
        for f in temp_dir.glob("trajectory_*"):
            shutil.move(str(f), output_dir / f.name)

        # Save outputs (ensure patches end with newline for git compatibility)
        if result.feature_patch:
            patch = result.feature_patch.rstrip() + "\n"
            (output_dir / "feature.patch").write_text(patch)
            logger.info(f"Saved feature.patch")
        if result.tests_patch:
            patch = result.tests_patch.rstrip() + "\n"
            (output_dir / "tests.patch").write_text(patch)
            logger.info(f"Saved tests.patch")
        if result.feature_md:
            (output_dir / "feature.md").write_text(result.feature_md)
            logger.info(f"Saved feature.md")

        # Save full result as JSON
        (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
        logger.info(f"Saved result.json")

        # Print summary
        print(f"\n{'='*60}")
        print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Output saved to: {output_dir}")
        if result.errors:
            print(f"Errors: {result.errors}")
        print(f"Agent: {result.agent_steps} steps, ${result.agent_cost:.4f}")
        print(f"{'='*60}")

        sys.exit(0 if result.success else 1)
    else:
        logger.info(f"Running {args.attempts} generation attempts for {args.task} (target: feature {args.feature or 'first'})")
        logger.info(f"Output directory: {base_output_dir}")

        results = generate_features_batch(
            task_dir=task_dir,
            feature_id=args.feature,
            num_attempts=args.attempts,
            model_name=args.model,
            backend=args.backend,
            output_dir=base_output_dir,
            step_limit=args.step_limit,
            cost_limit=args.cost_limit,
            debug=args.debug,
        )

        # Summary
        successful = sum(1 for r in results if r.success)
        print(f"\n{'='*60}")
        print(f"Summary: {successful}/{args.attempts} successful")
        print(f"Output saved to: {base_output_dir}")

        for i, r in enumerate(results, 1):
            status = "✓" if r.success else "✗"
            print(f"  {status} Attempt {i}: {r.errors or 'OK'}")

        print(f"{'='*60}")
        sys.exit(0 if successful > 0 else 1)


if __name__ == "__main__":
    main()
