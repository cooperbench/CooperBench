"""Main generator - orchestrates feature generation using agents."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from cooperbench.generation.prompt import build_prompt, list_features
from cooperbench.generation.splitter import extract_feature_description, split_patch
from cooperbench.generation.validator import (
    check_conflicts_in_sandbox,
    run_tests_in_sandbox,
)
from cooperbench.utils import get_image_name

logger = logging.getLogger(__name__)


def _extract_feature_md_from_patch(patch: str) -> str | None:
    """Extract .feature_description.md content from a patch."""
    if not patch or ".feature_description.md" not in patch:
        return None

    lines = patch.split("\n")
    in_feature_file = False
    content_lines = []

    for line in lines:
        if line.startswith("diff --git") and ".feature_description.md" in line:
            in_feature_file = True
            content_lines = []
        elif in_feature_file and line.startswith("diff --git"):
            # End of the feature file
            break
        elif in_feature_file and line.startswith("+") and not line.startswith("+++"):
            # Added line (strip the leading +)
            content_lines.append(line[1:])

    if content_lines:
        return "\n".join(content_lines).strip()
    return None


def _remove_feature_md_from_patch(patch: str) -> str:
    """Remove .feature_description.md from a patch (it's metadata, not code)."""
    if not patch or ".feature_description.md" not in patch:
        return patch

    lines = patch.split("\n")
    result_lines = []
    skip_file = False

    for line in lines:
        if line.startswith("diff --git") and ".feature_description.md" in line:
            skip_file = True
        elif line.startswith("diff --git"):
            skip_file = False

        if not skip_file:
            result_lines.append(line)

    return "\n".join(result_lines)


@dataclass
class GenerationResult:
    """Result of a feature generation attempt."""

    success: bool
    feature_md: str | None = None
    feature_patch: str | None = None
    tests_patch: str | None = None
    conflicts: list[int] = field(default_factory=list)
    conflicts_info: list[dict] = field(default_factory=list)  # [{id, title, conflict_diff}, ...]
    errors: list[str] = field(default_factory=list)
    agent_cost: float = 0.0
    agent_steps: int = 0
    duration_seconds: float = 0.0
    # Validation details
    tests_passed: bool | None = None
    tests_output: str | None = None
    validation_run: bool = False

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "feature_md": self.feature_md,
            "feature_patch": self.feature_patch,
            "tests_patch": self.tests_patch,
            "conflicts": self.conflicts,
            "conflicts_info": self.conflicts_info,
            "errors": self.errors,
            "agent_cost": self.agent_cost,
            "agent_steps": self.agent_steps,
            "duration_seconds": self.duration_seconds,
            "tests_passed": self.tests_passed,
            "tests_output": self.tests_output,
            "validation_run": self.validation_run,
        }


def _get_task_image(task_dir: Path) -> str:
    """Get the Docker image for a task using existing naming convention."""
    task_id = int(task_dir.name.replace("task", ""))
    repo_name = task_dir.parent.name
    return get_image_name(repo_name, task_id)


def _get_repo_and_task_id(task_dir: Path) -> tuple[str, int]:
    """Extract repo_name and task_id from task directory."""
    task_id = int(task_dir.name.replace("task", ""))
    repo_name = task_dir.parent.name
    return repo_name, task_id


def generate_feature(
    task_dir: str | Path,
    feature_id: int | None = None,
    model_name: str = "gemini/gemini-3-flash-preview",
    backend: str = "modal",
    timeout: int = 3600,
    validate: bool = True,
    step_limit: int = 75,
    cost_limit: float = 2.0,
    debug: bool = False,
    output_dir: Path | None = None,
) -> GenerationResult:
    """Generate a new feature for a task using an agent.

    Args:
        task_dir: Path to the task directory (e.g., dataset/dspy_task/task8394)
        feature_id: ID of the specific feature to target for conflicts (default: first)
        model_name: LLM model to use for the agent
        backend: Execution backend ("modal", "docker", or "gcp")
        timeout: Maximum time for generation in seconds
        validate: Whether to validate (run tests + check conflicts) after generation
        step_limit: Maximum number of agent steps (default: 75)
        cost_limit: Maximum cost in USD (default: 2.0)
        debug: Save full agent trajectory to file for inspection
        output_dir: Directory to save debug output (default: current dir)

    Returns:
        GenerationResult with the generated feature or errors.
    """
    task_dir = Path(task_dir)
    start_time = time.time()

    if not task_dir.exists():
        return GenerationResult(
            success=False,
            errors=[f"Task directory not found: {task_dir}"],
        )

    repo_name, task_id = _get_repo_and_task_id(task_dir)

    # Build the prompt
    logger.info(f"Building prompt for {task_dir} (target feature: {feature_id or 'first'})")
    prompt = build_prompt(task_dir, feature_id=feature_id)

    # Get the Docker image for this task
    image = _get_task_image(task_dir)
    logger.info(f"Using image: {image}")

    # Get existing feature IDs for conflict checking
    existing_feature_ids = list_features(task_dir)
    logger.info(f"Found {len(existing_feature_ids)} existing features: {existing_feature_ids}")

    # Run the agent
    logger.info(f"Running agent with model {model_name} on {backend}")

    try:
        from cooperbench.agents import get_runner

        agent = get_runner("mini_swe_agent")

        result = agent.run(
            task=prompt,
            image=image,
            model_name=model_name,
            config={
                "backend": backend,
                "agent": {
                    "step_limit": step_limit,
                    "cost_limit": cost_limit,
                },
            },
        )

        agent_cost = result.cost
        agent_steps = result.steps

        # Save/log agent trajectory for debugging
        if result.messages:
            logger.info(f"Agent trajectory: {len(result.messages)} messages, {agent_steps} steps, ${agent_cost:.4f}")

            # Save full trajectory to file if debug mode or output_dir specified
            if debug or output_dir:
                import re as re_module
                save_dir = output_dir or Path(".")
                save_dir.mkdir(parents=True, exist_ok=True)

                traj_file = save_dir / f"trajectory_{repo_name}_{task_id}.json"
                traj_data = {
                    "task": f"{repo_name}/task{task_id}",
                    "model": model_name,
                    "steps": agent_steps,
                    "cost": agent_cost,
                    "status": result.status,
                    "messages": result.messages,
                    "patch": result.patch,
                }
                with open(traj_file, "w") as f:
                    json.dump(traj_data, f, indent=2, default=str)
                logger.info(f"Saved trajectory to: {traj_file}")

                # Also save a human-readable version
                readable_file = save_dir / f"trajectory_{repo_name}_{task_id}.txt"
                with open(readable_file, "w") as f:
                    f.write(f"=== Agent Trajectory ===\n")
                    f.write(f"Task: {repo_name}/task{task_id}\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Steps: {agent_steps}, Cost: ${agent_cost:.4f}\n")
                    f.write(f"Status: {result.status}\n\n")

                    for i, msg in enumerate(result.messages):
                        role = msg.get("role", "?").upper()
                        content = msg.get("content", "")
                        f.write(f"\n{'='*60}\n")
                        f.write(f"[{i}] {role}\n")
                        f.write(f"{'='*60}\n")
                        f.write(content)
                        f.write("\n")
                logger.info(f"Saved readable trajectory to: {readable_file}")

            # Log summary to console
            import re as re_module
            for i, msg in enumerate(result.messages):
                role = msg.get("role", "?")
                content = msg.get("content", "")[:500]
                if role == "assistant":
                    actions = re_module.findall(r"```bash\s*\n(.*?)\n```", content, re_module.DOTALL)
                    if actions:
                        logger.info(f"  [{i}] AGENT: {actions[0][:200]}")
                elif role == "user" and "returncode" in content:
                    rc_match = re_module.search(r"<returncode>(\d+)</returncode>", content)
                    rc = rc_match.group(1) if rc_match else "?"
                    logger.info(f"  [{i}] RESULT: exit={rc}")

        # Check for agent errors
        if result.status == "Error" or result.error:
            return GenerationResult(
                success=False,
                errors=[f"Agent error: {result.error or result.status}"],
                agent_cost=agent_cost,
                agent_steps=agent_steps,
                duration_seconds=time.time() - start_time,
            )

        # Get the patch from agent
        full_patch = result.patch

        if not full_patch:
            return GenerationResult(
                success=False,
                errors=["Agent produced no changes"],
                agent_cost=agent_cost,
                agent_steps=agent_steps,
                duration_seconds=time.time() - start_time,
            )

        # Extract feature description from .feature_description.md in the patch (before removing it)
        feature_md = _extract_feature_md_from_patch(full_patch)

        # Remove .feature_description.md from patch (it's metadata, not code)
        clean_patch = _remove_feature_md_from_patch(full_patch)

        # Split patch into feature and tests
        logger.info("Splitting patch into feature and tests...")
        feature_patch, tests_patch = split_patch(clean_patch)

        # Fallback: try extracting from agent messages if not in patch
        if not feature_md and result.messages:
            for msg in result.messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        extracted = extract_feature_description(content)
                        if extracted:
                            feature_md = extracted
                            break

        # Basic validation
        errors = []
        if not feature_patch:
            errors.append("No feature changes in patch (only test files modified)")

        if not tests_patch:
            errors.append("No test changes in patch")

        # If basic validation fails, return early
        if errors:
            return GenerationResult(
                success=False,
                feature_md=feature_md,
                feature_patch=feature_patch,
                tests_patch=tests_patch,
                errors=errors,
                agent_cost=agent_cost,
                agent_steps=agent_steps,
                duration_seconds=time.time() - start_time,
            )

        # Run full validation if requested
        tests_passed = None
        tests_output = None
        conflicts = []
        conflicts_info = []
        validation_run = False

        if validate:
            logger.info("Running validation...")
            validation_run = True

            # Step 1: Run tests
            logger.info("Step 1/2: Running tests in sandbox...")
            test_result = run_tests_in_sandbox(
                repo_name=repo_name,
                task_id=task_id,
                feature_patch=feature_patch,
                tests_patch=tests_patch,
                timeout=600,
                backend=backend,
            )

            tests_passed = test_result["passed"]
            tests_output = test_result.get("output", "")

            if test_result.get("error"):
                errors.append(f"Test error: {test_result['error']}")

            if not tests_passed:
                errors.append(f"Tests failed: {test_result['tests_failed']} failed, {test_result['tests_passed']} passed")
                logger.warning(f"Tests failed: {test_result['tests_failed']} failed")
            else:
                logger.info(f"Tests passed: {test_result['tests_passed']} passed")

            # Step 2: Check conflicts (only if tests pass)
            if tests_passed:
                logger.info("Step 2/2: Checking conflicts with existing features...")
                conflict_result = check_conflicts_in_sandbox(
                    repo_name=repo_name,
                    task_id=task_id,
                    new_feature_patch=feature_patch,
                    existing_feature_ids=existing_feature_ids,
                    timeout=300,
                    backend=backend,
                )

                conflicts = conflict_result["conflicts"]
                conflicts_info = conflict_result.get("conflicts_info", [])

                if conflict_result.get("errors"):
                    for err in conflict_result["errors"]:
                        errors.append(f"Conflict check: {err}")

                if not conflicts:
                    errors.append("No conflicts with any existing feature - feature may be too independent")
                    logger.warning("No conflicts detected with existing features")
                else:
                    conflict_titles = [c.get("title", f"Feature {c['id']}") for c in conflicts_info]
                    logger.info(f"Conflicts detected with features: {conflict_titles}")

        # Determine success
        success = (
            len(errors) == 0
            and feature_patch
            and tests_patch
            and (not validate or (tests_passed and len(conflicts) > 0))
        )

        return GenerationResult(
            success=success,
            feature_md=feature_md,
            feature_patch=feature_patch,
            tests_patch=tests_patch,
            conflicts=conflicts,
            conflicts_info=conflicts_info,
            errors=errors,
            agent_cost=agent_cost,
            agent_steps=agent_steps,
            duration_seconds=time.time() - start_time,
            tests_passed=tests_passed,
            tests_output=tests_output,
            validation_run=validation_run,
        )

    except Exception as e:
        logger.exception("Generation failed")
        return GenerationResult(
            success=False,
            errors=[f"Generation failed: {e!s}"],
            duration_seconds=time.time() - start_time,
        )


def generate_features_batch(
    task_dir: str | Path,
    feature_id: int | None = None,
    num_attempts: int = 5,
    model_name: str = "gemini/gemini-3-flash-preview",
    backend: str = "modal",
    output_dir: str | Path | None = None,
    validate: bool = True,
    step_limit: int = 75,
    cost_limit: float = 2.0,
    debug: bool = False,
) -> list[GenerationResult]:
    """Generate multiple feature candidates for a task.

    Args:
        task_dir: Path to the task directory
        feature_id: Target feature ID (default: first)
        num_attempts: Number of generation attempts
        model_name: LLM model to use
        backend: Execution backend
        output_dir: Directory to save results (optional)
        validate: Whether to run validation after each generation
        debug: Save full trajectory for each attempt

    Returns:
        List of GenerationResults (including failures).
    """
    import hashlib
    import re as re_module
    import tempfile
    import shutil

    def make_feature_dir(base_dir: Path, feature_md: str | None, attempt_num: int) -> Path:
        """Create output directory named after the feature title + short hash."""
        title_slug = "unknown"
        if feature_md:
            match = re_module.search(r"\*\*Title\*\*:\s*(.+?)(?:\n|$)", feature_md)
            if match:
                title = match.group(1).strip()
                title_slug = re_module.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]

        content_hash = hashlib.md5((feature_md or f"attempt_{attempt_num}").encode()).hexdigest()[:5]
        folder_name = f"{title_slug}_{content_hash}"

        feature_dir = base_dir / folder_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        return feature_dir

    task_dir = Path(task_dir)
    base_output_dir = Path(output_dir) if output_dir else None
    results = []

    for i in range(num_attempts):
        logger.info(f"=== Generation attempt {i + 1}/{num_attempts} ===")

        # Use temp dir for trajectory during generation
        temp_dir = Path(tempfile.mkdtemp()) if base_output_dir else None

        result = generate_feature(
            task_dir=task_dir,
            feature_id=feature_id,
            model_name=model_name,
            backend=backend,
            validate=validate,
            step_limit=step_limit,
            cost_limit=cost_limit,
            debug=debug,
            output_dir=temp_dir,
        )

        results.append(result)

        # Save to named directory based on feature title
        if base_output_dir:
            attempt_dir = make_feature_dir(base_output_dir, result.feature_md, i + 1)

            # Move trajectory files from temp
            if temp_dir:
                for f in temp_dir.glob("trajectory_*"):
                    shutil.move(str(f), attempt_dir / f.name)

            # Save result JSON
            with open(attempt_dir / "result.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Save patches if available (ensure trailing newline for git compatibility)
            if result.feature_patch:
                patch = result.feature_patch.rstrip() + "\n"
                (attempt_dir / "feature.patch").write_text(patch)
            if result.tests_patch:
                patch = result.tests_patch.rstrip() + "\n"
                (attempt_dir / "tests.patch").write_text(patch)
            if result.feature_md:
                (attempt_dir / "feature.md").write_text(result.feature_md)

            logger.info(f"Saved attempt {i + 1} to {attempt_dir}")

        # Log result
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        logger.info(f"Attempt {i + 1} {status}: {result.errors or 'OK'}")

    # Summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"=== Generation complete: {successful}/{num_attempts} successful ===")

    return results
