"""Feature decomposition pipeline -- splits an existing large feature into
N smaller, independently-testable sub-features that conflict with each other
and are jointly solvable.

Uses a two-phase approach:
  1. A cheap ``litellm.completion()`` call analyses the original patch and
     produces a structured decomposition plan.
  2. Per-sub-feature MSA agents implement each planned sub-feature, iterating
     against the same ``validate_feature.sh`` used by :mod:`expand`.

Usage::

    python -m cooperbench.generation.decompose \\
        --task dspy_task/task8394 \\
        --feature 4 \\
        --model gemini/gemini-3-flash-preview

From Python::

    from cooperbench.generation.decompose import decompose_feature
    result = decompose_feature("dataset/dspy_task/task8394", feature_id=4)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import litellm

from cooperbench.generation.expand import (
    SETUP_SCRIPT,
    VALIDATE_FEATURE_SH,
    ExpansionResult,
    _LoggingAgent,
    _extract_patches_from_container,
)
from cooperbench.generation.onboard import generate_feature_md
from cooperbench.generation.validator import _get_existing_feature_ids
from cooperbench.generation.verify import (
    check_conflicts,
    check_solvability,
    check_tests,
)
from cooperbench.utils import get_image_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SubFeaturePlan:
    """One item from the decomposition plan."""

    title: str = ""
    description: str = ""
    target_files: list[str] = field(default_factory=list)
    overlap_with: list[int] = field(default_factory=list)


@dataclass
class DecompositionPlan:
    """Output of the LLM planning call."""

    decomposable: bool = False
    reason: str = ""
    sub_features: list[SubFeaturePlan] = field(default_factory=list)


@dataclass
class DecompositionResult:
    """Outcome of the full decomposition attempt."""

    success: bool = False
    plan: DecompositionPlan | None = None
    sub_feature_results: list[ExpansionResult] = field(default_factory=list)
    accepted_feature_ids: list[int] = field(default_factory=list)
    planning_cost: float = 0.0
    total_agent_cost: float = 0.0
    failure_reason: str | None = None
    failure_detail: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.plan:
            d["plan"] = asdict(self.plan)
        return d


# ---------------------------------------------------------------------------
# Heuristic gate
# ---------------------------------------------------------------------------

def _should_decompose(feature_patch: str, tests_patch: str) -> tuple[bool, str]:
    """Quick heuristic: is the patch large enough to bother decomposing?

    Returns (should_try, reason).
    """
    diff_headers = [
        line for line in feature_patch.splitlines()
        if line.startswith("diff --git")
    ]
    n_files = len(diff_headers)

    non_test_lines = 0
    skip = False
    for line in feature_patch.splitlines():
        if line.startswith("diff --git"):
            skip = bool(re.search(
                r"(/test_|/tests/|_test\.py|/test/|tests\.py)", line,
            ))
        if not skip:
            non_test_lines += 1

    if n_files >= 3:
        return True, f"Patch touches {n_files} files (>= 3)"
    if non_test_lines >= 50:
        return True, f"Patch has {non_test_lines} non-test lines (>= 50)"
    return False, f"Patch too small ({n_files} files, {non_test_lines} non-test lines)"


# ---------------------------------------------------------------------------
# Phase 1: Decomposition planning (cheap LLM call)
# ---------------------------------------------------------------------------

_PLAN_SYSTEM_PROMPT = """\
You are analysing a software patch to determine whether it can be meaningfully
decomposed into multiple independent sub-features.

Requirements for each sub-feature:
- It must be independently implementable and testable.
- It must modify at least one *source* file (not just docs/config).
- At least two sub-features must share a source file so that merging them
  creates git merge conflicts.
- Each sub-feature should represent a cohesive, independently-useful change.

Constraints:
- Minimum 2 sub-features, maximum 5.
- If the patch is too small, too tightly coupled, or otherwise cannot be
  cleanly decomposed, set "decomposable" to false.

Return ONLY valid JSON matching this schema (no markdown fences):
{
  "decomposable": true/false,
  "reason": "Short explanation of why it can or cannot be decomposed",
  "sub_features": [
    {
      "title": "Short title for the sub-feature",
      "description": "Detailed description of what this sub-feature does and how to implement it. Must be specific enough for a developer to implement it independently.",
      "target_files": ["src/foo.py", "src/bar.py"],
      "overlap_with": [1]
    }
  ]
}

The "overlap_with" field is a list of 0-indexed indices of other sub-features
that share target files with this one (i.e. will create merge conflicts).
"""


def _generate_decomposition_plan(
    feature_patch: str,
    tests_patch: str,
    feature_md: str | None,
    model_name: str = "gemini/gemini-3-flash-preview",
) -> tuple[DecompositionPlan, float]:
    """Call the LLM to produce a decomposition plan.

    Returns (plan, cost_usd).
    """
    patch_lines = feature_patch[:12_000]
    test_lines = tests_patch[:6_000] if tests_patch else "(no test patch)"
    desc = feature_md[:4_000] if feature_md else "(no description)"

    user_msg = f"""\
## Feature Description
{desc}

## Feature Patch
```diff
{patch_lines}
```

## Test Patch
```diff
{test_lines}
```

Analyse this patch and return the JSON decomposition plan."""

    resp = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content.strip()
    cost = float(getattr(resp, "_hidden_params", {}).get("response_cost", 0) or 0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            logger.error("LLM returned non-JSON: %s", raw[:500])
            return DecompositionPlan(reason="LLM returned non-JSON"), cost

    plan = DecompositionPlan(
        decomposable=data.get("decomposable", False),
        reason=data.get("reason", ""),
    )
    for sf in data.get("sub_features", []):
        plan.sub_features.append(SubFeaturePlan(
            title=sf.get("title", ""),
            description=sf.get("description", ""),
            target_files=sf.get("target_files", []),
            overlap_with=sf.get("overlap_with", []),
        ))

    return plan, cost


# ---------------------------------------------------------------------------
# Phase 2 helpers: pre-seeding and prompt building
# ---------------------------------------------------------------------------

def _preseed_decomposition_container(
    env,
    task_dir: Path,
    existing_feature_ids: list[int],
    original_feature_id: int,
    plan_json: str,
) -> None:
    """Copy validation script, setup, original feature, plan, and existing
    feature data into the MSA container before the agent starts.
    """
    docker = env.config.executable
    cid = env.container_id

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Scripts
        (tmp_path / "setup.sh").write_text(SETUP_SCRIPT)
        (tmp_path / "validate_feature.sh").write_text(VALIDATE_FEATURE_SH)

        # Decomposition plan
        (tmp_path / "decomposition_plan.json").write_text(plan_json)

        # Original feature (the one being decomposed)
        orig_dir = tmp_path / "original"
        orig_dir.mkdir()
        orig_feat = task_dir / f"feature{original_feature_id}"
        for fname in ("feature.patch", "tests.patch"):
            src = orig_feat / fname
            if src.exists():
                (orig_dir / fname).write_text(src.read_text())

        # Existing features (original + already-accepted sub-features)
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        for fid in existing_feature_ids:
            feat_src = task_dir / f"feature{fid}"
            feat_dst = existing_dir / f"feature{fid}"
            feat_dst.mkdir()
            for fname in ("feature.patch", "tests.patch"):
                src = feat_src / fname
                if src.exists():
                    (feat_dst / fname).write_text(src.read_text())

        # Feature descriptions
        desc_dir = tmp_path / "feature_descriptions"
        desc_dir.mkdir()
        for fid in existing_feature_ids:
            md_src = task_dir / f"feature{fid}" / "feature.md"
            if md_src.exists():
                (desc_dir / f"feature{fid}.md").write_text(md_src.read_text())

        # Copy everything into the container
        subprocess.run(
            [docker, "cp", f"{tmp_path}/.", f"{cid}:/patches/"],
            check=True, capture_output=True,
        )

    # Make scripts executable
    subprocess.run(
        [docker, "exec", cid, "chmod", "+x",
         "/patches/setup.sh", "/patches/validate_feature.sh"],
        check=True, capture_output=True,
    )


def _build_subfeature_prompt(
    plan: DecompositionPlan,
    sub_feature_index: int,
    task_dir: Path,
    existing_feature_ids: list[int],
    original_feature_id: int,
    hints: str | None = None,
) -> str:
    """Construct the MSA prompt for implementing one specific sub-feature."""
    sf = plan.sub_features[sub_feature_index]

    # Build existing feature descriptions
    feature_sections = []
    for fid in existing_feature_ids:
        md_path = task_dir / f"feature{fid}" / "feature.md"
        if md_path.exists():
            content = md_path.read_text().strip()
        else:
            content = "(no description available)"
        label = f"Feature {fid}"
        if fid == original_feature_id:
            label += " (the original feature being decomposed)"
        feature_sections.append(f"### Existing {label}\n{content}")

    existing_desc = "\n\n".join(feature_sections) if feature_sections else "(none yet)"

    # Format the full decomposition plan for context
    plan_summary = []
    for i, s in enumerate(plan.sub_features):
        marker = " <-- YOUR ASSIGNMENT" if i == sub_feature_index else ""
        plan_summary.append(
            f"  {i + 1}. {s.title}: {s.description[:200]}{marker}"
        )
    plan_text = "\n".join(plan_summary)

    patch_listing = "\n".join(
        f"- `/patches/existing/feature{fid}/feature.patch`"
        for fid in existing_feature_ids
    )

    return f"""You are a software engineer implementing ONE specific sub-feature from a
decomposition plan. A larger feature has been broken down into independent
sub-features, and your job is to implement ONLY the one assigned to you.

## Decomposition Plan

The original feature (feature {original_feature_id}) is being decomposed into
these sub-features:
{plan_text}

## Your Assignment

Implement sub-feature {sub_feature_index + 1}: **{sf.title}**

{sf.description}

Target files: {', '.join(sf.target_files) if sf.target_files else '(see description)'}

## Existing Features

These features already exist in the dataset. Your sub-feature MUST create merge
conflicts with at least one of them (this is almost guaranteed since you are
implementing a subset of the original feature's changes).

{existing_desc}

## Pre-loaded Files

- `/patches/setup.sh` -- run this FIRST to initialise the environment
- `/patches/validate_feature.sh` -- run this to check your work before submitting
- `/patches/original/feature.patch` -- the FULL original feature patch (for reference only)
- `/patches/original/tests.patch` -- the original test patch (for reference only)
- `/patches/decomposition_plan.json` -- the full decomposition plan
- Existing feature patches:
{patch_listing}
- Feature descriptions: `/patches/feature_descriptions/feature*.md`

## Instructions

### Step 1: Setup
```
cd /workspace/repo && bash /patches/setup.sh
```

### Step 2: Understand the context
Read the original feature patch at `/patches/original/feature.patch` and the
decomposition plan. Understand what your assigned sub-feature should do versus
what the OTHER sub-features will do. You must implement ONLY your piece.

Also explore the source code to understand the codebase structure.

### Step 3: Implement your sub-feature
Edit source files AND add/modify test files. Your sub-feature should:
- Implement ONLY the functionality described in your assignment above
- Do NOT implement the entire original feature -- only your subset
- You may use similar code patterns from the original patch for reference
- Include thorough tests that verify YOUR sub-feature works correctly
- Not break any existing functionality in the base codebase
- Modify some of the same files as at least one existing feature (to create
  merge conflicts)

### Step 4: Validate
```
cd /workspace/repo && bash /patches/validate_feature.sh
```

This checks:
1. Your changes include both source files and test files
2. Your tests pass when applied to the codebase
3. Your feature creates merge conflicts with at least one existing feature

**You MUST NOT submit until `validate_feature.sh` exits 0.**
If it fails, read the error output carefully, fix your code, and re-run.

### Step 5: Submit
Once `validate_feature.sh` exits 0, submit by running ONLY this:
```
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
Your patches are automatically saved by the validation script.
Do NOT cat any files or add anything else to this command.

## Important Rules

1. Run the setup script FIRST before doing anything else.
2. Implement ONLY your assigned sub-feature, not the whole original feature.
3. ALWAYS run `validate_feature.sh` before submitting.
4. If `validate_feature.sh` fails, iterate -- do not submit broken work.
5. Each command runs in a fresh shell. Use `cd /workspace/repo && ...` for every command.
6. Do NOT create entirely new source files just to force conflicts. Modify existing
   source files that the other features also modify.
""" + (f"\n\n## Additional Guidance\n\n{hints}\n" if hints else "")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def decompose_feature(
    task_dir: str | Path,
    feature_id: int,
    model_name: str = "gemini/gemini-3-flash-preview",
    cost_limit: float = 0.50,
    step_limit: int = 50,
    timeout: int = 600,
    backend: str = "docker",
    resolve: bool = True,
    resolve_cost_limit: float = 0.50,
    resolve_step_limit: int = 50,
    skip_heuristic: bool = False,
    hints: str | None = None,
) -> DecompositionResult:
    """Decompose an existing feature into multiple sub-features.

    Parameters
    ----------
    task_dir : path to the task directory
    feature_id : which feature to decompose (e.g. 4 for feature4)
    model_name : LLM model for both planning and generation agents
    cost_limit : max USD per sub-feature generation agent
    step_limit : max LLM calls per sub-feature generation agent
    timeout : sandbox timeout in seconds
    backend : sandbox backend
    resolve : whether to attempt MSA resolution for solvability
    resolve_cost_limit : max USD per resolution attempt
    resolve_step_limit : max steps per resolution attempt
    skip_heuristic : bypass the size heuristic and always attempt decomposition
    hints : optional strategic guidance appended to sub-feature prompts
    """
    task_dir = Path(task_dir)
    result = DecompositionResult()

    # Parse repo/task info
    parts = task_dir.parts
    repo_name = parts[-2] if len(parts) >= 2 else parts[-1]
    task_id_str = parts[-1].replace("task", "")
    try:
        task_id = int(task_id_str)
    except ValueError:
        result.error = f"Cannot parse task_id from {task_dir}"
        result.failure_reason = "bad_task_dir"
        return result

    # Load original feature
    orig_dir = task_dir / f"feature{feature_id}"
    if not orig_dir.exists():
        result.error = f"Feature directory not found: {orig_dir}"
        result.failure_reason = "feature_not_found"
        return result

    feature_patch = (orig_dir / "feature.patch").read_text()
    tests_patch_path = orig_dir / "tests.patch"
    tests_patch = tests_patch_path.read_text() if tests_patch_path.exists() else ""
    feature_md_path = orig_dir / "feature.md"
    feature_md = feature_md_path.read_text() if feature_md_path.exists() else None

    logger.info(
        "Decomposing %s/%s/feature%d (%d patch lines, %d test lines)",
        repo_name, task_dir.name, feature_id,
        len(feature_patch.splitlines()), len(tests_patch.splitlines()),
    )

    # -- Heuristic gate ------------------------------------------------------
    if not skip_heuristic:
        should, reason = _should_decompose(feature_patch, tests_patch)
        if not should:
            result.failure_reason = "too_small"
            result.error = reason
            logger.info("Heuristic says skip: %s", reason)
            return result
        logger.info("Heuristic says try: %s", reason)

    # -- Phase 1: Generate decomposition plan --------------------------------
    logger.info("Generating decomposition plan via LLM …")
    plan, plan_cost = _generate_decomposition_plan(
        feature_patch, tests_patch, feature_md, model_name,
    )
    result.plan = plan
    result.planning_cost = plan_cost
    result.total_agent_cost += plan_cost

    if not plan.decomposable:
        result.failure_reason = "not_decomposable"
        result.error = plan.reason
        logger.info("LLM says not decomposable: %s", plan.reason)
        return result

    if len(plan.sub_features) < 2:
        result.failure_reason = "too_few_sub_features"
        result.error = f"Plan has {len(plan.sub_features)} sub-features (need >= 2)"
        logger.info("Plan has too few sub-features")
        return result

    logger.info(
        "Decomposition plan: %d sub-features", len(plan.sub_features),
    )
    for i, sf in enumerate(plan.sub_features):
        logger.info("  %d. %s (files: %s)", i + 1, sf.title, sf.target_files)

    plan_json = json.dumps(asdict(plan), indent=2)

    # -- Phase 2: Per-sub-feature MSA agent loop -----------------------------
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    defaults_path = None
    try:
        import importlib.resources as ir
        defaults_path = Path(ir.files("minisweagent") / "config" / "default.yaml")
    except Exception:
        pass
    defaults = yaml.safe_load(defaults_path.read_text()) if defaults_path and defaults_path.exists() else {}
    agent_defaults = defaults.get("agent", {})
    model_defaults = defaults.get("model", {})
    env_defaults = defaults.get("environment", {})

    image = get_image_name(repo_name, task_id)

    # The "existing" set starts with the original feature being decomposed
    existing_ids = [feature_id]
    # Also include any other features already in the task
    all_existing = _get_existing_feature_ids(task_dir)
    for fid in all_existing:
        if fid not in existing_ids:
            existing_ids.append(fid)
    existing_ids.sort()

    next_feature_id = max(all_existing) + 1

    output_dir = Path("outputs") / "decompose" / f"{repo_name}_task{task_id}_f{feature_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the plan
    (output_dir / "decomposition_plan.json").write_text(plan_json)

    for sf_idx in range(len(plan.sub_features)):
        sf = plan.sub_features[sf_idx]
        sf_result = ExpansionResult()
        fid_for_this = next_feature_id + sf_idx

        logger.info(
            "=== Sub-feature %d/%d: %s (will be feature%d) ===",
            sf_idx + 1, len(plan.sub_features), sf.title, fid_for_this,
        )

        # Build prompt
        prompt = _build_subfeature_prompt(
            plan, sf_idx, task_dir, existing_ids, feature_id,
            hints=hints,
        )

        env = None
        agent = None
        try:
            model = LitellmModel(
                model_name=model_name,
                cost_tracking="ignore_errors",
                **{k: v for k, v in model_defaults.items()
                   if k not in ("model_name", "cost_tracking")},
            )
            env = DockerEnvironment(
                image=image,
                cwd="/workspace/repo",
                run_args=["--rm", "--entrypoint", ""],
                timeout=120,
                container_timeout="30m",
                forward_env=["GEMINI_API_KEY"],
                **{k: v for k, v in env_defaults.items()
                   if k not in ("image", "cwd", "run_args", "timeout",
                                "container_timeout", "forward_env")},
            )
            logger.info("Pre-seeding container %s …", env.container_id[:12])
            _preseed_decomposition_container(
                env, task_dir, existing_ids, feature_id, plan_json,
            )

            raw_agent = DefaultAgent(
                model, env,
                system_template=agent_defaults.get(
                    "system_template", "You are a helpful assistant.",
                ),
                instance_template="{{task}}",
                step_limit=step_limit,
                cost_limit=cost_limit,
            )
            agent = _LoggingAgent(raw_agent)
            agent_result = agent.run(task=prompt)

            sf_result.agent_cost = agent.cost
            sf_result.agent_steps = agent.n_calls
            exit_status = agent_result.get("exit_status", "?")
            logger.info(
                "Agent finished: status=%s, steps=%d, cost=$%.4f",
                exit_status, agent.n_calls, agent.cost,
            )

            # Extract patches from container
            logger.info("Extracting patches from container …")
            sf_feature_patch, sf_tests_patch = _extract_patches_from_container(env)

        except Exception as e:
            logger.error("MSA agent failed for sub-feature %d: %s", sf_idx + 1, e, exc_info=True)
            sf_result.error = f"Agent error: {e}"
            sf_result.failure_reason = "agent_error"
            result.sub_feature_results.append(sf_result)
            result.total_agent_cost += sf_result.agent_cost
            continue
        finally:
            if env is not None:
                env.cleanup()

        if agent is not None:
            sf_result.agent_cost = agent.cost
            sf_result.agent_steps = agent.n_calls
        result.total_agent_cost += sf_result.agent_cost

        if not sf_feature_patch.strip():
            sf_result.error = "No feature patch extracted"
            sf_result.failure_reason = "no_feature_patch"
            result.sub_feature_results.append(sf_result)
            continue
        if not sf_tests_patch.strip():
            sf_result.error = "No tests patch extracted"
            sf_result.failure_reason = "no_tests_patch"
            result.sub_feature_results.append(sf_result)
            continue

        sf_result.feature_patch = sf_feature_patch
        sf_result.tests_patch = sf_tests_patch
        logger.info(
            "Patches: feature=%d bytes, tests=%d bytes",
            len(sf_feature_patch), len(sf_tests_patch),
        )

        # Save raw patches
        sf_out = output_dir / f"sub_feature_{sf_idx + 1}"
        sf_out.mkdir(exist_ok=True)
        (sf_out / "candidate_feature.patch").write_text(sf_feature_patch)
        (sf_out / "candidate_tests.patch").write_text(sf_tests_patch)
        if agent is not None:
            try:
                traj = agent._agent.save(None)
                (sf_out / "trajectory.json").write_text(
                    json.dumps(traj, indent=2, default=str),
                )
            except Exception:
                logger.warning("Could not save trajectory", exc_info=True)

        # -- External verification: tests ------------------------------------
        logger.info("External verification: check_tests …")
        t = check_tests(
            repo_name, task_id, sf_feature_patch, sf_tests_patch,
            timeout=timeout, backend=backend,
        )
        sf_result.tests_ok = t.get("passed", False)
        if t.get("error"):
            sf_result.error = t["error"]
            sf_result.failure_reason = "ext_tests_error"
            result.sub_feature_results.append(sf_result)
            logger.info("External tests ERROR: %s", t["error"])
            continue
        if not sf_result.tests_ok:
            sf_result.failure_reason = "ext_tests_failed"
            logger.info("External tests FAILED")
            result.sub_feature_results.append(sf_result)
            continue
        logger.info("External tests PASSED")

        # -- External verification: conflicts --------------------------------
        logger.info("External verification: check_conflicts …")
        c = check_conflicts(
            repo_name, task_id, sf_feature_patch,
            existing_feature_ids=existing_ids,
            timeout=timeout, backend=backend,
        )
        sf_result.conflicts = c.get("conflicts", [])
        if not sf_result.conflicts:
            sf_result.failure_reason = "ext_no_conflicts"
            logger.info("External conflict check FAILED: no conflicts")
            result.sub_feature_results.append(sf_result)
            continue
        logger.info("Conflicts with features: %s", sf_result.conflicts)

        # -- External verification: solvability ------------------------------
        logger.info("External verification: check_solvability (resolve=%s) …", resolve)
        for cfid in sf_result.conflicts:
            logger.info("  Solvability: sub-feature vs feature %d", cfid)
            s = check_solvability(
                repo_name, task_id,
                candidate_patch=sf_feature_patch,
                candidate_tests=sf_tests_patch,
                conflicting_feature_id=cfid,
                timeout=timeout, backend=backend,
                resolve=resolve,
                model_name=model_name,
                cost_limit=resolve_cost_limit,
                step_limit=resolve_step_limit,
            )
            detail: dict = {
                "solvable": s.get("solvable", False),
                "both_passed": s.get("both_passed", False),
                "merge_strategy": (s.get("merge", {}) or {}).get("strategy"),
                "error": s.get("error"),
            }
            if "resolution" in s:
                res = s["resolution"]
                detail["resolution_strategy"] = res.get("strategy")
                detail["resolution_cost"] = res.get("agent_cost")
                detail["resolution_steps"] = res.get("agent_steps")
                detail["unsolvable_reason"] = res.get("unsolvable_reason")
                if res.get("resolution_patch"):
                    sf_result.resolution_patches[cfid] = res["resolution_patch"]
                result.total_agent_cost += res.get("agent_cost", 0)

            sf_result.solvability_details[cfid] = detail
            if s.get("solvable"):
                sf_result.solvable_pairs.append(cfid)
                logger.info("  Feature %d: SOLVABLE", cfid)
            else:
                logger.info("  Feature %d: NOT solvable", cfid)

        if not sf_result.solvable_pairs:
            sf_result.failure_reason = "not_solvable"
            logger.info("No solvable pairs -- sub-feature rejected")
            result.sub_feature_results.append(sf_result)
            continue

        # -- Generate feature.md ---------------------------------------------
        logger.info("Generating feature.md …")
        try:
            sf_result.feature_md = generate_feature_md(
                problem_statement=None,
                patch=sf_feature_patch,
                test_patch=sf_tests_patch,
                model_name=model_name,
            )
        except Exception as e:
            logger.warning("feature.md generation failed: %s", e)
            sf_result.feature_md = f"(auto-generated -- description failed: {e})"

        # -- Save to dataset directory ---------------------------------------
        sf_result.feature_id = fid_for_this
        feature_dir = task_dir / f"feature{fid_for_this}"
        feature_dir.mkdir(parents=True, exist_ok=True)

        (feature_dir / "feature.patch").write_text(sf_feature_patch)
        (feature_dir / "tests.patch").write_text(sf_tests_patch)
        (feature_dir / "feature.md").write_text(sf_result.feature_md)

        if sf_result.resolution_patches:
            res_dir = feature_dir / "resolution_patches"
            res_dir.mkdir(exist_ok=True)
            for cfid, rpatch in sf_result.resolution_patches.items():
                (res_dir / f"vs_feature{cfid}.patch").write_text(rpatch)

        sf_result.success = True
        sf_result.failure_reason = None

        logger.info(
            "ACCEPTED sub-feature %d as feature%d (solvable with: %s)",
            sf_idx + 1, fid_for_this, sf_result.solvable_pairs,
        )

        result.accepted_feature_ids.append(fid_for_this)
        result.sub_feature_results.append(sf_result)

        # Add this sub-feature to existing set for next iteration
        if fid_for_this not in existing_ids:
            existing_ids.append(fid_for_this)
            existing_ids.sort()

    # -- Final result --------------------------------------------------------
    if result.accepted_feature_ids:
        result.success = True
        result.failure_reason = None
    else:
        result.failure_reason = "no_sub_features_accepted"

    # Save result
    (output_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str),
    )

    logger.info(
        "=== DECOMPOSITION %s: %d/%d sub-features accepted (ids=%s, total_cost=$%.4f) ===",
        "SUCCEEDED" if result.success else "FAILED",
        len(result.accepted_feature_ids), len(plan.sub_features),
        result.accepted_feature_ids, result.total_agent_cost,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose an existing feature into multiple sub-features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True,
        help="Task path as repo_name/taskN (e.g. dspy_task/task8394)",
    )
    parser.add_argument(
        "--feature", required=True, type=int,
        help="Feature ID to decompose (e.g. 4 for feature4)",
    )
    parser.add_argument(
        "--model", default="gemini/gemini-3-flash-preview",
        help="LLM model (default: gemini/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--cost-limit", type=float, default=0.50,
        help="Max USD per sub-feature agent (default: 0.50)",
    )
    parser.add_argument(
        "--step-limit", type=int, default=50,
        help="Max LLM calls per sub-feature agent (default: 50)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Sandbox timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--backend", default="docker", choices=["docker", "modal"],
    )
    parser.add_argument(
        "--no-resolve", action="store_true",
        help="Skip MSA resolution in solvability check",
    )
    parser.add_argument(
        "--resolve-cost-limit", type=float, default=0.50,
        help="Max USD per resolution attempt (default: 0.50)",
    )
    parser.add_argument(
        "--resolve-step-limit", type=int, default=50,
        help="Max steps per resolution attempt (default: 50)",
    )
    parser.add_argument(
        "--skip-heuristic", action="store_true",
        help="Skip the size heuristic and always attempt decomposition",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )

    args = parser.parse_args()

    app_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("cooperbench", "__main__"):
        logging.getLogger(name).setLevel(app_level)
    for name in ("cooperbench.generation.decompose",
                 "cooperbench.generation.expand.agent",
                 "cooperbench.generation.resolve.agent",
                 "agent", "minisweagent", "minisweagent.environment"):
        logging.getLogger(name).setLevel(app_level)
    for name in ("LiteLLM", "litellm", "urllib3", "docker", "httpcore",
                 "httpx", "openai", "google", "grpc", "litellm_model"):
        logging.getLogger(name).setLevel(logging.WARNING)

    task_path = args.task.strip("/")
    parts = task_path.split("/")
    if len(parts) != 2 or not parts[1].startswith("task"):
        parser.error("--task must be repo_name/taskN, e.g. dspy_task/task8394")

    task_dir = Path("dataset") / parts[0] / parts[1]
    if not task_dir.exists():
        parser.error(f"Task directory not found: {task_dir}")

    r = decompose_feature(
        task_dir=task_dir,
        feature_id=args.feature,
        model_name=args.model,
        cost_limit=args.cost_limit,
        step_limit=args.step_limit,
        timeout=args.timeout,
        backend=args.backend,
        resolve=not args.no_resolve,
        resolve_cost_limit=args.resolve_cost_limit,
        resolve_step_limit=args.resolve_step_limit,
        skip_heuristic=args.skip_heuristic,
    )

    out = r.to_dict()
    for sfr in out.get("sub_feature_results", []):
        for key in ("feature_patch", "tests_patch", "feature_md"):
            val = sfr.get(key, "")
            if len(val) > 500:
                sfr[key] = val[:200] + f"\n... ({len(val)} bytes) ...\n" + val[-200:]
        for fid, rp in sfr.get("resolution_patches", {}).items():
            if len(rp) > 500:
                sfr["resolution_patches"][fid] = (
                    rp[:200] + f"\n... ({len(rp)} bytes) ...\n" + rp[-200:]
                )

    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")

    sys.exit(0 if r.success else 1)


if __name__ == "__main__":
    main()
