"""Feature expansion pipeline -- generates ONE additional feature for an
existing task, validates it end-to-end (tests pass, conflicts exist, jointly
solvable), and saves it to the dataset directory.

The generation agent (standalone mini-swe-agent running in the task Docker
container) iterates against an in-container validation script
(``validate_feature.sh``) that checks tests and conflicts.  After the agent
submits, we run full external verification via :mod:`verify` (including
MSA-based solvability resolution via :mod:`resolve`).

Usage::

    python -m cooperbench.generation.expand \
        --task pallets_click_task/task2068 \
        --model gemini/gemini-3-flash-preview

From Python::

    from cooperbench.generation.expand import expand_task
    result = expand_task("dataset/pallets_click_task/task2068")
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExpansionResult:
    """Outcome of a feature expansion attempt."""

    success: bool = False
    feature_id: int | None = None
    feature_patch: str = ""
    tests_patch: str = ""
    feature_md: str = ""

    agent_cost: float = 0.0
    agent_steps: int = 0

    tests_ok: bool = False
    conflicts: list[int] = field(default_factory=list)
    solvable_pairs: list[int] = field(default_factory=list)
    solvability_details: dict[int, dict] = field(default_factory=dict)
    resolution_patches: dict[int, str] = field(default_factory=dict)

    failure_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# In-container setup script
# ---------------------------------------------------------------------------

SETUP_SCRIPT = r"""#!/bin/bash
set -e
cd /workspace/repo
git config user.email "expand@cooperbench.local"
git config user.name "CooperBench Expander"
BASE_SHA=$(git rev-parse HEAD)
echo "$BASE_SHA" > /patches/.base_sha
echo "Base SHA: $BASE_SHA"
echo "SETUP COMPLETE -- you can now start writing your feature."
"""


# ---------------------------------------------------------------------------
# In-container validation script
# ---------------------------------------------------------------------------

VALIDATE_FEATURE_SH = r"""#!/bin/bash
# validate_feature.sh -- Generation agent MUST run this before submitting.
# Exits 0 only when:
#   1. Code changes exist (both feature + test files)
#   2. Tests pass (via runner.sh)
#   3. At least one merge conflict with an existing feature
# Non-zero exit blocks submission.

set -e
cd /workspace/repo

if [ ! -f /patches/.base_sha ]; then
    echo "ERROR: Run 'bash /patches/setup.sh' first!"
    exit 1
fi
BASE_SHA=$(cat /patches/.base_sha)

# ── Step 1: Extract patches from current changes ──────────────────────────
echo "=== Step 1: Extracting patches ==="

git add -A

CHANGED=$(git diff --cached --name-only "$BASE_SHA" 2>/dev/null || true)
if [ -z "$CHANGED" ]; then
    echo "ERROR: No changes detected. Write your feature code and tests first!"
    exit 1
fi

TEST_FILES=""
FEATURE_FILES=""
for f in $CHANGED; do
    if echo "$f" | grep -qE '(^tests/|/tests/|^test_|/test_|_test\.py$|/test/)'; then
        TEST_FILES="$TEST_FILES $f"
    else
        FEATURE_FILES="$FEATURE_FILES $f"
    fi
done

if [ -z "$FEATURE_FILES" ]; then
    echo "ERROR: No source files changed. Your feature must modify source code, not just tests."
    exit 1
fi
if [ -z "$TEST_FILES" ]; then
    echo "ERROR: No test files changed. Your feature must include tests."
    exit 1
fi

git diff --cached "$BASE_SHA" -- $FEATURE_FILES > /patches/candidate_feature.patch
git diff --cached "$BASE_SHA" -- $TEST_FILES > /patches/candidate_tests.patch

echo "Feature files:$FEATURE_FILES"
echo "Test files:$TEST_FILES"
echo "Feature patch: $(wc -l < /patches/candidate_feature.patch) lines"
echo "Tests patch: $(wc -l < /patches/candidate_tests.patch) lines"

# ── Step 2: Run tests ─────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Testing that your tests pass ==="

git checkout "$BASE_SHA" --force 2>/dev/null
git reset --hard 2>/dev/null
git clean -fdx 2>/dev/null

# runner.sh runs the task's default test suite -- ensures base tests still pass
bash /usr/local/bin/runner.sh candidate_tests.patch candidate_feature.patch
echo "Base tests PASSED"

# Also run any NEW test files the agent created (runner.sh only runs pre-set tests)
git checkout "$BASE_SHA" --force 2>/dev/null
git reset --hard 2>/dev/null
git clean -fdx 2>/dev/null
git apply /patches/candidate_feature.patch 2>/dev/null || git apply --3way /patches/candidate_feature.patch 2>/dev/null
git apply /patches/candidate_tests.patch 2>/dev/null || git apply --3way /patches/candidate_tests.patch 2>/dev/null
pip install -e ".[test]" > /dev/null 2>&1 || pip install -e . > /dev/null 2>&1
pip install pytest pytest-xdist pytest_mock > /dev/null 2>&1

NEW_TEST_FILES=$(echo "$TEST_FILES" | tr ' ' '\n' | head -5)
echo "Running new test files: $NEW_TEST_FILES"
python -m pytest $NEW_TEST_FILES -v
echo "New tests PASSED"

# ── Step 3: Check merge conflicts with existing features ──────────────────
echo ""
echo "=== Step 3: Checking for merge conflicts with existing features ==="

git config user.email "expand@cooperbench.local" 2>/dev/null || true
git config user.name "CooperBench Expander" 2>/dev/null || true

CONFLICT_COUNT=0
CONFLICT_LIST=""
TOTAL_CHECKED=0

for feature_dir in /patches/existing/feature*/; do
    [ -d "$feature_dir" ] || continue
    FEATURE_ID=$(basename "$feature_dir" | sed 's/feature//')
    EXISTING_PATCH="$feature_dir/feature.patch"
    [ -f "$EXISTING_PATCH" ] || continue
    TOTAL_CHECKED=$((TOTAL_CHECKED + 1))

    # Clean state
    git checkout "$BASE_SHA" --force 2>/dev/null
    git reset --hard 2>/dev/null
    git clean -fdx 2>/dev/null

    # Delete temp branches from prior runs
    git branch -D "existing_$FEATURE_ID" 2>/dev/null || true
    git branch -D "candidate_vs_$FEATURE_ID" 2>/dev/null || true

    # Branch: existing feature
    git checkout -b "existing_$FEATURE_ID" 2>/dev/null
    if ! git apply "$EXISTING_PATCH" 2>/dev/null && ! git apply --3way "$EXISTING_PATCH" 2>/dev/null; then
        echo "  Feature $FEATURE_ID: could not apply existing patch (skip)"
        git checkout "$BASE_SHA" --force 2>/dev/null
        continue
    fi
    git add -A && git commit -m "Existing feature $FEATURE_ID" --allow-empty 2>/dev/null

    # Branch: candidate feature
    git checkout "$BASE_SHA" --force 2>/dev/null
    git checkout -b "candidate_vs_$FEATURE_ID" 2>/dev/null
    git apply /patches/candidate_feature.patch 2>/dev/null || git apply --3way /patches/candidate_feature.patch 2>/dev/null
    git add -A && git commit -m "Candidate feature" --allow-empty 2>/dev/null

    # Attempt merge
    MERGE_OUTPUT=$(git merge "existing_$FEATURE_ID" --no-ff --no-commit 2>&1 || true)
    if echo "$MERGE_OUTPUT" | grep -qi "conflict"; then
        CONFLICT_COUNT=$((CONFLICT_COUNT + 1))
        CONFLICT_LIST="$CONFLICT_LIST $FEATURE_ID"
        echo "  Feature $FEATURE_ID: CONFLICT (good!)"
    else
        echo "  Feature $FEATURE_ID: clean merge (no conflict)"
    fi

    git merge --abort 2>/dev/null || true
    git checkout "$BASE_SHA" --force 2>/dev/null
done

echo ""
echo "Checked $TOTAL_CHECKED existing feature(s). Conflicts: $CONFLICT_COUNT (features:$CONFLICT_LIST)"

if [ "$CONFLICT_COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: Your feature must create merge conflicts with at least one existing feature."
    echo "Your changes should modify overlapping code regions (same functions/methods/lines)"
    echo "as at least one of the existing features. Re-examine the existing feature patches"
    echo "at /patches/existing/feature*/feature.patch and adjust your code."
    exit 1
fi

# ── Mark validation as passed ─────────────────────────────────────────────
echo "PASSED" > /patches/.validation_status

# ── Restore agent working state ──────────────────────────────────────────
git checkout "$BASE_SHA" --force 2>/dev/null
git reset --hard 2>/dev/null
git clean -fdx 2>/dev/null
git apply /patches/candidate_feature.patch 2>/dev/null || git apply --3way /patches/candidate_feature.patch 2>/dev/null
git apply /patches/candidate_tests.patch 2>/dev/null || git apply --3way /patches/candidate_tests.patch 2>/dev/null

echo ""
echo "VALIDATION PASSED (tests pass, conflicts with features:$CONFLICT_LIST)"
echo "Patches saved to /patches/candidate_feature.patch and /patches/candidate_tests.patch"
echo ""
echo "Now submit by running exactly: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
"""


# ---------------------------------------------------------------------------
# Logging agent subclass (reuse pattern from resolve.py)
# ---------------------------------------------------------------------------

class _LoggingAgent:
    """Wraps standalone minisweagent DefaultAgent to add INFO-level traces."""

    def __init__(self, agent):
        self._agent = agent
        self._step = 0
        self._logger = logging.getLogger("cooperbench.generation.expand.agent")

    def run(self, task: str = "", **kwargs) -> dict:
        original_step = self._agent.step

        def logged_step():
            self._step += 1
            self._logger.info(
                "--- Step %d (calls=%d, cost=$%.4f) ---",
                self._step, self._agent.n_calls, self._agent.cost,
            )
            return original_step()

        self._agent.step = logged_step

        original_execute = self._agent.execute_actions

        def logged_execute(message):
            content = message.get("content") or ""
            if content.strip():
                self._logger.info("THOUGHT: %s", content.strip()[:300])
            for action in message.get("extra", {}).get("actions", []):
                cmd = action.get("command", "")
                self._logger.info("CMD: %s", cmd[:500])
            results = original_execute(message)
            for msg in results:
                out = msg.get("content", "")
                if out and len(out) > 800:
                    self._logger.info(
                        "OUTPUT (%d chars): %s … %s",
                        len(out), out[:200], out[-200:],
                    )
                elif out:
                    self._logger.info("OUTPUT: %s", out)
            return results

        self._agent.execute_actions = logged_execute

        result = self._agent.run(task=task, **kwargs)
        self._logger.info(
            "Agent finished: status=%s, steps=%d, cost=$%.4f",
            result.get("exit_status", "?"), self._agent.n_calls, self._agent.cost,
        )
        return result

    @property
    def cost(self):
        return self._agent.cost

    @property
    def n_calls(self):
        return self._agent.n_calls

    @property
    def messages(self):
        return self._agent.messages


# ---------------------------------------------------------------------------
# Pre-seeding the container
# ---------------------------------------------------------------------------

def _preseed_generation_container(
    env,
    task_dir: Path,
    existing_feature_ids: list[int],
) -> None:
    """Copy validation script, setup script, and existing feature data into the
    MSA container before the agent starts.

    Layout inside the container::

        /patches/setup.sh
        /patches/validate_feature.sh
        /patches/existing/feature{N}/feature.patch
        /patches/existing/feature{N}/tests.patch
        /patches/feature_descriptions/feature{N}.md
    """
    docker = env.config.executable
    cid = env.container_id

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        (tmp / "setup.sh").write_text(SETUP_SCRIPT)
        (tmp / "validate_feature.sh").write_text(VALIDATE_FEATURE_SH)

        for fid in existing_feature_ids:
            fdir = task_dir / f"feature{fid}"

            patch_file = fdir / "feature.patch"
            tests_file = fdir / "tests.patch"
            md_file = fdir / "feature.md"

            out_dir = tmp / "existing" / f"feature{fid}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if patch_file.exists():
                (out_dir / "feature.patch").write_text(patch_file.read_text())
            if tests_file.exists():
                (out_dir / "tests.patch").write_text(tests_file.read_text())

            desc_dir = tmp / "feature_descriptions"
            desc_dir.mkdir(exist_ok=True)
            if md_file.exists():
                (desc_dir / f"feature{fid}.md").write_text(md_file.read_text())

        subprocess.run(
            [docker, "exec", cid, "mkdir", "-p", "/patches"],
            check=True,
        )
        subprocess.run(
            [docker, "cp", f"{tmpdir}/.", f"{cid}:/patches/"],
            check=True,
        )
        subprocess.run(
            [docker, "exec", cid, "bash", "-c",
             "chmod +x /patches/setup.sh /patches/validate_feature.sh"],
            check=True,
        )


# ---------------------------------------------------------------------------
# Generation prompt builder
# ---------------------------------------------------------------------------

def _build_generation_prompt(
    task_dir: Path,
    existing_feature_ids: list[int],
) -> str:
    """Construct the MSA prompt for the generation agent.

    Describes the existing features (from their feature.md files) and
    instructs the agent to write a new feature with tests that conflicts with
    at least one existing feature.
    """
    feature_sections = []
    for fid in existing_feature_ids:
        md_path = task_dir / f"feature{fid}" / "feature.md"
        if md_path.exists():
            content = md_path.read_text().strip()
        else:
            content = "(no description available)"
        feature_sections.append(f"### Existing Feature {fid}\n{content}")

    existing_desc = "\n\n".join(feature_sections)

    patch_listing = "\n".join(
        f"- `/patches/existing/feature{fid}/feature.patch`"
        for fid in existing_feature_ids
    )

    return f"""You are a software engineer generating a NEW feature for this codebase.
Your goal is to create a meaningful, non-trivial feature enhancement along with
comprehensive tests -- and the feature MUST modify overlapping code regions so
that it creates merge conflicts with at least one existing feature.

## Existing Features

The following features already exist in this codebase. Your new feature should
be DIFFERENT from all of these but must touch some of the same files/functions
so that merge conflicts arise.

{existing_desc}

## Pre-loaded files

- `/patches/setup.sh` -- run this FIRST to initialise the environment
- `/patches/validate_feature.sh` -- run this to check your work before submitting
- Existing feature patches (for reference):
{patch_listing}
- Existing feature descriptions: `/patches/feature_descriptions/feature*.md`

## Instructions

### Step 1: Setup
```
cd /workspace/repo && bash /patches/setup.sh
```

### Step 2: Explore the codebase
Read the existing feature patches and the source code to understand:
- What the existing features do
- Which files and functions they modify
- What kind of feature would be valuable AND create merge conflicts

### Step 3: Implement your feature
Write your feature by editing source files AND adding/modifying test files.
Your feature should:
- Be a genuine, useful enhancement (not artificial or trivial)
- Modify some of the same functions/methods as at least one existing feature
- Include thorough tests that verify your feature works correctly
- Not break any existing functionality in the base codebase

### Step 4: Validate
Run the validation script:
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

## Important rules

1. Run the setup script FIRST before doing anything else.
2. ALWAYS run `validate_feature.sh` before submitting.
3. If `validate_feature.sh` fails, iterate -- do not submit broken work.
4. Your feature must be genuinely useful, not just a dummy change to create conflicts.
5. Each command runs in a fresh shell. Use `cd /workspace/repo && ...` for every command.
6. Do NOT create entirely new source files just to force conflicts. Modify existing
   source files that the other features also modify.
"""


# ---------------------------------------------------------------------------
# Container file extraction
# ---------------------------------------------------------------------------

def _extract_patches_from_container(env) -> tuple[str, str]:
    """Read candidate patches from the container's /patches/ directory.

    Returns (feature_patch, tests_patch).  Either may be empty if files
    don't exist (agent didn't reach validation).
    """
    docker = env.config.executable
    cid = env.container_id

    def _read_file(path: str) -> str:
        r = subprocess.run(
            [docker, "exec", cid, "cat", path],
            capture_output=True, text=True,
        )
        return r.stdout if r.returncode == 0 else ""

    status = _read_file("/patches/.validation_status").strip()
    if status != "PASSED":
        logger.warning("Validation status file says %r (expected 'PASSED')", status)

    feature_patch = _read_file("/patches/candidate_feature.patch")
    tests_patch = _read_file("/patches/candidate_tests.patch")

    if feature_patch and not feature_patch.endswith("\n"):
        feature_patch += "\n"
    if tests_patch and not tests_patch.endswith("\n"):
        tests_patch += "\n"

    return feature_patch, tests_patch


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def expand_task(
    task_dir: str | Path,
    model_name: str = "gemini/gemini-3-flash-preview",
    cost_limit: float = 0.50,
    step_limit: int = 50,
    timeout: int = 600,
    backend: str = "docker",
    resolve: bool = True,
    resolve_cost_limit: float = 0.50,
    resolve_step_limit: int = 50,
) -> ExpansionResult:
    """Generate one additional feature for an existing task.

    Parameters
    ----------
    task_dir : path to the task directory (e.g. ``dataset/pallets_click_task/task2068``)
    model_name : LLM model for the generation agent
    cost_limit : max USD spend for the generation agent
    step_limit : max LLM calls for the generation agent
    timeout : sandbox timeout in seconds
    backend : sandbox backend
    resolve : whether to attempt MSA-based resolution for solvability
    resolve_cost_limit : max USD per MSA resolution attempt
    resolve_step_limit : max steps per MSA resolution attempt
    """
    task_dir = Path(task_dir)
    result = ExpansionResult()

    parts = task_dir.parts
    repo_name = parts[-2] if len(parts) >= 2 else parts[-1]
    task_id_str = parts[-1].replace("task", "")
    try:
        task_id = int(task_id_str)
    except ValueError:
        result.error = f"Cannot parse task_id from {task_dir}"
        result.failure_reason = "bad_task_dir"
        return result

    existing_ids = _get_existing_feature_ids(task_dir)
    if not existing_ids:
        result.error = f"No existing features found in {task_dir}"
        result.failure_reason = "no_existing_features"
        return result

    next_feature_id = max(existing_ids) + 1
    logger.info(
        "Expanding %s/%s: %d existing features (%s), generating feature%d",
        repo_name, task_dir.name, len(existing_ids), existing_ids, next_feature_id,
    )

    # -- Step 1: Build prompt -----------------------------------------------
    prompt = _build_generation_prompt(task_dir, existing_ids)

    # -- Step 2: Invoke standalone mini-swe-agent ----------------------------
    logger.info(
        "Invoking MSA generation agent (model=%s, cost=$%.2f, steps=%d) …",
        model_name, cost_limit, step_limit,
    )

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
    env = None
    agent = None
    agent_result = None
    try:
        model = LitellmModel(
            model_name=model_name,
            cost_tracking="ignore_errors",
            **{k: v for k, v in model_defaults.items() if k not in ("model_name", "cost_tracking")},
        )
        env = DockerEnvironment(
            image=image,
            cwd="/workspace/repo",
            run_args=["--rm", "--entrypoint", ""],
            timeout=120,
            container_timeout="30m",
            forward_env=["GEMINI_API_KEY"],
            **{k: v for k, v in env_defaults.items() if k not in (
                "image", "cwd", "run_args", "timeout", "container_timeout", "forward_env",
            )},
        )
        logger.info("Pre-seeding container %s …", env.container_id[:12])
        _preseed_generation_container(env, task_dir, existing_ids)

        raw_agent = DefaultAgent(
            model, env,
            system_template=agent_defaults.get("system_template", "You are a helpful assistant."),
            instance_template="{{task}}",
            step_limit=step_limit,
            cost_limit=cost_limit,
        )
        agent = _LoggingAgent(raw_agent)
        agent_result = agent.run(task=prompt)

        result.agent_cost = agent.cost
        result.agent_steps = agent.n_calls
        exit_status = agent_result.get("exit_status", "?")
        logger.info(
            "Generation agent finished: status=%s, steps=%d, cost=$%.4f",
            exit_status, agent.n_calls, agent.cost,
        )

        # -- Step 3: Extract patches from container -------------------------
        # The agent writes patches to /patches/ via validate_feature.sh.
        # We extract them directly instead of parsing the submission text,
        # which avoids format errors from MSA's output truncation.
        logger.info("Extracting patches from container …")
        feature_patch, tests_patch = _extract_patches_from_container(env)

    except Exception as e:
        logger.error("MSA generation agent failed: %s", e, exc_info=True)
        result.error = f"MSA generation agent failed: {e}"
        result.failure_reason = "agent_error"
        return result
    finally:
        if env is not None:
            env.cleanup()

    if agent is not None:
        result.agent_cost = agent.cost
        result.agent_steps = agent.n_calls

    if not feature_patch.strip():
        result.error = "No feature patch found in container (agent may not have passed validation)"
        result.failure_reason = "no_feature_patch"
        return result
    if not tests_patch.strip():
        result.error = "No tests patch found in container (agent may not have passed validation)"
        result.failure_reason = "no_tests_patch"
        return result

    result.feature_patch = feature_patch
    result.tests_patch = tests_patch
    logger.info(
        "Extracted patches: feature=%d bytes, tests=%d bytes",
        len(feature_patch), len(tests_patch),
    )

    # Save raw patches for inspection
    output_dir = Path("outputs") / "expand" / f"{repo_name}_task{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_feature.patch").write_text(feature_patch)
    (output_dir / "candidate_tests.patch").write_text(tests_patch)
    if agent is not None:
        try:
            traj_data = agent._agent.save(None)
            traj_path = output_dir / "trajectory.json"
            traj_path.write_text(json.dumps(traj_data, indent=2, default=str))
        except Exception:
            logger.warning("Could not save trajectory", exc_info=True)
    logger.info("Saved candidate patches and trajectory to %s", output_dir)

    # -- Step 4: External verification – tests ------------------------------
    logger.info("External verification: check_tests …")
    t = check_tests(
        repo_name, task_id, feature_patch, tests_patch,
        timeout=timeout, backend=backend,
    )
    result.tests_ok = t.get("passed", False)
    if t.get("error"):
        result.error = t["error"]
        result.failure_reason = "ext_tests_error"
        return result
    if not result.tests_ok:
        result.failure_reason = "ext_tests_failed"
        logger.info("External test check FAILED (passed=%s, failed=%s)",
                     t.get("tests_passed"), t.get("tests_failed"))
        return result
    logger.info("External tests PASSED (%d passed)", t.get("tests_passed", 0))

    # -- Step 5: External verification – conflicts --------------------------
    logger.info("External verification: check_conflicts …")
    c = check_conflicts(
        repo_name, task_id, feature_patch,
        existing_feature_ids=existing_ids,
        timeout=timeout, backend=backend,
    )
    result.conflicts = c.get("conflicts", [])
    if not result.conflicts:
        result.failure_reason = "ext_no_conflicts"
        logger.info("External conflict check FAILED: no conflicts found")
        return result
    logger.info("External conflicts found with features: %s", result.conflicts)

    # -- Step 6: External verification – solvability ------------------------
    logger.info("External verification: check_solvability (resolve=%s) …", resolve)
    for fid in result.conflicts:
        logger.info("  Solvability check: candidate vs feature %d", fid)
        s = check_solvability(
            repo_name, task_id,
            candidate_patch=feature_patch,
            candidate_tests=tests_patch,
            conflicting_feature_id=fid,
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
                result.resolution_patches[fid] = res["resolution_patch"]

        result.solvability_details[fid] = detail
        if s.get("solvable"):
            result.solvable_pairs.append(fid)
            logger.info("  Feature %d: SOLVABLE", fid)
        else:
            logger.info("  Feature %d: NOT solvable (error=%s)", fid, detail.get("error"))

    if not result.solvable_pairs:
        result.failure_reason = "not_solvable"
        logger.info("No solvable pairs found -- feature rejected")
        return result

    # -- Step 7: Generate feature.md via LLM --------------------------------
    logger.info("Generating feature.md …")
    try:
        result.feature_md = generate_feature_md(
            problem_statement=None,
            patch=feature_patch,
            test_patch=tests_patch,
            model_name=model_name,
        )
    except Exception as e:
        logger.warning("feature.md generation failed (%s); using placeholder", e)
        result.feature_md = f"(auto-generated feature -- description generation failed: {e})"

    # -- Step 8: Save to dataset directory ----------------------------------
    result.feature_id = next_feature_id
    feature_dir = task_dir / f"feature{next_feature_id}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    (feature_dir / "feature.patch").write_text(feature_patch)
    (feature_dir / "tests.patch").write_text(tests_patch)
    (feature_dir / "feature.md").write_text(result.feature_md)

    if result.resolution_patches:
        res_dir = feature_dir / "resolution_patches"
        res_dir.mkdir(exist_ok=True)
        for fid, rpatch in result.resolution_patches.items():
            (res_dir / f"vs_feature{fid}.patch").write_text(rpatch)

    logger.info(
        "Saved feature%d to %s (solvable with: %s)",
        next_feature_id, feature_dir, result.solvable_pairs,
    )

    result.success = True
    result.failure_reason = None

    # Save full result JSON to outputs dir
    (output_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str)
    )
    logger.info(
        "=== EXPANSION SUCCEEDED: feature%d (cost=$%.4f, steps=%d, solvable_with=%s) ===",
        next_feature_id, result.agent_cost, result.agent_steps, result.solvable_pairs,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one additional feature for an existing CooperBench task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True,
        help="Task path as repo_name/taskN (e.g. pallets_click_task/task2068)",
    )
    parser.add_argument(
        "--model", default="gemini/gemini-3-flash-preview",
        help="LLM model for generation agent (default: gemini/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--cost-limit", type=float, default=0.50,
        help="Max USD for generation agent (default: 0.50)",
    )
    parser.add_argument(
        "--step-limit", type=int, default=50,
        help="Max LLM calls for generation agent (default: 50)",
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
        help="Skip MSA-based resolution in solvability check (auto-merge only)",
    )
    parser.add_argument(
        "--resolve-cost-limit", type=float, default=0.50,
        help="Max USD per MSA resolution attempt (default: 0.50)",
    )
    parser.add_argument(
        "--resolve-step-limit", type=int, default=50,
        help="Max steps per MSA resolution attempt (default: 50)",
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
    for name in ("cooperbench.generation.expand.agent", "cooperbench.generation.resolve.agent",
                 "agent", "minisweagent", "minisweagent.environment"):
        logging.getLogger(name).setLevel(app_level)
    for name in ("LiteLLM", "litellm", "urllib3", "docker", "httpcore",
                 "httpx", "openai", "google", "grpc", "litellm_model"):
        logging.getLogger(name).setLevel(logging.WARNING)

    task_path = args.task.strip("/")
    parts = task_path.split("/")
    if len(parts) != 2 or not parts[1].startswith("task"):
        parser.error("--task must be repo_name/taskN, e.g. pallets_click_task/task2068")

    task_dir = Path("dataset") / parts[0] / parts[1]
    if not task_dir.exists():
        parser.error(f"Task directory not found: {task_dir}")

    result = expand_task(
        task_dir=task_dir,
        model_name=args.model,
        cost_limit=args.cost_limit,
        step_limit=args.step_limit,
        timeout=args.timeout,
        backend=args.backend,
        resolve=not args.no_resolve,
        resolve_cost_limit=args.resolve_cost_limit,
        resolve_step_limit=args.resolve_step_limit,
    )

    out = result.to_dict()
    for key in ("feature_patch", "tests_patch", "feature_md"):
        val = out.get(key, "")
        if len(val) > 500:
            out[key] = val[:200] + f"\n... ({len(val)} bytes total) ...\n" + val[-200:]
    for fid, rpatch in out.get("resolution_patches", {}).items():
        if len(rpatch) > 500:
            out["resolution_patches"][fid] = rpatch[:200] + f"\n... ({len(rpatch)} bytes) ...\n" + rpatch[-200:]

    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
