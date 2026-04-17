"""MSA-based conflict resolution for jointly-solvable feature pairs.

When two features conflict on merge and automatic resolution (naive + union)
fails, this module hands the conflict to the **standalone mini-swe-agent**
running inside the task Docker container.  MSA resolves the conflict, tests
both suites via ``verify_resolve.sh``, and iterates until they pass -- all
within a cost/step budget.  After MSA returns, we do a final external
verification.

Usage from Python::

    from cooperbench.generation.resolve import resolve_conflict

    result = resolve_conflict(
        "pallets_click_task", 2068,
        patch1=existing_patch, patch2=candidate_patch,
        tests1=existing_tests, tests2=candidate_tests,
        feature_md1=existing_md, feature_md2=candidate_md,
    )
    print(result.resolved, result.strategy)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from cooperbench.eval.backends import get_backend
from cooperbench.eval.sandbox import (
    _filter_test_files,
    _parse_results,
    _setup_branches,
    _write_patch,
    merge_and_test,
)
from cooperbench.utils import get_image_name

logger = logging.getLogger(__name__)

UNSOLVABLE_MARKER = "UNSOLVABLE:"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResolutionResult:
    """Outcome of attempting to resolve a merge conflict."""

    resolved: bool = False
    resolution_patch: str = ""
    strategy: str = ""              # "auto" | "msa" | "msa_unsolvable"
    both_passed: bool = False
    feature1_passed: bool = False
    feature2_passed: bool = False
    agent_cost: float = 0.0
    agent_steps: int = 0
    unsolvable_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# In-container validation script
# ---------------------------------------------------------------------------

VERIFY_RESOLVE_SH = r"""#!/bin/bash
# verify_resolve.sh -- MSA must run this before submitting.
# Exits 0 only when BOTH test suites pass.  Non-zero blocks submission.
# Pass --unsolvable "reason" to declare the pair genuinely unmergeable.

if [ "$1" = "--unsolvable" ]; then
    shift
    REASON="$*"
    echo "UNSOLVABLE: $REASON" | tee /patches/.unsolvable
    echo "Marked as unsolvable. You may now submit with:"
    echo "  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo 'UNSOLVABLE: $REASON'"
    exit 0
fi

set -e
cd /workspace/repo

BASE_SHA=$(cat /patches/.base_sha)

echo "=== Generating resolution patch ==="
# Remove any temp scripts the agent may have created (resolve_*.py, etc.)
# so they don't pollute the resolution patch.
git ls-files --others --exclude-standard | xargs -r rm -f 2>/dev/null || true

# Only stage changes to files that existed at BASE_SHA (source files),
# not any new files the agent created as helper tools.
TRACKED_FILES=$(git diff --name-only "$BASE_SHA" -- 2>/dev/null || true)
if [ -n "$TRACKED_FILES" ]; then
    echo "$TRACKED_FILES" | xargs git add --
fi

# Generate diff of only staged source changes vs the base commit.
git diff --cached "$BASE_SHA" > /patches/resolution.patch

if [ ! -s /patches/resolution.patch ]; then
    echo "ERROR: resolution patch is empty. Did you forget to resolve the conflicts?"
    exit 1
fi

# Reject patches that still contain merge conflict markers
if grep -qE '^[<>=]{7}' /patches/resolution.patch; then
    echo "ERROR: resolution patch still contains merge conflict markers."
    echo "Unresolved conflicts:"
    grep -nE '^[<>=]{7}' /patches/resolution.patch
    exit 1
fi

# Also reject source files with leftover conflict markers
CONFLICT_FILES=$(grep -rlE '^[<>=]{7}' --include='*.py' . 2>/dev/null || true)
if [ -n "$CONFLICT_FILES" ]; then
    echo "ERROR: source files still contain conflict markers:"
    echo "$CONFLICT_FILES"
    for cf in $CONFLICT_FILES; do
        echo "--- $cf ---"
        grep -nE '^[<>=]{7}' "$cf"
    done
    exit 1
fi

echo "Resolution patch: $(wc -l < /patches/resolution.patch) lines"

echo "=== Testing Feature 1 ==="
git checkout "$BASE_SHA" --force 2>/dev/null
git reset --hard 2>/dev/null
git clean -fdx 2>/dev/null
bash /usr/local/bin/runner.sh tests1.patch resolution.patch
echo "Feature 1 tests PASSED"

echo "=== Testing Feature 2 ==="
git checkout "$BASE_SHA" --force 2>/dev/null
git reset --hard 2>/dev/null
git clean -fdx 2>/dev/null
bash /usr/local/bin/runner.sh tests2.patch resolution.patch
echo "Feature 2 tests PASSED"

echo "VERIFICATION PASSED -- both test suites pass."
echo "Submit with: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /patches/resolution.patch"
"""


# ---------------------------------------------------------------------------
# Logging agent subclass
# ---------------------------------------------------------------------------

class _LoggingAgent:
    """Wraps the standalone minisweagent DefaultAgent to add INFO-level
    step traces (thought, command, output, cost) so we can actually see
    what the agent is doing without drowning in litellm debug spam."""

    def __init__(self, agent):
        self._agent = agent
        self._step = 0
        self._logger = logging.getLogger("cooperbench.generation.resolve.agent")

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
                thought = content.strip()[:300]
                self._logger.info("THOUGHT: %s", thought)

            for action in message.get("extra", {}).get("actions", []):
                cmd = action.get("command", "")
                self._logger.info("CMD: %s", cmd[:500])

            results = original_execute(message)

            for msg in results:
                out = msg.get("content", "")
                if out and len(out) > 800:
                    self._logger.info("OUTPUT (%d chars): %s … %s", len(out), out[:200], out[-200:])
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
# Conflict state extraction
# ---------------------------------------------------------------------------

def _extract_conflict_state(
    repo_name: str,
    task_id: int,
    patch1: str,
    patch2: str,
    timeout: int = 300,
    backend: str = "docker",
) -> dict:
    """Apply two patches to branches and capture the merge-conflict state.

    Returns dict with:
        conflicted_files: dict[filepath, content_with_markers]
        summary: human-readable summary
        base_sha: the base commit SHA
        error: str | None
    """
    image = get_image_name(repo_name, task_id)
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)

    try:
        _write_patch(sb, "patch1.patch", patch1)
        _write_patch(sb, "patch2.patch", patch2)

        setup = _setup_branches(sb)
        if setup.get("error"):
            return {"conflicted_files": {}, "summary": "", "base_sha": "", "error": setup["error"]}

        base_sha = setup.get("base_sha", "")

        merge_script = """
cd /workspace/repo
git checkout agent2 2>&1
git merge agent1 --no-commit --no-ff 2>&1 || true
# List conflicted files
CONFLICTED=$(git diff --name-only --diff-filter=U 2>/dev/null)
if [ -z "$CONFLICTED" ]; then
    echo "NO_CONFLICTS"
else
    echo "CONFLICTED_FILES_START"
    for f in $CONFLICTED; do
        echo "FILE:$f"
        cat "$f" 2>/dev/null
        echo "FILE_END:$f"
    done
    echo "CONFLICTED_FILES_END"
fi
"""
        result = sb.exec("bash", "-c", merge_script)
        output = result.stdout_read() + result.stderr_read()

        if "NO_CONFLICTS" in output:
            return {"conflicted_files": {}, "summary": "No conflicts detected", "base_sha": base_sha, "error": None}

        conflicted_files: dict[str, str] = {}
        current_file = None
        current_content: list[str] = []

        for line in output.split("\n"):
            if line.startswith("FILE:") and not line.startswith("FILE_END:"):
                if current_file:
                    conflicted_files[current_file] = "\n".join(current_content)
                current_file = line[5:]
                current_content = []
            elif line.startswith("FILE_END:"):
                if current_file:
                    conflicted_files[current_file] = "\n".join(current_content)
                current_file = None
                current_content = []
            elif current_file is not None:
                current_content.append(line)

        summary = f"{len(conflicted_files)} file(s) with conflicts: {', '.join(conflicted_files.keys())}"
        return {"conflicted_files": conflicted_files, "summary": summary, "base_sha": base_sha, "error": None}

    except Exception as e:
        return {"conflicted_files": {}, "summary": "", "base_sha": "", "error": str(e)}
    finally:
        sb.terminate()


# ---------------------------------------------------------------------------
# MSA prompt builder
# ---------------------------------------------------------------------------

SETUP_SCRIPT = r"""#!/bin/bash
# Recreates the merge-conflict state. Patches and tests are already at /patches/.
set -e
cd /workspace/repo
git config user.email "resolve@cooperbench.local"
git config user.name "CooperBench Resolver"

BASE_SHA=$(git rev-parse HEAD)
echo "$BASE_SHA" > /patches/.base_sha
echo "Base SHA: $BASE_SHA"

# Branch: feature1
git checkout -b feature1 2>&1
git apply /patches/patch1.patch 2>&1 || git apply --3way /patches/patch1.patch 2>&1
git add -A && git commit -m "Feature 1" --allow-empty 2>&1

# Branch: feature2 (from base)
git checkout "$BASE_SHA" 2>&1
git checkout -b feature2 2>&1
git apply /patches/patch2.patch 2>&1 || git apply --3way /patches/patch2.patch 2>&1
git add -A && git commit -m "Feature 2" --allow-empty 2>&1

# Attempt merge (will create conflict markers)
git merge feature1 --no-ff --no-commit 2>&1 || true

echo "SETUP COMPLETE -- resolve the conflicts above, then run: bash /usr/local/bin/verify_resolve.sh"
"""


def _preseed_container(
    env,
    patch1: str,
    patch2: str,
    tests1: str,
    tests2: str,
) -> None:
    """Write all required files into the MSA container before the agent starts.

    Uses ``docker cp`` so we avoid base64 encoding in agent prompts.
    """
    import subprocess
    import tempfile

    docker = env.config.executable
    cid = env.container_id

    with tempfile.TemporaryDirectory() as tmpdir:
        files = {
            "patch1.patch": patch1,
            "patch2.patch": patch2,
            "tests1.patch": tests1,
            "tests2.patch": tests2,
            "verify_resolve.sh": VERIFY_RESOLVE_SH,
            "setup.sh": SETUP_SCRIPT,
        }
        for name, content in files.items():
            Path(tmpdir, name).write_text(content)

        subprocess.run([docker, "exec", cid, "mkdir", "-p", "/patches"], check=True)
        for name in files:
            subprocess.run(
                [docker, "cp", str(Path(tmpdir, name)), f"{cid}:/patches/{name}"],
                check=True,
            )
        subprocess.run(
            [docker, "exec", cid, "bash", "-c",
             "chmod +x /patches/verify_resolve.sh /patches/setup.sh && "
             "cp /patches/verify_resolve.sh /usr/local/bin/verify_resolve.sh"],
            check=True,
        )


def _build_resolver_prompt(
    feature_md1: str | None,
    feature_md2: str | None,
    conflict_state: dict,
) -> str:
    """Build the full task prompt for MSA.

    Patches, tests, and scripts have already been pre-seeded into the container
    at ``/patches/`` via ``_preseed_container``.
    """
    conflict_section = ""
    if conflict_state.get("conflicted_files"):
        parts = []
        for fpath, content in conflict_state["conflicted_files"].items():
            truncated = content[:3000] if len(content) > 3000 else content
            parts.append(f"### {fpath}\n```\n{truncated}\n```")
        conflict_section = "\n\n".join(parts)
    else:
        conflict_section = "Conflict details were not pre-computed. Run the setup script first to see them."

    return f"""You are resolving merge conflicts between two independently-developed features
in the same codebase. Your goal is to produce a single merged version of the code
where BOTH features work correctly and ALL tests from both features pass.

## Feature 1 (existing)
{feature_md1 or "No description available. Read the patch and tests to understand the feature."}

## Feature 2 (candidate)
{feature_md2 or "No description available. Read the patch and tests to understand the feature."}

## Conflict Summary
{conflict_state.get("summary", "Unknown")}

{conflict_section}

## Pre-loaded files

All necessary files have been pre-loaded in the container:
- `/patches/patch1.patch` -- Feature 1 implementation
- `/patches/patch2.patch` -- Feature 2 implementation
- `/patches/tests1.patch` -- Feature 1 test suite
- `/patches/tests2.patch` -- Feature 2 test suite
- `/patches/setup.sh` -- Sets up git branches and creates the merge conflict
- `/usr/local/bin/verify_resolve.sh` -- Verification script

## Step 1: Run the setup script

```
bash /patches/setup.sh
```

After running this, you will be on the `feature2` branch with merge conflict markers
in the conflicted files. The base SHA is saved in `/patches/.base_sha`.

## Step 2: Resolve the conflicts

Edit the conflicted files to integrate both features correctly. Read both features'
descriptions and tests carefully to understand what each side needs. The resolution
should preserve ALL functionality from both features.

## Step 3: Verify your resolution

After resolving conflicts, you MUST run:

```
cd /workspace/repo && bash /usr/local/bin/verify_resolve.sh
```

This script:
1. Generates a resolution patch from your current state
2. Tests Feature 1's test suite against your resolution
3. Tests Feature 2's test suite against your resolution
4. Exits 0 ONLY if both pass; exits non-zero otherwise

**You MUST NOT submit until `verify_resolve.sh` exits 0.**
If it fails, read the error output, fix your code, and re-run.

## Step 4: Submit

Once `verify_resolve.sh` passes, submit your resolution:

```
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /patches/resolution.patch
```

## Unsolvable escape hatch

If after multiple genuine attempts you conclude the two features are **fundamentally
incompatible** (e.g. they require contradictory behavior in the same function and
there is no way to satisfy both), you may run:

```
bash /usr/local/bin/verify_resolve.sh --unsolvable "brief explanation of why"
```

Then submit:

```
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo "UNSOLVABLE: brief explanation of why"
```

Use this ONLY as a last resort. Most merge conflicts ARE resolvable with careful
code integration. Do not give up just because the first or second attempt failed.

## Important rules

1. Run the setup script FIRST before doing anything else.
2. After resolving, ALWAYS run `verify_resolve.sh` before submitting.
3. If `verify_resolve.sh` fails, iterate -- do not submit a broken resolution.
4. Keep your changes minimal: only modify the conflicted source files. Do not refactor unrelated code.
5. Each command runs in a fresh shell. Use `cd /workspace/repo && ...` for every command.
6. Clean up any temporary helper scripts you create (e.g. `rm resolve_*.py`) before running verify_resolve.sh.
   The resolution patch should contain ONLY changes to the original source files.
7. You MUST resolve ALL conflict markers (<<<<<<< / ======= / >>>>>>>) in ALL files,
   including docstrings and comments. verify_resolve.sh will reject files with leftover markers.
"""


# ---------------------------------------------------------------------------
# External post-MSA verification
# ---------------------------------------------------------------------------

def _verify_resolution(
    repo_name: str,
    task_id: int,
    resolution_patch: str,
    tests1: str,
    tests2: str,
    timeout: int = 600,
    backend: str = "docker",
) -> dict:
    """Verify a resolution patch by running both test suites independently.

    Returns dict with both_passed, feature1_passed, feature2_passed, and outputs.
    """
    image = get_image_name(repo_name, task_id)
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)

    try:
        _write_patch(sb, "resolution.patch", resolution_patch)
        _write_patch(sb, "tests1.patch", tests1)
        _write_patch(sb, "tests2.patch", tests2)

        base_result = sb.exec("bash", "-c", "cd /workspace/repo && git rev-parse HEAD")
        base_sha = base_result.stdout_read().strip()

        test_cmd = """
cd /workspace/repo
git checkout {base} --force 2>/dev/null
git reset --hard 2>/dev/null
git clean -fdx 2>/dev/null
bash /usr/local/bin/runner.sh {tests_name} resolution.patch
"""
        f1_result = sb.exec("bash", "-c", test_cmd.format(base=base_sha, tests_name="tests1.patch"))
        f1_output = f1_result.stdout_read() + f1_result.stderr_read()
        f1_parsed = _parse_results(f1_output)
        f1_passed = f1_result.returncode == 0 and f1_parsed["passed"] > 0
        logger.info(
            "  ext-verify f1: returncode=%d, parsed=%s, last_500=%s",
            f1_result.returncode, f1_parsed, f1_output[-500:],
        )

        f2_result = sb.exec("bash", "-c", test_cmd.format(base=base_sha, tests_name="tests2.patch"))
        f2_output = f2_result.stdout_read() + f2_result.stderr_read()
        f2_parsed = _parse_results(f2_output)
        f2_passed = f2_result.returncode == 0 and f2_parsed["passed"] > 0
        logger.info(
            "  ext-verify f2: returncode=%d, parsed=%s, last_500=%s",
            f2_result.returncode, f2_parsed, f2_output[-500:],
        )

        return {
            "both_passed": f1_passed and f2_passed,
            "feature1_passed": f1_passed,
            "feature2_passed": f2_passed,
            "feature1_output": f1_output[-2000:],
            "feature2_output": f2_output[-2000:],
        }
    except Exception as e:
        return {
            "both_passed": False,
            "feature1_passed": False,
            "feature2_passed": False,
            "error": str(e),
        }
    finally:
        sb.terminate()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resolve_conflict(
    repo_name: str,
    task_id: int,
    patch1: str,
    patch2: str,
    tests1: str,
    tests2: str,
    feature_md1: str | None = None,
    feature_md2: str | None = None,
    model_name: str = "gemini/gemini-3-flash-preview",
    cost_limit: float = 0.50,
    step_limit: int = 50,
    timeout: int = 600,
    backend: str = "docker",
) -> ResolutionResult:
    """Attempt to resolve a merge conflict between two feature patches.

    Tries automatic merge first (naive + union via ``merge_and_test``).
    If that fails, invokes the standalone mini-swe-agent to resolve the
    conflict interactively inside a Docker container.

    Parameters
    ----------
    repo_name, task_id : task identifiers
    patch1, patch2 : feature implementation patches
    tests1, tests2 : corresponding test patches
    feature_md1, feature_md2 : optional feature descriptions for LLM context
    model_name : LLM model for MSA
    cost_limit : max USD spend for MSA
    step_limit : max LLM calls for MSA
    timeout : sandbox timeout in seconds
    backend : "docker" or "modal"
    """
    result = ResolutionResult()

    # -- Step 1: Auto-merge fast path ---------------------------------------
    logger.info("Step 1: trying auto-merge (naive + union) …")
    patch1_filtered = _filter_test_files(patch1)
    patch2_filtered = _filter_test_files(patch2)

    image = get_image_name(repo_name, task_id)
    eval_backend = get_backend(backend)
    sb = eval_backend.create_sandbox(image, timeout)
    try:
        auto = merge_and_test(sb, patch1_filtered, patch2_filtered, tests1, tests2)
    except Exception as e:
        auto = {"both_passed": False, "error": str(e)}
    finally:
        sb.terminate()

    if auto.get("both_passed"):
        logger.info("Auto-merge succeeded (strategy: %s)", auto.get("merge", {}).get("strategy"))
        result.resolved = True
        result.strategy = "auto"
        result.both_passed = True
        result.feature1_passed = True
        result.feature2_passed = True
        result.resolution_patch = auto.get("merge", {}).get("diff", "")
        return result

    auto_err = auto.get("error", "no details")
    logger.info("Auto-merge failed (%s). Proceeding with MSA resolver.", auto_err)

    # -- Step 2: Extract conflict state -------------------------------------
    logger.info("Step 2: extracting conflict state …")
    conflict_state = _extract_conflict_state(
        repo_name, task_id, patch1_filtered, patch2_filtered,
        timeout=timeout, backend=backend,
    )
    if conflict_state.get("error"):
        logger.warning("Conflict extraction error: %s", conflict_state["error"])
    else:
        n_files = len(conflict_state.get("conflicted_files", {}))
        logger.info("Found %d conflicted file(s): %s", n_files, conflict_state.get("summary", ""))

    # -- Step 3: Build MSA prompt -------------------------------------------
    logger.info("Step 3: building MSA prompt …")
    prompt = _build_resolver_prompt(feature_md1, feature_md2, conflict_state)

    # -- Step 4: Invoke standalone mini-swe-agent ---------------------------
    logger.info(
        "Step 4: invoking standalone MSA (model=%s, cost_limit=$%.2f, step_limit=%d) …",
        model_name, cost_limit, step_limit,
    )

    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    defaults_path = Path(LitellmModel.__module__.replace(".", "/")).parents[1] / "config" / "default.yaml"
    try:
        import importlib.resources as ir
        defaults_path = Path(ir.files("minisweagent") / "config" / "default.yaml")
    except Exception:
        pass
    defaults = yaml.safe_load(defaults_path.read_text()) if defaults_path.exists() else {}
    agent_defaults = defaults.get("agent", {})
    model_defaults = defaults.get("model", {})
    env_defaults = defaults.get("environment", {})

    env = None
    try:
        model = LitellmModel(
            model_name=model_name,
            cost_tracking="ignore_errors",
            **{k: v for k, v in model_defaults.items()
               if k not in ("model_name", "cost_tracking", "format_error_template")},
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
        logger.info("Pre-seeding patches and scripts into container %s …", env.container_id[:12])
        _preseed_container(env, patch1, patch2, tests1, tests2)

        raw_agent = DefaultAgent(
            model, env,
            system_template=agent_defaults.get("system_template", "You are a helpful assistant."),
            instance_template="{{task}}",
            step_limit=step_limit,
            cost_limit=cost_limit,
        )
        agent = _LoggingAgent(raw_agent)

        agent_result = agent.run(task=prompt)
    except Exception as e:
        logger.error("MSA invocation failed: %s", e, exc_info=True)
        result.error = f"MSA invocation failed: {e}"
        return result
    finally:
        if env is not None:
            env.cleanup()

    result.agent_cost = agent.cost
    result.agent_steps = agent.n_calls
    logger.info(
        "MSA finished: status=%s, steps=%d, cost=$%.4f",
        agent_result.get("exit_status", "?"), agent.n_calls, agent.cost,
    )

    submission = agent_result.get("submission", "")

    # Check for UNSOLVABLE marker
    if UNSOLVABLE_MARKER in submission:
        for line in submission.split("\n"):
            if UNSOLVABLE_MARKER in line:
                reason = line.split(UNSOLVABLE_MARKER, 1)[1].strip()
                result.unsolvable_reason = reason
                break
        result.strategy = "msa_unsolvable"
        result.resolved = False
        logger.info("MSA declared pair unsolvable: %s", result.unsolvable_reason)
        return result

    # Preserve the exact patch content from git diff. Only strip leading
    # whitespace (from the submission format), never trailing -- trailing
    # blank lines may be part of diff hunk context and their removal
    # corrupts hunk line counts.
    resolution_patch = submission.lstrip()
    if resolution_patch and not resolution_patch.endswith("\n"):
        resolution_patch += "\n"
    if not resolution_patch.strip():
        logger.warning("MSA returned empty submission (status: %s)", agent_result.get("exit_status"))
        result.error = f"MSA returned empty submission (status: {agent_result.get('exit_status')})"
        result.strategy = "msa"
        return result

    logger.info("MSA produced a resolution patch (%d bytes)", len(resolution_patch))
    result.resolution_patch = resolution_patch
    result.strategy = "msa"

    # -- Save patch and trajectory to disk for inspection -------------------
    output_dir = Path("outputs") / "resolve" / f"{repo_name}_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_path = output_dir / "resolution.patch"
    patch_path.write_text(resolution_patch)
    logger.info("Saved resolution patch to %s", patch_path)

    traj_path = output_dir / "trajectory.json"
    import json
    traj_data = agent._agent.save(None)
    traj_path.write_text(json.dumps(traj_data, indent=2, default=str))
    logger.info("Saved agent trajectory to %s", traj_path)

    # -- Step 5: External verification --------------------------------------
    logger.info("Step 5: external verification of MSA's resolution …")
    verification = _verify_resolution(
        repo_name, task_id,
        resolution_patch, tests1, tests2,
        timeout=timeout, backend=backend,
    )

    result.both_passed = verification.get("both_passed", False)
    result.feature1_passed = verification.get("feature1_passed", False)
    result.feature2_passed = verification.get("feature2_passed", False)
    result.resolved = result.both_passed

    if result.resolved:
        logger.info("Resolution VERIFIED -- both test suites pass.")
    else:
        logger.info(
            "Resolution FAILED external verification (f1=%s, f2=%s).",
            result.feature1_passed, result.feature2_passed,
        )
        if verification.get("error"):
            result.error = verification["error"]

    return result
