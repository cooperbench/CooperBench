"""Controller agent for orchestrating feature generation across a task.

The controller is a litellm tool-calling LLM that decides whether to
decompose an existing feature, expand with a new one, assess quality, or
stop.  It operates per-task with full visibility into the feature
inventory, entanglement graph, and action history.

Hard guardrails (enforced in code, not prompt):
  - Max features per task
  - Max total cost
  - Max consecutive failures (actions producing zero accepted features)

Usage::

    python -m cooperbench.generation.controller \\
        --task dspy_task/task8394 \\
        --model gemini/gemini-3-flash-preview

    # Run on all tasks in a repo
    python -m cooperbench.generation.controller \\
        --repo dspy_task

From Python::

    from cooperbench.generation.controller import run_controller, ControllerConfig
    state = run_controller("dataset/dspy_task/task8394", ControllerConfig())
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import litellm

from cooperbench.generation.validator import _get_existing_feature_ids
from cooperbench.generation.verify import check_conflicts, check_solvability
from cooperbench.utils import get_image_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureInfo:
    """Metadata about one feature in the task."""

    feature_id: int = 0
    patch_lines: int = 0
    test_lines: int = 0
    files_modified: list[str] = field(default_factory=list)
    description: str = ""
    provenance: str = "seed"  # "seed" | "expanded" | "decomposed_from:{id}"


@dataclass
class EntanglementEdge:
    """Conflict/solvability status for one feature pair."""

    f1: int = 0
    f2: int = 0
    has_conflict: bool = False
    is_solvable: bool = False
    solvability_checked: bool = False  # distinguishes "unchecked" from "not solvable"
    resolution_strategy: str | None = None  # "auto" | "msa" | None
    resolution_cost: float = 0.0


@dataclass
class ActionRecord:
    """Log entry for one controller action."""

    action: str = ""             # "expand" | "decompose" | "assess_quality"
    timestamp: str = ""
    cost: float = 0.0
    succeeded: bool = False
    accepted_features: list[int] = field(default_factory=list)
    new_edges: list[dict] = field(default_factory=list)
    hard_edges: int = 0      # MSA-resolved pairs (real datapoints)
    trivial_edges: int = 0   # auto-solvable pairs (not datapoints)
    failure_reason: str | None = None
    failure_detail: str | None = None
    hints_given: str | None = None
    attempted_sub_features: int | None = None
    accepted_sub_features: int | None = None


@dataclass
class TaskState:
    """Full observable state of a task for the controller."""

    task_dir: str = ""
    repo_name: str = ""
    task_id: int = 0
    features: list[FeatureInfo] = field(default_factory=list)
    entanglement: list[EntanglementEdge] = field(default_factory=list)
    history: list[ActionRecord] = field(default_factory=list)
    total_cost: float = 0.0
    consecutive_failures: int = 0


@dataclass
class ControllerConfig:
    """Tunable knobs for the controller."""

    target_features: int = 6
    max_features: int = 8
    max_cost: float = 5.0
    max_consecutive_failures: int = 3
    model_name: str = "gemini/gemini-3-flash-preview"
    expand_cost_limit: float = 0.50
    expand_step_limit: int = 50
    decompose_cost_limit: float = 0.50
    decompose_step_limit: int = 50
    resolve_cost_limit: float = 0.50
    resolve_step_limit: int = 50
    timeout: int = 600
    backend: str = "docker"


# ---------------------------------------------------------------------------
# State computation from disk
# ---------------------------------------------------------------------------

def _files_from_patch(patch: str) -> list[str]:
    """Extract file paths modified in a unified diff."""
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                # "diff --git a/foo.py b/foo.py" -> "foo.py"
                files.append(parts[3].removeprefix("b/"))
    return files


def _load_feature_info(task_dir: Path, fid: int) -> FeatureInfo:
    """Build FeatureInfo for one feature from its dataset directory."""
    fdir = task_dir / f"feature{fid}"
    info = FeatureInfo(feature_id=fid)

    patch_path = fdir / "feature.patch"
    if patch_path.exists():
        patch = patch_path.read_text()
        info.patch_lines = len(patch.splitlines())
        info.files_modified = _files_from_patch(patch)

    tests_path = fdir / "tests.patch"
    if tests_path.exists():
        info.test_lines = len(tests_path.read_text().splitlines())

    md_path = fdir / "feature.md"
    if md_path.exists():
        text = md_path.read_text().strip()
        info.description = text[:300] + ("…" if len(text) > 300 else "")

    # Provenance detection: check controller_log.json if it exists
    info.provenance = "seed"
    return info


def _compute_task_state(task_dir: Path) -> TaskState:
    """Scan a task directory and build the full TaskState."""
    task_dir = Path(task_dir)
    parts = task_dir.parts
    repo_name = parts[-2] if len(parts) >= 2 else parts[-1]
    task_id_str = parts[-1].replace("task", "")
    try:
        task_id = int(task_id_str)
    except ValueError:
        task_id = 0

    state = TaskState(
        task_dir=str(task_dir),
        repo_name=repo_name,
        task_id=task_id,
    )

    feature_ids = _get_existing_feature_ids(task_dir)
    for fid in feature_ids:
        state.features.append(_load_feature_info(task_dir, fid))

    # Load persisted state (entanglement, history, provenance) if it exists
    log_path = task_dir / "controller_log.json"
    if log_path.exists():
        try:
            saved = json.loads(log_path.read_text())
            # Restore entanglement edges
            for e in saved.get("entanglement", []):
                state.entanglement.append(EntanglementEdge(**e))
            # Restore history
            for h in saved.get("history", []):
                state.history.append(ActionRecord(**h))
            state.total_cost = saved.get("total_cost", 0.0)
            state.consecutive_failures = saved.get("consecutive_failures", 0)
            # Restore provenance from saved features
            saved_features = {f["feature_id"]: f for f in saved.get("features", [])}
            for fi in state.features:
                if fi.feature_id in saved_features:
                    fi.provenance = saved_features[fi.feature_id].get("provenance", "seed")
        except Exception:
            logger.warning("Could not load controller_log.json", exc_info=True)

    return state


# ---------------------------------------------------------------------------
# Entanglement computation
# ---------------------------------------------------------------------------

def _known_pair_ids(state: TaskState) -> set[tuple[int, int]]:
    """Return set of (f1, f2) pairs already in the entanglement graph."""
    return {(e.f1, e.f2) for e in state.entanglement}


def _bootstrap_conflicts(state: TaskState, config: ControllerConfig) -> None:
    """Compute conflict-only entanglement for all unchecked pairs.

    Runs on cold start (no prior controller_log.json) to give the controller
    an initial conflict graph.  Skips solvability checks to keep cost low.
    """
    known = _known_pair_ids(state)
    fids = [f.feature_id for f in state.features]
    pairs = [
        (f1, f2) for i, f1 in enumerate(fids) for f2 in fids[i + 1:]
        if (f1, f2) not in known
    ]
    if not pairs:
        return

    logger.info("Cold start: computing conflict status for %d pair(s) …", len(pairs))
    task_dir = Path(state.task_dir)

    for f1, f2 in pairs:
        edge = EntanglementEdge(f1=f1, f2=f2)
        patch2_path = task_dir / f"feature{f2}" / "feature.patch"
        if not patch2_path.exists():
            state.entanglement.append(edge)
            continue
        try:
            c = check_conflicts(
                state.repo_name, state.task_id,
                feature_patch=patch2_path.read_text(),
                existing_feature_ids=[f1],
                timeout=config.timeout,
                backend=config.backend,
            )
            edge.has_conflict = f1 in c.get("conflicts", [])
        except Exception as e:
            logger.warning("  Conflict check failed for (%d, %d): %s", f1, f2, e)

        state.entanglement.append(edge)
        status = "CONFLICT" if edge.has_conflict else "clean"
        logger.info("  (%d, %d): %s", f1, f2, status)

    conflicts = sum(1 for e in state.entanglement if e.has_conflict)
    logger.info("Bootstrap done: %d/%d pairs have conflicts", conflicts, len(state.entanglement))


def _compute_entanglement(
    state: TaskState,
    new_feature_ids: list[int] | None = None,
    config: ControllerConfig | None = None,
) -> list[EntanglementEdge]:
    """Compute entanglement edges for new/unchecked feature pairs.

    If *new_feature_ids* is given, only check those features against all
    others.  Otherwise, check all pairs not yet in the entanglement graph.

    Returns the list of newly computed edges (also appended to state).
    """
    cfg = config or ControllerConfig()
    known = _known_pair_ids(state)
    feature_ids = [f.feature_id for f in state.features]
    new_edges: list[EntanglementEdge] = []

    pairs_to_check: list[tuple[int, int]] = []
    if new_feature_ids:
        for nid in new_feature_ids:
            for oid in feature_ids:
                if nid == oid:
                    continue
                pair = (min(nid, oid), max(nid, oid))
                if pair not in known:
                    pairs_to_check.append(pair)
    else:
        for i, f1 in enumerate(feature_ids):
            for f2 in feature_ids[i + 1:]:
                pair = (f1, f2)
                if pair not in known:
                    pairs_to_check.append(pair)

    if not pairs_to_check:
        return new_edges

    logger.info("Computing entanglement for %d new pair(s) …", len(pairs_to_check))

    task_dir = Path(state.task_dir)
    for f1, f2 in pairs_to_check:
        edge = EntanglementEdge(f1=f1, f2=f2)

        patch1_path = task_dir / f"feature{f1}" / "feature.patch"
        patch2_path = task_dir / f"feature{f2}" / "feature.patch"
        tests1_path = task_dir / f"feature{f1}" / "tests.patch"
        tests2_path = task_dir / f"feature{f2}" / "tests.patch"

        if not patch1_path.exists() or not patch2_path.exists():
            logger.warning("Missing patch for pair (%d, %d), skipping", f1, f2)
            state.entanglement.append(edge)
            new_edges.append(edge)
            continue

        patch1 = patch1_path.read_text()

        # Check conflicts: use f2's patch as candidate vs [f1]
        try:
            c = check_conflicts(
                state.repo_name, state.task_id,
                feature_patch=patch2_path.read_text(),
                existing_feature_ids=[f1],
                timeout=cfg.timeout,
                backend=cfg.backend,
            )
            edge.has_conflict = f1 in c.get("conflicts", [])
        except Exception as e:
            logger.warning("Conflict check failed for (%d, %d): %s", f1, f2, e)

        if edge.has_conflict and tests1_path.exists() and tests2_path.exists():
            edge.solvability_checked = True
            try:
                s = check_solvability(
                    state.repo_name, state.task_id,
                    candidate_patch=patch2_path.read_text(),
                    candidate_tests=tests2_path.read_text(),
                    conflicting_feature_id=f1,
                    timeout=cfg.timeout,
                    backend=cfg.backend,
                    resolve=True,
                    model_name=cfg.model_name,
                    cost_limit=cfg.resolve_cost_limit,
                    step_limit=cfg.resolve_step_limit,
                )
                edge.is_solvable = s.get("solvable", False)
                res = s.get("resolution", {})
                if res:
                    edge.resolution_strategy = res.get("strategy")
                    edge.resolution_cost = res.get("agent_cost", 0.0)
                elif s.get("both_passed"):
                    edge.resolution_strategy = "auto"
            except Exception as e:
                logger.warning("Solvability check failed for (%d, %d): %s", f1, f2, e)

        state.entanglement.append(edge)
        new_edges.append(edge)
        logger.info(
            "  Pair (%d, %d): conflict=%s, solvable=%s",
            f1, f2, edge.has_conflict, edge.is_solvable,
        )

    return new_edges


# ---------------------------------------------------------------------------
# LLM tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": (
                "Generate one new feature for the task. The feature will go "
                "through 3-check validation (tests pass, conflicts exist, "
                "jointly solvable). Returns structured result with success/failure "
                "details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hints": {
                        "type": "string",
                        "description": (
                            "Optional strategic guidance appended to the "
                            "generation agent's prompt, e.g. 'Target functions "
                            "in src/cache.py that features 1 and 3 both modify.'"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decompose",
            "description": (
                "Decompose an existing feature into 2-5 smaller sub-features. "
                "Each sub-feature goes through 3-check validation independently. "
                "Use when a feature is large (many files/lines) and contains "
                "multiple logical changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_id": {
                        "type": "integer",
                        "description": "The feature ID to decompose (e.g. 1 for feature1).",
                    },
                    "hints": {
                        "type": "string",
                        "description": "Optional strategic guidance for the decomposition agent.",
                    },
                },
                "required": ["feature_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_quality",
            "description": (
                "Evaluate the coordination richness of the current feature pairs. "
                "An LLM reviews gold patches, resolution patches, and feature "
                "descriptions to judge whether pairs are trivially or richly "
                "coordinating. Use periodically to decide whether to continue."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": (
                "Stop the controller loop. Use when the feature set has "
                "sufficient quality and quantity, or when further expansion "
                "is unlikely to improve the dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you're stopping.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Quality assessment (separate LLM call)
# ---------------------------------------------------------------------------

_QUALITY_SYSTEM = """\
You are a benchmark quality evaluator for CooperBench, a benchmark that \
tests how well two AI agents can cooperate to implement overlapping software \
features.

In CooperBench, two agents each receive ONLY their own feature description \
(feature.md).  They work in separate containers and may optionally \
communicate via messaging.  At evaluation time, their patches are merged \
with git.  Success requires BOTH test suites to pass on the merged result.

A pair encourages RICH COORDINATION when:
- Features modify overlapping functions, APIs, or data structures (not just \
adjacent lines or imports)
- The resolution patch shows non-trivial semantic changes (not just picking \
one side of a conflict)
- Features are functionally distinct (different capabilities, not variants)
- Agents would need to understand each other's intent to produce compatible code

A pair is TRIVIALLY COORDINATING when:
- Conflicts are only in imports, config, or formatting
- Union merge would likely produce passing tests automatically
- The features are nearly identical or affect completely disjoint logic \
despite conflicting on superficial lines

For each conflicting pair (both verified and unverified), assess based on \
the patches whether the pair would likely encourage rich or trivial \
coordination.  If solvability is not yet checked, still assess based on \
the patch content.

Respond with JSON:
{
  "pair_assessments": [
    {"f1": <id>, "f2": <id>, "verdict": "rich"|"trivial", "justification": "<1 sentence>"}
  ],
  "overall": "<1-2 sentence summary>",
  "suggestion": "<what to do next: expand, stop, or target specific areas>"
}
"""


def _assess_quality(
    state: TaskState,
    model_name: str = "gemini/gemini-3-flash-preview",
) -> dict:
    """Run quality assessment on the current feature set.

    Returns parsed JSON dict from the LLM, or an error dict.
    """
    task_dir = Path(state.task_dir)

    # Build context for the quality assessor
    sections: list[str] = []

    # Feature descriptions and patch summaries
    for fi in state.features:
        fdir = task_dir / f"feature{fi.feature_id}"
        patch_path = fdir / "feature.patch"
        patch_preview = ""
        if patch_path.exists():
            lines = patch_path.read_text().splitlines()
            if len(lines) > 200:
                patch_preview = "\n".join(lines[:200]) + f"\n… ({len(lines)} lines total)"
            else:
                patch_preview = "\n".join(lines)

        sections.append(
            f"### Feature {fi.feature_id} ({fi.provenance})\n"
            f"**Description**: {fi.description}\n"
            f"**Files**: {', '.join(fi.files_modified)}\n"
            f"**Patch** ({fi.patch_lines} lines):\n```\n{patch_preview}\n```"
        )

    # Entanglement edges
    hard_pairs = [e for e in state.entanglement
                  if e.has_conflict and e.is_solvable
                  and e.resolution_strategy not in (None, "auto")]
    trivial_solv = [e for e in state.entanglement
                    if e.has_conflict and e.is_solvable
                    and e.resolution_strategy in (None, "auto")]
    conflict_unchecked = [e for e in state.entanglement
                          if e.has_conflict and not e.solvability_checked]
    all_conflict = [e for e in state.entanglement if e.has_conflict]

    pair_lines = []
    for e in hard_pairs:
        res_patch_preview = ""
        res_dir = task_dir / f"feature{e.f2}" / "resolution_patches"
        res_path = res_dir / f"vs_feature{e.f1}.patch"
        if not res_path.exists():
            res_dir = task_dir / f"feature{e.f1}" / "resolution_patches"
            res_path = res_dir / f"vs_feature{e.f2}.patch"
        if res_path.exists():
            rlines = res_path.read_text().splitlines()
            if len(rlines) > 100:
                res_patch_preview = "\n".join(rlines[:100]) + f"\n… ({len(rlines)} lines)"
            else:
                res_patch_preview = "\n".join(rlines)

        pair_lines.append(
            f"**Pair (f{e.f1}, f{e.f2})**: strategy={e.resolution_strategy}, "
            f"cost=${e.resolution_cost:.2f}\n"
            f"Resolution patch:\n```\n{res_patch_preview}\n```"
        )

    unchecked_lines = []
    for e in conflict_unchecked:
        unchecked_lines.append(f"- f{e.f1} <-> f{e.f2}: conflict (solvability not yet verified)")

    trivial_lines = []
    for e in trivial_solv:
        trivial_lines.append(
            f"- f{e.f1} <-> f{e.f2}: auto-solvable (union merge passes)"
        )

    user_content = (
        "## Features\n\n"
        + "\n\n".join(sections)
        + "\n\n## Conflicting Pairs\n\n"
        + f"Total conflicting pairs: {len(all_conflict)}\n\n"
        + "### Hard datapoints (MSA-resolved, these count)\n\n"
        + ("\n\n".join(pair_lines) if pair_lines else "(none yet)")
        + "\n\n### Trivially solvable (auto-merge, NOT datapoints)\n\n"
        + ("\n".join(trivial_lines) if trivial_lines else "(none)")
        + "\n\n### Conflict detected, solvability unchecked\n\n"
        + ("\n".join(unchecked_lines) if unchecked_lines else "(all pairs checked)")
    )

    try:
        resp = litellm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": _QUALITY_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(text)
    except Exception as e:
        logger.warning("Quality assessment failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# System prompt and state message
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a controller agent for CooperBench dataset generation.  Your job is \
to build a diverse set of validated feature pairs for a single task that \
encourage rich cooperation between AI agents.

## Background

In CooperBench, two agents each receive ONLY their own feature description \
(feature.md).  They implement it in separate containers, optionally \
communicate, then their patches are merged with git.  Success requires both \
test suites to pass.  The harder the merge conflicts are to resolve \
(requiring semantic understanding, not just mechanical resolution), the more \
the pair tests real cooperation.

## What counts as a datapoint

A feature pair is a valid datapoint ONLY when:
1. Both features' tests pass independently
2. Their patches produce git merge conflicts (naive merge fails)
3. The conflicts require an MSA (mini-swe-agent) to resolve -- i.e., \
a simple union merge does NOT produce passing tests

Pairs where union merge automatically passes both tests are "trivially \
solvable" and are NOT counted as datapoints.  The entanglement graph will \
show these as separate categories.  A feature that only produces trivial \
conflicts is not a failure (the feature is still kept), but it means the \
conflicts are too superficial -- the features likely touch different hunks \
in the same files rather than genuinely overlapping logic.

## Your tools

- **expand(hints?)** -- Generate one new feature.  It goes through 3-check \
validation (tests pass, conflicts with >= 1 existing feature, jointly \
solvable).  If validation fails the feature is discarded and you get \
feedback about WHY it failed.
- **decompose(feature_id, hints?)** -- Split a large feature into 2-5 \
smaller sub-features.  Each goes through the same 3-check validation.  \
Use when a feature has many files/lines and contains multiple logical \
changes.
- **assess_quality()** -- Get an independent LLM's assessment of whether \
the current feature pairs encourage rich coordination or are trivial.  \
Use periodically, not after every action.
- **stop(reason)** -- End the loop.  Use when quality and quantity are \
satisfactory, or when further attempts are unlikely to improve things.

## Guidelines (recommendations, not strict rules)

- Consider decomposing first if a feature is large (>= 3 files or \
>= 100 patch lines).
- Features that modify THE SAME functions, classes, and data structures \
create hard conflicts.  Features that merely touch adjacent lines, imports, \
or config in the same file tend to auto-merge trivially.  Aim for \
overlapping logic, not just overlapping files.
- Look at the entanglement graph.  Focus on "HARD DATAPOINTS" count, not \
total solvable pairs.  If most pairs are trivial, try generating features \
that more deeply interleave with existing code.
- If it is star-shaped (everything conflicts with one feature but not \
each other), try to broaden it.
- Look at the action history.  Actions that produced only trivial edges \
signal that the generation hints need to target deeper code overlap.  \
If the same pattern repeats, try a different approach or consider stopping.
- Use the hints parameter to guide generation agents toward specific \
functions, classes, or data structures that existing features modify.  \
This is the primary lever for producing hard conflicts.
- Use assess_quality periodically (e.g. after every 2-3 accepted features) \
to gauge whether the current pairs are substantive.
- Quality over quantity.  Do not generate features just to hit a count \
target.  If the feature set is good, stop early.

## Hard limits (enforced automatically, you cannot override)

- Max {max_features} features per task
- Max ${max_cost:.2f} total budget
- {max_consecutive_failures} consecutive failures will auto-stop the loop

## Target

Aim for roughly {target_features} features, but stop earlier if quality is \
good or further expansion keeps failing.  What matters most is the number \
of HARD datapoint pairs, not the raw feature count.
"""


def _build_system_prompt(
    config: ControllerConfig,
    cross_task_summary: str | None = None,
) -> str:
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        max_features=config.max_features,
        max_cost=config.max_cost,
        max_consecutive_failures=config.max_consecutive_failures,
        target_features=config.target_features,
    )
    if cross_task_summary:
        prompt += f"\n## Other tasks in this repo\n\n{cross_task_summary}\n"
    return prompt


def _build_state_message(state: TaskState, config: ControllerConfig) -> str:
    """Format the current TaskState as a user message for the LLM."""
    lines: list[str] = []

    lines.append(f"## Task: {state.repo_name}/task{state.task_id}")
    lines.append(f"Budget: ${state.total_cost:.2f} / ${config.max_cost:.2f} "
                 f"(${config.max_cost - state.total_cost:.2f} remaining)")
    lines.append(f"Consecutive failures: {state.consecutive_failures} / "
                 f"{config.max_consecutive_failures}")
    lines.append("")

    # Feature inventory
    lines.append(f"## Features ({len(state.features)})")
    for fi in state.features:
        lines.append(
            f"- **feature{fi.feature_id}** ({fi.provenance}): "
            f"{fi.patch_lines} patch lines, {fi.test_lines} test lines, "
            f"files: [{', '.join(fi.files_modified)}]"
        )
        if fi.description:
            desc = fi.description[:200] + ("…" if len(fi.description) > 200 else "")
            lines.append(f"  Description: {desc}")
    lines.append("")

    # Entanglement graph -- separate hard (MSA) from trivial (auto) pairs
    hard_pairs = [e for e in state.entanglement
                  if e.has_conflict and e.is_solvable
                  and e.resolution_strategy not in (None, "auto")]
    trivial_pairs = [e for e in state.entanglement
                     if e.has_conflict and e.is_solvable
                     and e.resolution_strategy in (None, "auto")]
    conflict_unchecked = [e for e in state.entanglement
                          if e.has_conflict and not e.solvability_checked]
    conflict_unsolvable = [e for e in state.entanglement
                           if e.has_conflict and e.solvability_checked and not e.is_solvable]
    clean_pairs = [e for e in state.entanglement if not e.has_conflict]

    lines.append(f"## Entanglement Graph")
    lines.append(f"HARD DATAPOINTS (conflict + MSA-resolved, THESE COUNT): {len(hard_pairs)}")
    for e in hard_pairs:
        lines.append(f"  f{e.f1} <-> f{e.f2}: conflict, MSA-resolved "
                     f"(${e.resolution_cost:.2f})")
    lines.append(f"Trivially solvable (auto-merge, NOT datapoints): {len(trivial_pairs)}")
    if trivial_pairs:
        for e in trivial_pairs:
            lines.append(f"  f{e.f1} <-> f{e.f2}: conflict but union-merge passes both tests")
    if conflict_unchecked:
        lines.append(f"Conflict, solvability NOT YET CHECKED: {len(conflict_unchecked)}")
        for e in conflict_unchecked:
            lines.append(f"  f{e.f1} <-> f{e.f2}: conflict (solvability unknown)")
    if conflict_unsolvable:
        lines.append(f"Conflict but NOT solvable: {len(conflict_unsolvable)}")
        for e in conflict_unsolvable:
            lines.append(f"  f{e.f1} <-> f{e.f2}: conflict, NOT solvable")
    if clean_pairs:
        lines.append(f"No conflict (not datapoints): {len(clean_pairs)}")
        for e in clean_pairs:
            lines.append(f"  f{e.f1} <-> f{e.f2}: clean merge")

    # Show unchecked pairs
    known = _known_pair_ids(state)
    fids = [f.feature_id for f in state.features]
    unchecked = []
    for i, f1 in enumerate(fids):
        for f2 in fids[i + 1:]:
            if (f1, f2) not in known:
                unchecked.append((f1, f2))
    if unchecked:
        lines.append(f"Unchecked pairs: {len(unchecked)} "
                     f"({', '.join(f'({a},{b})' for a, b in unchecked[:10])})")
    lines.append("")

    # Action history
    if state.history:
        lines.append("## Action History")
        for i, h in enumerate(state.history):
            status = "OK" if h.succeeded else "FAILED"
            detail = ""
            if h.succeeded:
                detail = f"accepted: {h.accepted_features}"
                if h.hard_edges or h.trivial_edges:
                    detail += f" (hard={h.hard_edges}, trivial={h.trivial_edges})"
            elif h.failure_reason:
                detail = f"reason: {h.failure_reason}"
                if h.failure_detail:
                    detail += f" -- {h.failure_detail}"
            if h.action == "decompose" and h.attempted_sub_features is not None:
                detail += f" ({h.accepted_sub_features}/{h.attempted_sub_features} sub-features)"
            hints_note = f" [hints: {h.hints_given}]" if h.hints_given else ""
            lines.append(
                f"  {i + 1}. {h.action} [{status}] cost=${h.cost:.2f} {detail}{hints_note}"
            )
        lines.append("")

    lines.append("Decide your next action: expand, decompose, assess_quality, or stop.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(
    tool_name: str,
    tool_args: dict,
    state: TaskState,
    config: ControllerConfig,
) -> tuple[ActionRecord, list[int]]:
    """Dispatch a tool call and return (action_record, new_feature_ids).

    The new_feature_ids list is used to update entanglement incrementally.
    """
    record = ActionRecord(
        action=tool_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hints_given=tool_args.get("hints"),
    )
    new_feature_ids: list[int] = []

    if tool_name == "expand":
        from cooperbench.generation.expand import expand_task

        logger.info("=== Controller: EXPAND (hints=%s) ===", tool_args.get("hints"))
        result = expand_task(
            task_dir=state.task_dir,
            model_name=config.model_name,
            cost_limit=config.expand_cost_limit,
            step_limit=config.expand_step_limit,
            timeout=config.timeout,
            backend=config.backend,
            resolve=True,
            resolve_cost_limit=config.resolve_cost_limit,
            resolve_step_limit=config.resolve_step_limit,
            hints=tool_args.get("hints"),
        )
        record.cost = result.agent_cost
        # Add resolution costs
        for fid, details in result.solvability_details.items():
            record.cost += details.get("resolution_cost", 0) or 0

        if result.success:
            record.accepted_features = [result.feature_id]
            new_feature_ids = [result.feature_id]
            # Build edge info from the expansion result
            for fid in result.solvable_pairs:
                strategy = (result.solvability_details.get(fid, {})
                            .get("resolution_strategy", "auto"))
                cost = (result.solvability_details.get(fid, {})
                        .get("resolution_cost", 0) or 0)
                edge = EntanglementEdge(
                    f1=min(result.feature_id, fid),
                    f2=max(result.feature_id, fid),
                    has_conflict=True,
                    is_solvable=True,
                    solvability_checked=True,
                    resolution_strategy=strategy,
                    resolution_cost=cost,
                )
                state.entanglement.append(edge)
                record.new_edges.append(asdict(edge))
                if strategy in (None, "auto"):
                    record.trivial_edges += 1
                else:
                    record.hard_edges += 1
            for fid in result.conflicts:
                if fid not in result.solvable_pairs:
                    edge = EntanglementEdge(
                        f1=min(result.feature_id, fid),
                        f2=max(result.feature_id, fid),
                        has_conflict=True,
                        is_solvable=False,
                        solvability_checked=True,
                    )
                    state.entanglement.append(edge)
                    record.new_edges.append(asdict(edge))
            # Determine success: need at least one hard (MSA-resolved) pair
            if record.hard_edges > 0:
                record.succeeded = True
            else:
                record.succeeded = False
                record.failure_reason = "all_trivially_solvable"
                record.failure_detail = (
                    f"Feature saved to disk with {record.trivial_edges} conflict(s), "
                    f"but ALL were auto-solvable (union merge). Zero hard datapoints. "
                    f"Conflicts are too superficial -- features likely modify "
                    f"different hunks in the same files, not overlapping logic."
                )
        else:
            record.failure_reason = result.failure_reason or "unknown"
            # Build failure detail from the result
            details = []
            if result.error:
                details.append(result.error)
            if result.conflicts:
                details.append(f"Conflicts with: {result.conflicts}")
            if result.solvable_pairs:
                details.append(f"Solvable with: {result.solvable_pairs}")
            if result.failure_reason == "ext_no_conflicts":
                # Try to give file-level info
                patch_files = _files_from_patch(result.feature_patch) if result.feature_patch else []
                existing_files = set()
                for fi in state.features:
                    existing_files.update(fi.files_modified)
                overlap = set(patch_files) & existing_files
                if patch_files:
                    details.append(
                        f"Agent modified: {patch_files}. "
                        f"Existing features touch: {sorted(existing_files)[:10]}. "
                        f"Overlap: {sorted(overlap) if overlap else 'NONE'}"
                    )
            record.failure_detail = " | ".join(details) if details else None

    elif tool_name == "decompose":
        from cooperbench.generation.decompose import decompose_feature

        feature_id = tool_args.get("feature_id", 0)
        logger.info("=== Controller: DECOMPOSE feature%d (hints=%s) ===",
                     feature_id, tool_args.get("hints"))

        result = decompose_feature(
            task_dir=state.task_dir,
            feature_id=feature_id,
            model_name=config.model_name,
            cost_limit=config.decompose_cost_limit,
            step_limit=config.decompose_step_limit,
            timeout=config.timeout,
            backend=config.backend,
            resolve=True,
            resolve_cost_limit=config.resolve_cost_limit,
            resolve_step_limit=config.resolve_step_limit,
            hints=tool_args.get("hints"),
        )
        record.cost = result.total_agent_cost
        record.attempted_sub_features = (
            len(result.plan.sub_features) if result.plan else 0
        )
        record.accepted_sub_features = len(result.accepted_feature_ids)

        if result.accepted_feature_ids:
            record.accepted_features = result.accepted_feature_ids
            new_feature_ids = result.accepted_feature_ids

            # Collect entanglement from sub-feature results
            for sf_result in result.sub_feature_results:
                if not sf_result.success or sf_result.feature_id is None:
                    continue
                fid_new = sf_result.feature_id
                for fid_old in sf_result.solvable_pairs:
                    strategy = (sf_result.solvability_details.get(fid_old, {})
                                .get("resolution_strategy", "auto"))
                    cost = (sf_result.solvability_details.get(fid_old, {})
                            .get("resolution_cost", 0) or 0)
                    edge = EntanglementEdge(
                        f1=min(fid_new, fid_old),
                        f2=max(fid_new, fid_old),
                        has_conflict=True,
                        is_solvable=True,
                        solvability_checked=True,
                        resolution_strategy=strategy,
                        resolution_cost=cost,
                    )
                    state.entanglement.append(edge)
                    record.new_edges.append(asdict(edge))
                    if strategy in (None, "auto"):
                        record.trivial_edges += 1
                    else:
                        record.hard_edges += 1
                for fid_old in sf_result.conflicts:
                    if fid_old not in sf_result.solvable_pairs:
                        edge = EntanglementEdge(
                            f1=min(fid_new, fid_old),
                            f2=max(fid_new, fid_old),
                            has_conflict=True,
                            is_solvable=False,
                            solvability_checked=True,
                        )
                        state.entanglement.append(edge)
                        record.new_edges.append(asdict(edge))
            # Determine success: need at least one hard pair
            if record.hard_edges > 0:
                record.succeeded = True
            else:
                record.succeeded = False
                record.failure_reason = "all_trivially_solvable"
                record.failure_detail = (
                    f"Sub-features saved to disk with {record.trivial_edges} "
                    f"conflict(s), but ALL were auto-solvable (union merge). "
                    f"Zero hard datapoints. Decomposed pieces likely modify "
                    f"different hunks than existing features, not overlapping logic."
                )
        else:
            record.failure_reason = result.failure_reason or "unknown"
            details = []
            if result.error:
                details.append(result.error)
            # Summarize per-sub-feature failures
            for i, sf in enumerate(result.sub_feature_results):
                if not sf.success:
                    details.append(
                        f"Sub-feature {i + 1}: {sf.failure_reason or 'unknown'}"
                    )
            record.failure_detail = " | ".join(details) if details else None

    elif tool_name == "assess_quality":
        logger.info("=== Controller: ASSESS_QUALITY ===")
        assessment = _assess_quality(state, model_name=config.model_name)
        record.succeeded = True
        record.cost = 0.01  # negligible LLM cost for assessment
        # Store the assessment in failure_detail (reuse the field for assessment text)
        record.failure_detail = json.dumps(assessment, indent=2)

    return record, new_feature_ids


def _format_result(record: ActionRecord, state: TaskState) -> str:
    """Format an ActionRecord as a tool response message for the LLM."""
    lines: list[str] = []

    if record.action == "assess_quality":
        lines.append("## Quality Assessment Result")
        if record.failure_detail:
            lines.append(record.failure_detail)
        return "\n".join(lines)

    status = "SUCCESS" if record.succeeded else "FAILED"
    lines.append(f"## {record.action.upper()} Result: {status}")
    lines.append(f"Cost: ${record.cost:.2f}")

    if record.succeeded:
        lines.append(f"Accepted features: {record.accepted_features}")
        if record.hard_edges or record.trivial_edges:
            lines.append(f"Hard datapoints (MSA-resolved): {record.hard_edges}")
            lines.append(f"Trivially solvable (auto-merge, NOT datapoints): {record.trivial_edges}")
        if record.new_edges:
            lines.append("New entanglement edges:")
            for e in record.new_edges:
                solv = "solvable" if e.get("is_solvable") else "NOT solvable"
                strat = e.get("resolution_strategy", "auto")
                label = "HARD" if strat not in (None, "auto") else "trivial"
                lines.append(f"  f{e['f1']} <-> f{e['f2']}: {solv} "
                             f"({strat}) [{label}]")
        if record.failure_detail:
            lines.append(f"Note: {record.failure_detail}")
    else:
        lines.append(f"Failure reason: {record.failure_reason}")
        if record.failure_detail:
            lines.append(f"Details: {record.failure_detail}")

    if record.attempted_sub_features is not None:
        lines.append(f"Sub-features: {record.accepted_sub_features}/"
                     f"{record.attempted_sub_features} accepted")

    # Include updated summary counts
    hard = sum(1 for e in state.entanglement
               if e.has_conflict and e.is_solvable
               and e.resolution_strategy not in (None, "auto"))
    trivial = sum(1 for e in state.entanglement
                  if e.has_conflict and e.is_solvable
                  and e.resolution_strategy in (None, "auto"))
    lines.append(f"\nCurrent state: {len(state.features)} features, "
                 f"{hard} hard datapoints, {trivial} trivial (not datapoints)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_state(state: TaskState, path: Path | None = None) -> None:
    """Save the controller state to controller_log.json."""
    if path is None:
        path = Path(state.task_dir) / "controller_log.json"

    data = {
        "task": f"{state.repo_name}/task{state.task_id}",
        "features": [asdict(f) for f in state.features],
        "entanglement": [asdict(e) for e in state.entanglement],
        "history": [asdict(h) for h in state.history],
        "total_cost": state.total_cost,
        "consecutive_failures": state.consecutive_failures,
        "final_feature_count": len(state.features),
        "hard_datapoints": sum(
            1 for e in state.entanglement
            if e.has_conflict and e.is_solvable
            and e.resolution_strategy not in (None, "auto")
        ),
        "trivial_pairs": sum(
            1 for e in state.entanglement
            if e.has_conflict and e.is_solvable
            and e.resolution_strategy in (None, "auto")
        ),
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Persisted controller state to %s", path)


# ---------------------------------------------------------------------------
# Main controller loop
# ---------------------------------------------------------------------------

def run_controller(
    task_dir: str | Path,
    config: ControllerConfig | None = None,
    cross_task_summary: str | None = None,
) -> TaskState:
    """Run the controller agent loop for one task.

    Returns the final TaskState after the loop completes.
    """
    cfg = config or ControllerConfig()
    task_dir = Path(task_dir)

    # Load / build initial state
    state = _compute_task_state(task_dir)
    logger.info(
        "Controller starting for %s/task%d: %d features, %d entanglement edges, "
        "cost=$%.2f, consecutive_failures=%d",
        state.repo_name, state.task_id, len(state.features),
        len(state.entanglement), state.total_cost, state.consecutive_failures,
    )

    # Cold start: compute conflict graph for pre-existing features
    if not state.entanglement and len(state.features) > 1:
        _bootstrap_conflicts(state, cfg)
        _persist_state(state)

    system_prompt = _build_system_prompt(cfg, cross_task_summary)
    state_msg = _build_state_message(state, cfg)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state_msg},
    ]

    stop_reason = None

    while True:
        # --- Hard guardrails (checked BEFORE LLM call) ---
        if len(state.features) >= cfg.max_features:
            stop_reason = f"max features reached ({cfg.max_features})"
            break
        if state.total_cost >= cfg.max_cost:
            stop_reason = f"budget exhausted (${state.total_cost:.2f} >= ${cfg.max_cost:.2f})"
            break
        if state.consecutive_failures >= cfg.max_consecutive_failures:
            stop_reason = (
                f"too many consecutive failures "
                f"({state.consecutive_failures} >= {cfg.max_consecutive_failures})"
            )
            break
        remaining = cfg.max_cost - state.total_cost
        min_action_cost = min(cfg.expand_cost_limit, cfg.decompose_cost_limit)
        if remaining < min_action_cost:
            stop_reason = f"insufficient budget for next action (${remaining:.2f} remaining)"
            break

        # --- LLM decides next action ---
        logger.info("Querying controller LLM for next action …")
        try:
            response = litellm.completion(
                model=cfg.model_name,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="required",
                temperature=0.3,
            )
        except Exception as e:
            logger.error("Controller LLM call failed: %s", e)
            stop_reason = f"LLM error: {e}"
            break

        choice = response.choices[0]
        assistant_msg = choice.message

        # Extract tool call
        if not assistant_msg.tool_calls:
            logger.warning("Controller LLM returned no tool call; stopping")
            stop_reason = "LLM returned no tool call"
            break

        tool_call = assistant_msg.tool_calls[0]
        tool_name = tool_call.function.name
        try:
            tool_args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            tool_args = {}

        logger.info("Controller chose: %s(%s)", tool_name, tool_args)

        # Handle stop
        if tool_name == "stop":
            stop_reason = tool_args.get("reason", "controller decided to stop")
            logger.info("Controller stopped: %s", stop_reason)
            break

        # Check budget before expensive operations
        if tool_name in ("expand", "decompose"):
            action_cost = (cfg.expand_cost_limit if tool_name == "expand"
                           else cfg.decompose_cost_limit)
            if remaining < action_cost:
                stop_reason = (
                    f"insufficient budget for {tool_name} "
                    f"(${remaining:.2f} < ${action_cost:.2f})"
                )
                break

        # --- Execute tool ---
        record, new_feature_ids = _execute_tool(tool_name, tool_args, state, cfg)

        # --- Update state ---
        state.history.append(record)
        state.total_cost += record.cost

        if tool_name in ("expand", "decompose"):
            # Always refresh features from disk if new features were created,
            # even on failure (e.g. all_trivially_solvable -- feature exists
            # on disk but didn't produce hard datapoints)
            if new_feature_ids:
                new_fids = _get_existing_feature_ids(task_dir)
                state.features = [_load_feature_info(task_dir, fid) for fid in new_fids]
                for nfid in new_feature_ids:
                    for fi in state.features:
                        if fi.feature_id == nfid:
                            if tool_name == "expand":
                                fi.provenance = "expanded"
                            elif tool_name == "decompose":
                                fi.provenance = f"decomposed_from:{tool_args.get('feature_id', '?')}"
            if record.succeeded:
                state.consecutive_failures = 0
            else:
                state.consecutive_failures += 1

        # Persist after every action
        _persist_state(state)

        # --- Feed result back to LLM ---
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_call.function.arguments or "{}",
                },
            }
        ]})
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": _format_result(record, state),
        })

        # Also append updated state as user message so LLM always has fresh context
        messages.append({
            "role": "user",
            "content": _build_state_message(state, cfg),
        })

    # Final persistence
    if stop_reason:
        state.history.append(ActionRecord(
            action="stop",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            succeeded=True,
            failure_detail=stop_reason,
        ))
    _persist_state(state)

    hard_dp = sum(1 for e in state.entanglement
                  if e.has_conflict and e.is_solvable
                  and e.resolution_strategy not in (None, "auto"))
    trivial = sum(1 for e in state.entanglement
                  if e.has_conflict and e.is_solvable
                  and e.resolution_strategy in (None, "auto"))
    logger.info(
        "=== Controller finished: %d features, %d hard datapoints, "
        "%d trivial pairs, cost=$%.2f, reason=%s ===",
        len(state.features), hard_dp, trivial, state.total_cost, stop_reason,
    )
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controller agent for CooperBench feature generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task",
        help="Single task as repo_name/taskN (e.g. dspy_task/task8394)",
    )
    group.add_argument(
        "--repo",
        help="Run on all tasks in a repo (e.g. dspy_task)",
    )

    parser.add_argument("--model", default="gemini/gemini-3-flash-preview")
    parser.add_argument("--target-features", type=int, default=6)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--expand-cost-limit", type=float, default=0.50)
    parser.add_argument("--expand-step-limit", type=int, default=50)
    parser.add_argument("--decompose-cost-limit", type=float, default=0.50)
    parser.add_argument("--decompose-step-limit", type=int, default=50)
    parser.add_argument("--resolve-cost-limit", type=float, default=0.50)
    parser.add_argument("--resolve-step-limit", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--backend", default="docker", choices=["docker", "modal"])
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Logging setup
    app_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("cooperbench", "__main__"):
        logging.getLogger(name).setLevel(app_level)
    for name in ("cooperbench.generation.expand.agent",
                 "cooperbench.generation.resolve.agent",
                 "cooperbench.generation.decompose.agent",
                 "agent", "minisweagent", "minisweagent.environment"):
        logging.getLogger(name).setLevel(app_level)
    for name in ("LiteLLM", "litellm", "urllib3", "docker", "httpcore",
                 "httpx", "openai", "google", "grpc", "litellm_model"):
        logging.getLogger(name).setLevel(logging.WARNING)

    config = ControllerConfig(
        model_name=args.model,
        target_features=args.target_features,
        max_features=args.max_features,
        max_cost=args.max_cost,
        max_consecutive_failures=args.max_failures,
        expand_cost_limit=args.expand_cost_limit,
        expand_step_limit=args.expand_step_limit,
        decompose_cost_limit=args.decompose_cost_limit,
        decompose_step_limit=args.decompose_step_limit,
        resolve_cost_limit=args.resolve_cost_limit,
        resolve_step_limit=args.resolve_step_limit,
        timeout=args.timeout,
        backend=args.backend,
    )

    if args.task:
        task_path = args.task.strip("/")
        parts = task_path.split("/")
        if len(parts) != 2 or not parts[1].startswith("task"):
            parser.error("--task must be repo_name/taskN")

        task_dir = Path("dataset") / parts[0] / parts[1]
        if not task_dir.exists():
            parser.error(f"Task directory not found: {task_dir}")

        state = run_controller(task_dir, config)
        result = {
            "task": f"{state.repo_name}/task{state.task_id}",
            "features": len(state.features),
            "hard_datapoints": sum(
                1 for e in state.entanglement
                if e.has_conflict and e.is_solvable
                and e.resolution_strategy not in (None, "auto")
            ),
            "trivial_pairs": sum(
                1 for e in state.entanglement
                if e.has_conflict and e.is_solvable
                and e.resolution_strategy in (None, "auto")
            ),
            "total_cost": state.total_cost,
            "history_length": len(state.history),
        }
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")

    elif args.repo:
        repo_dir = Path("dataset") / args.repo
        if not repo_dir.exists():
            parser.error(f"Repo directory not found: {repo_dir}")

        task_dirs = sorted(
            d for d in repo_dir.iterdir()
            if d.is_dir() and d.name.startswith("task")
        )
        if not task_dirs:
            parser.error(f"No tasks found in {repo_dir}")

        logger.info("Found %d task(s) in %s", len(task_dirs), args.repo)

        # Build cross-task summary for diversity awareness
        results = []
        for i, td in enumerate(task_dirs):
            cross_summary = None
            if i > 0:
                summary_lines = []
                for prev_state_info in results:
                    summary_lines.append(
                        f"- {prev_state_info['task']}: "
                        f"{prev_state_info['features']} features, "
                        f"{prev_state_info['hard_datapoints']} hard datapoints, "
                        f"{prev_state_info['trivial_pairs']} trivial"
                    )
                cross_summary = "\n".join(summary_lines)

            logger.info("--- Processing task %d/%d: %s ---", i + 1, len(task_dirs), td.name)
            state = run_controller(td, config, cross_task_summary=cross_summary)
            results.append({
                "task": f"{state.repo_name}/task{state.task_id}",
                "features": len(state.features),
                "hard_datapoints": sum(
                    1 for e in state.entanglement
                    if e.has_conflict and e.is_solvable
                    and e.resolution_strategy not in (None, "auto")
                ),
                "trivial_pairs": sum(
                    1 for e in state.entanglement
                    if e.has_conflict and e.is_solvable
                    and e.resolution_strategy in (None, "auto")
                ),
                "total_cost": state.total_cost,
            })

        # Save repo-level report
        report_dir = Path("logs") / "controller"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{args.repo}_report.json"
        report_path.write_text(json.dumps(results, indent=2))
        logger.info("Saved repo report to %s", report_path)

        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
