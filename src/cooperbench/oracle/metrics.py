"""Oracle evaluation metrics — faithfulness and coordination analysis.

``compute_faithfulness`` measures how closely an agent's produced patch
matches the ground-truth solution patch that was provided to it.

Three levels of similarity (``FaithfulnessLevel``):

  exact     — identical diffs (whitespace-normalised).
  syntactic — same files modified, same hunks added/removed (line-level).
  absent    — agent produced an empty patch (gave up or timed out).
  diverged  — agent produced a non-empty patch that differs substantially.

The returned ``FaithfulnessResult`` can be serialised to JSON and written
alongside the existing ``eval.json`` as ``oracle_eval.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class FaithfulnessLevel(str, Enum):
    EXACT = "exact"
    SYNTACTIC = "syntactic"
    DIVERGED = "diverged"
    ABSENT = "absent"


@dataclass
class FeatureFaithfulness:
    feature_id: int
    level: FaithfulnessLevel
    oracle_lines: int
    agent_lines: int
    shared_lines: int
    overlap_ratio: float


@dataclass
class FaithfulnessResult:
    features: list[FeatureFaithfulness]
    overall_level: FaithfulnessLevel
    mean_overlap_ratio: float

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert enums to their string values for JSON serialisation.
        d["overall_level"] = self.overall_level.value
        for f in d["features"]:
            f["level"] = FaithfulnessLevel(f["level"]).value
        return d


def _normalise_patch(patch: str) -> list[str]:
    """Return the added/removed lines from a unified diff, whitespace-stripped."""
    lines = []
    for line in patch.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            lines.append(line[1:].strip())
    return lines


def _faithfulness_level(oracle_lines: list[str], agent_lines: list[str]) -> tuple[FaithfulnessLevel, float]:
    """Return (level, overlap_ratio) for one feature."""
    if not agent_lines:
        return FaithfulnessLevel.ABSENT, 0.0

    oracle_set = set(oracle_lines)
    agent_set = set(agent_lines)
    if not oracle_set:
        return FaithfulnessLevel.DIVERGED, 0.0

    shared = oracle_set & agent_set
    overlap = len(shared) / len(oracle_set)

    if overlap >= 0.95 and len(agent_lines) <= len(oracle_lines) * 1.1:
        return FaithfulnessLevel.EXACT, overlap
    elif overlap >= 0.5:
        return FaithfulnessLevel.SYNTACTIC, overlap
    else:
        return FaithfulnessLevel.DIVERGED, overlap


def compute_faithfulness(
    log_dir: Path,
    dataset_dir: Path,
    repo_name: str,
    task_id: int,
    features: list[int],
    setting: str,
) -> FaithfulnessResult | None:
    """Compute faithfulness between agent patches and ground-truth patches.

    Args:
        log_dir:      Path to the run log directory (where agentN.patch lives).
        dataset_dir:  Root dataset directory.
        repo_name:    Repository name (e.g., "dspy_task").
        task_id:      Task ID.
        features:     Feature IDs evaluated.
        setting:      Run setting string (oracle_coop / oracle_coop_full / oracle_solo).

    Returns:
        FaithfulnessResult, or None if oracle patches are not available.
    """
    log_dir = Path(log_dir)
    task_dir = Path(dataset_dir) / repo_name / f"task{task_id}"

    feature_results: list[FeatureFaithfulness] = []

    for fid in features:
        oracle_patch_path = task_dir / f"feature{fid}" / "feature.patch"
        if not oracle_patch_path.exists():
            return None

        oracle_patch = oracle_patch_path.read_text()
        oracle_lines = _normalise_patch(oracle_patch)

        if setting == "oracle_solo":
            agent_patch_path = log_dir / "solo.patch"
        else:
            agent_patch_path = log_dir / f"agent{fid}.patch"

        agent_patch = agent_patch_path.read_text() if agent_patch_path.exists() else ""
        agent_lines = _normalise_patch(agent_patch)

        shared = set(oracle_lines) & set(agent_lines)
        level, overlap = _faithfulness_level(oracle_lines, agent_lines)

        feature_results.append(
            FeatureFaithfulness(
                feature_id=fid,
                level=level,
                oracle_lines=len(oracle_lines),
                agent_lines=len(agent_lines),
                shared_lines=len(shared),
                overlap_ratio=round(overlap, 4),
            )
        )

    if not feature_results:
        return None

    # Roll up to overall level: worst level among features.
    level_order = [
        FaithfulnessLevel.EXACT,
        FaithfulnessLevel.SYNTACTIC,
        FaithfulnessLevel.DIVERGED,
        FaithfulnessLevel.ABSENT,
    ]
    worst_idx = max(level_order.index(f.level) for f in feature_results)
    overall_level = level_order[worst_idx]

    mean_overlap = sum(f.overlap_ratio for f in feature_results) / len(feature_results)

    return FaithfulnessResult(
        features=feature_results,
        overall_level=overall_level,
        mean_overlap_ratio=round(mean_overlap, 4),
    )
