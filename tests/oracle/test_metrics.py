"""Tests for oracle faithfulness metrics (cooperbench.oracle.metrics)."""

import pytest

from cooperbench.oracle.metrics import (
    FaithfulnessLevel,
    FaithfulnessResult,
    FeatureFaithfulness,
    _faithfulness_level,
    _normalise_patch,
    compute_faithfulness,
)


SAMPLE_PATCH = """\
diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,2 +1,4 @@
 def hello():
+    # new
     return "hello"
+NEW_CONSTANT = 42
"""


class TestNormalisePatch:
    def test_extracts_added_lines(self):
        lines = _normalise_patch(SAMPLE_PATCH)
        assert "# new" in lines
        assert "NEW_CONSTANT = 42" in lines

    def test_excludes_context_lines(self):
        lines = _normalise_patch(SAMPLE_PATCH)
        assert "def hello():" not in lines

    def test_excludes_diff_header_lines(self):
        lines = _normalise_patch(SAMPLE_PATCH)
        assert any("+++" in l for l in lines) is False
        assert any("---" in l for l in lines) is False

    def test_empty_patch_returns_empty_list(self):
        assert _normalise_patch("") == []


class TestFaithfulnessLevel:
    def test_identical_returns_exact(self):
        lines = ["a = 1", "b = 2"]
        level, overlap = _faithfulness_level(lines, lines)
        assert level == FaithfulnessLevel.EXACT
        assert overlap == 1.0

    def test_empty_agent_returns_absent(self):
        level, overlap = _faithfulness_level(["a = 1"], [])
        assert level == FaithfulnessLevel.ABSENT
        assert overlap == 0.0

    def test_low_overlap_returns_diverged(self):
        oracle = [f"line{i}" for i in range(20)]
        agent = ["completely_different_line"]
        level, overlap = _faithfulness_level(oracle, agent)
        assert level == FaithfulnessLevel.DIVERGED

    def test_high_overlap_returns_exact_or_syntactic(self):
        oracle = ["a", "b", "c", "d", "e"]
        # Missing one out of five (80% overlap) → syntactic
        agent = ["a", "b", "c", "d"]
        level, overlap = _faithfulness_level(oracle, agent)
        assert level in (FaithfulnessLevel.SYNTACTIC, FaithfulnessLevel.EXACT)
        assert overlap >= 0.8

    def test_empty_oracle_returns_diverged(self):
        level, overlap = _faithfulness_level([], ["something"])
        assert level == FaithfulnessLevel.DIVERGED


class TestComputeFaithfulness:
    def _make_dataset(self, tmp_path, patches_by_fid: dict[int, str]) -> tuple:
        """Create a minimal dataset directory and return (dataset_dir, repo_name, task_id)."""
        repo_name = "test_repo_task"
        task_id = 1
        task_dir = tmp_path / repo_name / f"task{task_id}"
        for fid, patch_text in patches_by_fid.items():
            feature_dir = task_dir / f"feature{fid}"
            feature_dir.mkdir(parents=True)
            (feature_dir / "feature.patch").write_text(patch_text)
        return tmp_path, repo_name, task_id

    def test_oracle_coop_exact_faithfulness(self, tmp_path):
        """Agent patch identical to oracle → EXACT."""
        oracle_patch = "+line1\n+line2\n"
        dataset_dir, repo, task_id = self._make_dataset(tmp_path, {1: oracle_patch, 2: oracle_patch})

        log_dir = tmp_path / "logs" / "run1" / "oracle_coop" / repo / str(task_id) / "f1_f2"
        log_dir.mkdir(parents=True)
        (log_dir / "agent1.patch").write_text(oracle_patch)
        (log_dir / "agent2.patch").write_text(oracle_patch)

        result = compute_faithfulness(
            log_dir=log_dir,
            dataset_dir=dataset_dir,
            repo_name=repo,
            task_id=task_id,
            features=[1, 2],
            setting="oracle_coop",
        )
        assert result is not None
        assert result.overall_level == FaithfulnessLevel.EXACT
        assert result.mean_overlap_ratio == 1.0

    def test_oracle_coop_absent_faithfulness(self, tmp_path):
        """Agent produced empty patch → ABSENT."""
        oracle_patch = "+line1\n+line2\n"
        dataset_dir, repo, task_id = self._make_dataset(tmp_path, {1: oracle_patch, 2: oracle_patch})

        log_dir = tmp_path / "logs" / "run2" / "oracle_coop" / repo / str(task_id) / "f1_f2"
        log_dir.mkdir(parents=True)
        (log_dir / "agent1.patch").write_text("")
        (log_dir / "agent2.patch").write_text("")

        result = compute_faithfulness(
            log_dir=log_dir,
            dataset_dir=dataset_dir,
            repo_name=repo,
            task_id=task_id,
            features=[1, 2],
            setting="oracle_coop",
        )
        assert result is not None
        assert result.overall_level == FaithfulnessLevel.ABSENT

    def test_oracle_solo_reads_solo_patch(self, tmp_path):
        """oracle_solo mode reads solo.patch, not agentN.patch."""
        oracle_patch = "+line1\n"
        dataset_dir, repo, task_id = self._make_dataset(tmp_path, {1: oracle_patch, 2: oracle_patch})

        log_dir = tmp_path / "logs" / "run3" / "oracle_solo" / repo / str(task_id) / "f1_f2"
        log_dir.mkdir(parents=True)
        (log_dir / "solo.patch").write_text(oracle_patch)

        result = compute_faithfulness(
            log_dir=log_dir,
            dataset_dir=dataset_dir,
            repo_name=repo,
            task_id=task_id,
            features=[1, 2],
            setting="oracle_solo",
        )
        assert result is not None
        assert result.overall_level == FaithfulnessLevel.EXACT

    def test_missing_oracle_patch_returns_none(self, tmp_path):
        """If the ground-truth patch file is absent, returns None gracefully."""
        repo_name = "missing_repo_task"
        task_id = 99
        task_dir = tmp_path / repo_name / f"task{task_id}" / "feature1"
        task_dir.mkdir(parents=True)
        # Intentionally do NOT create feature.patch

        log_dir = tmp_path / "logs" / "run4"
        log_dir.mkdir(parents=True)

        result = compute_faithfulness(
            log_dir=log_dir,
            dataset_dir=tmp_path,
            repo_name=repo_name,
            task_id=task_id,
            features=[1],
            setting="oracle_coop",
        )
        assert result is None

    def test_to_dict_is_json_serialisable(self, tmp_path):
        """FaithfulnessResult.to_dict() must produce JSON-safe types."""
        import json

        oracle_patch = "+x = 1\n"
        dataset_dir, repo, task_id = self._make_dataset(tmp_path, {1: oracle_patch})

        log_dir = tmp_path / "logs" / "run5"
        log_dir.mkdir(parents=True)
        (log_dir / "agent1.patch").write_text(oracle_patch)

        result = compute_faithfulness(
            log_dir=log_dir,
            dataset_dir=dataset_dir,
            repo_name=repo,
            task_id=task_id,
            features=[1],
            setting="oracle_coop",
        )
        assert result is not None
        d = result.to_dict()
        # Should not raise
        json.dumps(d)
        assert isinstance(d["overall_level"], str)
        for f in d["features"]:
            assert isinstance(f["level"], str)
