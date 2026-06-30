"""Unit tests for cooperbench.eval.evaluate module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cooperbench.eval import evaluate
from cooperbench.eval.evaluate import _evaluate_single


class TestEvaluate:
    """Tests for evaluate function."""

    def test_evaluate_requires_name(self):
        """Test that evaluate requires run_name."""
        with pytest.raises(TypeError):
            evaluate()  # type: ignore

    def test_evaluate_handles_no_runs(self):
        """Test that evaluate handles case with no runs gracefully."""
        with patch("cooperbench.eval.evaluate.discover_runs", return_value=[]):
            # Should not raise, just do nothing
            evaluate(run_name="nonexistent-run")


class TestEvalResultSchema:
    """Tests for evaluation result schema."""

    def test_eval_json_schema(self, tmp_path):
        """Test that eval.json follows expected schema."""
        eval_result = {
            "run_name": "test-run",
            "repo": "test_repo",
            "task_id": 1,
            "features": [1, 2],
            "setting": "coop",
            "merge_status": "success",
            "test_results": {
                "feature1": {"passed": 5, "failed": 0, "total": 5},
                "feature2": {"passed": 3, "failed": 1, "total": 4},
            },
            "overall_passed": True,
            "evaluated_at": "2026-01-31T12:00:00",
        }

        eval_file = tmp_path / "eval.json"
        eval_file.write_text(json.dumps(eval_result))

        loaded = json.loads(eval_file.read_text())
        assert "merge_status" in loaded
        assert "test_results" in loaded
        assert "overall_passed" in loaded


class TestEvaluateSingleTeamRouting:
    """Tests that _evaluate_single routes team setting to test_merged_n."""

    def _make_run_info(self, tmp_path: Path, features: list[int], setting: str = "team") -> dict:
        """Create a minimal run_info with patch files for each feature."""
        log_dir = tmp_path / "logs" / "myrun" / setting / "repo_task" / "1" / "_".join(f"f{f}" for f in features)
        log_dir.mkdir(parents=True)
        for fid in features:
            (log_dir / f"agent{fid}.patch").write_text(f"patch for feature {fid}")
        return {
            "log_dir": str(log_dir),
            "setting": setting,
            "repo": "repo_task",
            "task_id": 1,
            "features": features,
        }

    def test_team_setting_calls_test_merged_n(self, tmp_path):
        """_evaluate_single with setting=team must call test_merged_n, not test_merged."""
        run_info = self._make_run_info(tmp_path, [1, 2])
        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "applied"},
            "merge": {"status": "clean", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {
                "1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            },
            "all_passed": True,
            "feature1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            "feature2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            "both_passed": True,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result) as mock_n, \
             patch("cooperbench.eval.evaluate.test_merged") as mock_2:
            result = _evaluate_single(run_info, force=True)

        mock_n.assert_called_once()
        mock_2.assert_not_called()
        assert result["setting"] == "team"
        assert result["all_passed"] is True

    def test_team_setting_2_agents_dual_writes_legacy_keys(self, tmp_path):
        """For 2-agent team runs, eval.json must include feature1/feature2/both_passed."""
        run_info = self._make_run_info(tmp_path, [1, 2])
        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "applied"},
            "merge": {"status": "clean", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {
                "1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "2": {"feature_id": 2, "passed": False, "exit_code": 1, "tests_passed": 0, "tests_failed": 1, "test_output": ""},
            },
            "all_passed": False,
            "feature1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            "feature2": {"feature_id": 2, "passed": False, "exit_code": 1, "tests_passed": 0, "tests_failed": 1, "test_output": ""},
            "both_passed": False,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result):
            result = _evaluate_single(run_info, force=True)

        assert "feature1" in result
        assert "feature2" in result
        assert "both_passed" in result
        assert result["both_passed"] is False
        assert result["all_passed"] is False

    def test_team_setting_3_agents_no_legacy_keys(self, tmp_path):
        """For 3-agent team runs, eval.json must NOT have feature1/feature2/both_passed."""
        run_info = self._make_run_info(tmp_path, [1, 2, 3])
        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "applied", "agent3": "applied"},
            "merge": {"status": "clean", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {
                "1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "3": {"feature_id": 3, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            },
            "all_passed": True,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result):
            result = _evaluate_single(run_info, force=True)

        assert result["all_passed"] is True
        assert result["setting"] == "team"
        assert "feature1" not in result
        assert "feature2" not in result
        assert "both_passed" not in result

    def test_team_feature_ids_passed_correctly(self, tmp_path):
        """test_merged_n must receive feature_ids matching the run's features list."""
        run_info = self._make_run_info(tmp_path, [2, 5, 7])
        fake_result = {
            "apply_status": {},
            "merge": {"status": "clean", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {
                "2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "5": {"feature_id": 5, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "7": {"feature_id": 7, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            },
            "all_passed": True,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result) as mock_n:
            _evaluate_single(run_info, force=True)

        call_kwargs = mock_n.call_args
        assert call_kwargs.kwargs["feature_ids"] == [2, 5, 7]

    def test_coop_setting_still_calls_test_merged(self, tmp_path):
        """Coop setting must still use test_merged (not test_merged_n)."""
        log_dir = tmp_path / "logs" / "coop_task" / "1" / "f1_f2"
        log_dir.mkdir(parents=True)
        (log_dir / "agent1.patch").write_text("p1")
        (log_dir / "agent2.patch").write_text("p2")
        run_info = {
            "log_dir": str(log_dir),
            "setting": "coop",
            "repo": "coop_task",
            "task_id": 1,
            "features": [1, 2],
        }
        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "applied"},
            "merge": {"status": "clean", "strategy": "naive", "diff": ""},
            "feature1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            "feature2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            "both_passed": True,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged", return_value=fake_result) as mock_2, \
             patch("cooperbench.eval.evaluate.test_merged_n") as mock_n:
            result = _evaluate_single(run_info, force=True)

        mock_2.assert_called_once()
        mock_n.assert_not_called()
        assert result["setting"] == "coop"

