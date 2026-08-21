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

    def test_coop_setting_3_agents_calls_test_merged_n(self, tmp_path):
        """Coop with N>2 agents must route to test_merged_n."""
        run_info = self._make_run_info(tmp_path, [1, 2, 3], setting="coop")
        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "applied", "agent3": "applied"},
            "merge": {"status": "clean", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {
                "1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "3": {"feature_id": 3, "passed": False, "exit_code": 1, "tests_passed": 0, "tests_failed": 1, "test_output": ""},
            },
            "all_passed": False,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result) as mock_n, \
             patch("cooperbench.eval.evaluate.test_merged") as mock_2:
            result = _evaluate_single(run_info, force=True)

        mock_n.assert_called_once()
        mock_2.assert_not_called()
        assert mock_n.call_args.kwargs["feature_ids"] == [1, 2, 3]
        assert result["setting"] == "coop"
        assert result["all_passed"] is False
        assert "features_result" in result
        assert "feature1" not in result
        assert "both_passed" not in result


class TestEvaluateSingleSoloRouting:
    """Tests that _evaluate_single routes solo setting to the right eval fn."""

    def _make_run_info(self, tmp_path: Path, features: list[int]) -> dict:
        log_dir = tmp_path / "logs" / "myrun" / "solo" / "repo_task" / "1" / "_".join(f"f{f}" for f in features)
        log_dir.mkdir(parents=True)
        (log_dir / "solo.patch").write_text("the solo patch")
        return {
            "log_dir": str(log_dir),
            "setting": "solo",
            "repo": "repo_task",
            "task_id": 1,
            "features": features,
        }

    def test_solo_setting_2_features_still_calls_test_solo(self, tmp_path):
        """Solo with 2 features must keep using the legacy test_solo path."""
        run_info = self._make_run_info(tmp_path, [1, 2])
        fake_result = {
            "setting": "solo",
            "patch_lines": 1,
            "feature1": {"passed": True, "test_output": ""},
            "feature2": {"passed": True, "test_output": ""},
            "both_passed": True,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_solo", return_value=fake_result) as mock_2, \
             patch("cooperbench.eval.evaluate.test_solo_n") as mock_n:
            result = _evaluate_single(run_info, force=True)

        mock_2.assert_called_once()
        mock_n.assert_not_called()
        assert result["setting"] == "solo"
        assert result["both_passed"] is True

    def test_solo_setting_3_features_calls_test_solo_n(self, tmp_path):
        """Solo with N>2 features must route to test_solo_n."""
        run_info = self._make_run_info(tmp_path, [1, 2, 3])
        fake_result = {
            "setting": "solo",
            "patch_lines": 1,
            "features": {
                "1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "3": {"feature_id": 3, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            },
            "all_passed": True,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_solo_n", return_value=fake_result) as mock_n, \
             patch("cooperbench.eval.evaluate.test_solo") as mock_2:
            result = _evaluate_single(run_info, force=True)

        mock_n.assert_called_once()
        mock_2.assert_not_called()
        assert mock_n.call_args.kwargs["feature_ids"] == [1, 2, 3]
        assert result["setting"] == "solo"
        assert result["all_passed"] is True
        assert "features_result" in result
        assert "feature1" not in result
        assert "both_passed" not in result


class TestEvaluateSingleCachePath:
    """Tests for the force=False caching path in _evaluate_single."""

    def _make_run_info(self, tmp_path: Path, features: list[int], setting: str = "team") -> dict:
        log_dir = tmp_path / "logs" / setting / "repo_task" / "1"
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

    def test_force_false_returns_cached_result(self, tmp_path):
        """With force=False and a pre-existing eval.json, the cached result is returned."""
        import json as _json
        run_info = self._make_run_info(tmp_path, [1, 2])
        cached = {"repo": "repo_task", "task_id": 1, "all_passed": True, "from_cache": True}
        (Path(run_info["log_dir"]) / "eval.json").write_text(_json.dumps(cached))

        with patch("cooperbench.eval.evaluate.test_merged_n") as mock_n, \
             patch("cooperbench.eval.evaluate.test_merged") as mock_2, \
             patch("cooperbench.eval.evaluate.test_solo") as mock_solo, \
             patch("cooperbench.eval.evaluate.test_solo_n") as mock_solo_n:
            result = _evaluate_single(run_info, force=False)

        mock_n.assert_not_called()
        mock_2.assert_not_called()
        mock_solo.assert_not_called()
        mock_solo_n.assert_not_called()
        assert result["skipped"] is True
        assert result["from_cache"] is True

    def test_force_true_reruns_even_when_cache_exists(self, tmp_path):
        """With force=True, cached eval.json is ignored and the eval reruns."""
        import json as _json
        run_info = self._make_run_info(tmp_path, [1, 2])
        cached = {"repo": "repo_task", "task_id": 1, "all_passed": True, "from_cache": True}
        (Path(run_info["log_dir"]) / "eval.json").write_text(_json.dumps(cached))

        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "applied"},
            "merge": {"status": "clean", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {"1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                         "2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""}},
            "all_passed": True, "feature1": {}, "feature2": {}, "both_passed": True, "error": None,
        }
        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result) as mock_n:
            result = _evaluate_single(run_info, force=True)

        mock_n.assert_called_once()
        assert result.get("skipped") is not True


class TestMissingPatchFlagging:
    """Tests that a missing agent patch file is flagged as missing_input, not clean."""

    def _make_run_info(self, tmp_path: Path, features: list[int], present_fids: list[int]) -> dict:
        log_dir = tmp_path / "logs" / "team" / "repo_task" / "1"
        log_dir.mkdir(parents=True)
        for fid in present_fids:
            (log_dir / f"agent{fid}.patch").write_text(f"patch for feature {fid}")
        return {
            "log_dir": str(log_dir),
            "setting": "team",
            "repo": "repo_task",
            "task_id": 1,
            "features": features,
        }

    def test_missing_patch_overrides_clean_merge_to_missing_input(self, tmp_path):
        """When agent2 has no patch file, a 'clean' merge result becomes 'missing_input'."""
        run_info = self._make_run_info(tmp_path, [1, 2, 3], present_fids=[1, 3])
        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "skipped", "agent3": "applied"},
            "merge": {"status": "clean", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {
                "1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "2": {"feature_id": 2, "passed": False, "exit_code": 0, "tests_passed": 0, "tests_failed": 0, "test_output": ""},
                "3": {"feature_id": 3, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            },
            "all_passed": False,
            "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result):
            result = _evaluate_single(run_info, force=True)

        assert result["merge"]["status"] == "missing_input"
        assert result["all_passed"] is False

    def test_all_patches_present_does_not_override(self, tmp_path):
        """When all patch files are present, a 'clean' merge status is preserved."""
        run_info = self._make_run_info(tmp_path, [1, 2], present_fids=[1, 2])
        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "applied"},
            "merge": {"status": "clean", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {
                "1": {"feature_id": 1, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
                "2": {"feature_id": 2, "passed": True, "exit_code": 0, "tests_passed": 1, "tests_failed": 0, "test_output": ""},
            },
            "all_passed": True, "feature1": {}, "feature2": {}, "both_passed": True, "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result):
            result = _evaluate_single(run_info, force=True)

        assert result["merge"]["status"] == "clean"
        assert result["all_passed"] is True

    def test_missing_patch_with_conflict_merge_not_overridden(self, tmp_path):
        """When merge already reports conflicts, missing patch doesn't change it."""
        run_info = self._make_run_info(tmp_path, [1, 2], present_fids=[1])
        fake_result = {
            "apply_status": {"agent1": "applied", "agent2": "skipped"},
            "merge": {"status": "conflicts", "strategy": "sequential-fold", "steps": [], "diff": ""},
            "features": {
                "1": {"feature_id": 1, "passed": False, "exit_code": 1, "tests_passed": 0, "tests_failed": 1, "test_output": ""},
                "2": {"feature_id": 2, "passed": False, "exit_code": 1, "tests_passed": 0, "tests_failed": 1, "test_output": ""},
            },
            "all_passed": False, "feature1": {}, "feature2": {}, "both_passed": False, "error": None,
        }

        with patch("cooperbench.eval.evaluate.test_merged_n", return_value=fake_result):
            result = _evaluate_single(run_info, force=True)

        assert result["merge"]["status"] == "conflicts"

