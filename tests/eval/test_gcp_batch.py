"""Unit tests for N-agent support in the GCP batch eval path.

These don't touch GCP — they test the EvalTask/EvalResult dataclass
normalisation and the _run_gcp_batch task-building/result-processing logic
with a mocked batch evaluator.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from cooperbench.eval.backends.gcp import EvalResult, EvalTask, GCPBatchEvaluator
from cooperbench.eval.evaluate import _run_gcp_batch


class TestEvalTaskNormalization:
    """EvalTask must accept both legacy 2-feature and N-generic construction."""

    def test_legacy_fields_populate_lists(self):
        task = EvalTask(
            task_index=0,
            repo_name="repo",
            task_id=1,
            feature1_id=1,
            feature2_id=2,
            setting="coop",
            log_dir="/logs",
            patch1="p1",
            patch2="p2",
            tests1_patch="t1",
            tests2_patch="t2",
        )
        assert task.feature_ids == [1, 2]
        assert task.patches == ["p1", "p2"]
        assert task.tests_patches == ["t1", "t2"]

    def test_legacy_solo_uploads_single_patch(self):
        task = EvalTask(
            task_index=0,
            repo_name="repo",
            task_id=1,
            feature1_id=1,
            feature2_id=2,
            setting="solo",
            log_dir="/logs",
            patch1="p1",
            tests1_patch="t1",
            tests2_patch="t2",
        )
        assert task.patches == ["p1"]

    def test_n_generic_fields_populate_legacy(self):
        task = EvalTask(
            task_index=0,
            repo_name="repo",
            task_id=1,
            setting="team",
            log_dir="/logs",
            feature_ids=[3, 5, 7],
            patches=["p3", "p5", "p7"],
            tests_patches=["t3", "t5", "t7"],
        )
        assert task.feature1_id == 3
        assert task.feature2_id == 5
        assert task.patch1 == "p3"
        assert task.patch2 == "p5"
        assert task.tests1_patch == "t3"
        assert task.tests2_patch == "t5"


class TestEvalResultNormalization:
    def test_legacy_construction_populates_lists(self):
        result = EvalResult(
            task_index=0,
            repo_name="repo",
            task_id=1,
            features=[1, 2],
            setting="coop",
            feature1_passed=True,
            feature2_passed=False,
            both_passed=False,
        )
        assert result.features_passed == [True, False]

    def test_n_generic_construction(self):
        result = EvalResult(
            task_index=0,
            repo_name="repo",
            task_id=1,
            features=[1, 2, 3],
            setting="team",
            feature1_passed=True,
            feature2_passed=True,
            both_passed=False,
            features_passed=[True, True, False],
            all_passed=False,
        )
        assert result.features_passed == [True, True, False]
        assert result.all_passed is False


class TestEvalScriptNGeneric:
    """The batch VM script must be N-generic."""

    def test_script_loops_over_n_features(self):
        script = GCPBatchEvaluator.EVAL_SCRIPT
        assert "N_FEATURES" in script
        assert "seq 1 $N_FEATURES" in script

    def test_container_script_receives_n(self):
        script = GCPBatchEvaluator.EVAL_SCRIPT
        assert '/run_eval.sh "$SETTING" /output/result.json "$N_FEATURES"' in script

    def test_container_script_writes_features_passed(self):
        script = GCPBatchEvaluator.EVAL_SCRIPT
        assert "features_passed" in script


class TestRunGcpBatchNAgents:
    """_run_gcp_batch must handle N>2 runs instead of skipping them."""

    def _make_dataset(self, tmp_path: Path, features: list[int]) -> Path:
        dataset_dir = tmp_path / "dataset"
        for fid in features:
            feature_dir = dataset_dir / "repo_task" / "task1" / f"feature{fid}"
            feature_dir.mkdir(parents=True)
            (feature_dir / "tests.patch").write_text(f"tests patch {fid}\n")
        return dataset_dir

    def _make_run(self, tmp_path: Path, features: list[int], setting: str) -> dict:
        log_dir = tmp_path / "logs" / "myrun" / setting / "repo_task" / "1" / "_".join(f"f{f}" for f in features)
        log_dir.mkdir(parents=True)
        if setting == "solo":
            (log_dir / "solo.patch").write_text("solo patch\n")
        else:
            for fid in features:
                (log_dir / f"agent{fid}.patch").write_text(f"agent patch {fid}\n")
        return {
            "repo": "repo_task",
            "task_id": 1,
            "features": features,
            "setting": setting,
            "log_dir": str(log_dir),
        }

    def test_team_3_agents_not_skipped(self, tmp_path):
        features = [1, 2, 3]
        dataset_dir = self._make_dataset(tmp_path, features)
        run_info = self._make_run(tmp_path, features, "team")

        fake_evaluator = MagicMock()

        def fake_run_batch(tasks, parallelism=50, on_progress=None):
            assert len(tasks) == 1, "3-agent team run must be submitted, not skipped"
            task = tasks[0]
            assert task.feature_ids == features
            assert len(task.patches) == 3
            assert len(task.tests_patches) == 3
            return [
                EvalResult(
                    task_index=task.task_index,
                    repo_name=task.repo_name,
                    task_id=task.task_id,
                    features=task.feature_ids,
                    setting=task.setting,
                    feature1_passed=True,
                    feature2_passed=True,
                    both_passed=True,
                    merge_status="clean",
                    features_passed=[True, True, True],
                    features_output=["out1", "out2", "out3"],
                    all_passed=True,
                )
            ]

        fake_evaluator.run_batch.side_effect = fake_run_batch

        with patch("cooperbench.eval.backends.get_batch_evaluator", return_value=fake_evaluator):
            passed, failed, errors, skipped, results = _run_gcp_batch(
                [run_info], parallelism=2, force=True, dataset_dir=dataset_dir
            )

        assert passed == 1
        assert failed == 0
        assert errors == 0

        eval_json = json.loads((Path(run_info["log_dir"]) / "eval.json").read_text())
        assert eval_json["all_passed"] is True
        assert set(eval_json["features_result"].keys()) == {"1", "2", "3"}
        assert "feature1" not in eval_json
        assert "both_passed" not in eval_json

    def test_coop_2_agents_dual_writes_legacy_keys(self, tmp_path):
        features = [1, 2]
        dataset_dir = self._make_dataset(tmp_path, features)
        run_info = self._make_run(tmp_path, features, "coop")

        fake_evaluator = MagicMock()
        fake_evaluator.run_batch.return_value = [
            EvalResult(
                task_index=0,
                repo_name="repo_task",
                task_id=1,
                features=features,
                setting="coop",
                feature1_passed=True,
                feature2_passed=False,
                both_passed=False,
                merge_status="clean",
                features_passed=[True, False],
                features_output=["out1", "out2"],
                all_passed=False,
            )
        ]

        with patch("cooperbench.eval.backends.get_batch_evaluator", return_value=fake_evaluator):
            passed, failed, errors, skipped, results = _run_gcp_batch(
                [run_info], parallelism=2, force=True, dataset_dir=dataset_dir
            )

        assert passed == 0
        assert failed == 1

        eval_json = json.loads((Path(run_info["log_dir"]) / "eval.json").read_text())
        assert eval_json["both_passed"] is False
        assert eval_json["feature1"]["passed"] is True
        assert eval_json["feature2"]["passed"] is False
        assert eval_json["all_passed"] is False

    def test_solo_3_features_submits_single_patch(self, tmp_path):
        features = [1, 2, 3]
        dataset_dir = self._make_dataset(tmp_path, features)
        run_info = self._make_run(tmp_path, features, "solo")

        fake_evaluator = MagicMock()

        def fake_run_batch(tasks, parallelism=50, on_progress=None):
            task = tasks[0]
            assert task.setting == "solo"
            assert task.feature_ids == features
            assert len(task.patches) == 1
            assert len(task.tests_patches) == 3
            return [
                EvalResult(
                    task_index=task.task_index,
                    repo_name=task.repo_name,
                    task_id=task.task_id,
                    features=task.feature_ids,
                    setting="solo",
                    feature1_passed=True,
                    feature2_passed=True,
                    both_passed=True,
                    features_passed=[True, True, True],
                    features_output=["o1", "o2", "o3"],
                    all_passed=True,
                )
            ]

        fake_evaluator.run_batch.side_effect = fake_run_batch

        with patch("cooperbench.eval.backends.get_batch_evaluator", return_value=fake_evaluator):
            passed, failed, errors, skipped, results = _run_gcp_batch(
                [run_info], parallelism=2, force=True, dataset_dir=dataset_dir
            )

        assert passed == 1
        eval_json = json.loads((Path(run_info["log_dir"]) / "eval.json").read_text())
        assert eval_json["merge"] is None
        assert eval_json["all_passed"] is True
