"""Unit tests for cooperbench.eval.runs module."""

import json
import os

from cooperbench.eval.runs import _discover_runs_in_dir, discover_runs


class TestDiscoverRuns:
    """Tests for discover_runs function."""

    def test_nonexistent_directory(self, tmp_path):
        """Test discovering runs from nonexistent directory returns empty."""
        os.chdir(tmp_path)
        runs = discover_runs(run_name="nonexistent")
        assert runs == []

    def test_empty_run_directory(self, tmp_path):
        """Test discovering runs from empty run directory returns empty."""
        os.chdir(tmp_path)
        log_dir = tmp_path / "logs" / "my-run"
        log_dir.mkdir(parents=True)

        runs = discover_runs(run_name="my-run")
        assert runs == []

    def test_coop_structure(self, tmp_path):
        """Test discovering runs with coop directory structure."""
        os.chdir(tmp_path)
        run_dir = tmp_path / "logs" / "test-run" / "coop" / "llama_index_task" / "17244" / "f1_f2"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps({"setting": "coop", "status": "done"}))

        runs = discover_runs(run_name="test-run")
        assert len(runs) == 1
        assert runs[0]["repo"] == "llama_index_task"
        assert runs[0]["task_id"] == 17244
        assert runs[0]["features"] == [1, 2]
        assert runs[0]["setting"] == "coop"

    def test_solo_structure(self, tmp_path):
        """Test discovering runs with solo directory structure."""
        os.chdir(tmp_path)
        run_dir = tmp_path / "logs" / "test-run" / "solo" / "dspy_task" / "123" / "f3_f4"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps({"setting": "solo", "status": "done"}))

        runs = discover_runs(run_name="test-run")
        assert len(runs) == 1
        assert runs[0]["repo"] == "dspy_task"
        assert runs[0]["task_id"] == 123
        assert runs[0]["features"] == [3, 4]
        assert runs[0]["setting"] == "solo"

    def test_multiple_runs(self, tmp_path):
        """Test discovering multiple runs."""
        os.chdir(tmp_path)
        base = tmp_path / "logs" / "multi-run" / "coop"

        for repo, task_id, features in [
            ("repo1_task", 1, "f1_f2"),
            ("repo1_task", 1, "f2_f3"),
            ("repo2_task", 2, "f1_f2"),
        ]:
            run_dir = base / repo / str(task_id) / features
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(json.dumps({"setting": "coop"}))

        runs = discover_runs(run_name="multi-run")
        assert len(runs) == 3

    def test_filter_by_repo(self, tmp_path):
        """Test filtering runs by repository name."""
        os.chdir(tmp_path)
        base = tmp_path / "logs" / "run" / "coop"

        for repo in ["repo1_task", "repo2_task"]:
            run_dir = base / repo / "1" / "f1_f2"
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(json.dumps({"setting": "coop"}))

        runs = discover_runs(run_name="run", repo_filter="repo1_task")
        assert len(runs) == 1
        assert runs[0]["repo"] == "repo1_task"

    def test_filter_by_task_id(self, tmp_path):
        """Test filtering runs by task ID."""
        os.chdir(tmp_path)
        base = tmp_path / "logs" / "run" / "coop" / "repo_task"

        for task_id in ["100", "200"]:
            run_dir = base / task_id / "f1_f2"
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(json.dumps({"setting": "coop"}))

        runs = discover_runs(run_name="run", task_filter=100)
        assert len(runs) == 1
        assert runs[0]["task_id"] == 100

    def test_filter_by_features(self, tmp_path):
        """Test filtering runs by specific feature pair."""
        os.chdir(tmp_path)
        base = tmp_path / "logs" / "run" / "coop" / "repo_task" / "1"

        for features in ["f1_f2", "f1_f3", "f2_f3"]:
            run_dir = base / features
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(json.dumps({"setting": "coop"}))

        runs = discover_runs(run_name="run", features_filter=[1, 3])
        assert len(runs) == 1
        assert set(runs[0]["features"]) == {1, 3}

    def test_skips_non_task_directories(self, tmp_path):
        """Test that directories not ending with _task are skipped."""
        os.chdir(tmp_path)
        base = tmp_path / "logs" / "run" / "coop"

        # Valid repo dir (ends with _task)
        valid = base / "valid_task" / "1" / "f1_f2"
        valid.mkdir(parents=True)
        (valid / "result.json").write_text(json.dumps({"setting": "coop"}))

        # Invalid - doesn't end with _task
        invalid = base / "invalid_repo" / "1" / "f1_f2"
        invalid.mkdir(parents=True)
        (invalid / "result.json").write_text(json.dumps({"setting": "coop"}))

        runs = discover_runs(run_name="run")
        assert len(runs) == 1
        assert runs[0]["repo"] == "valid_task"

    def test_skips_incomplete_runs(self, tmp_path):
        """Test that runs without result.json are skipped."""
        os.chdir(tmp_path)
        base = tmp_path / "logs" / "run" / "coop" / "repo_task" / "1"

        # Complete run with result.json
        complete = base / "f1_f2"
        complete.mkdir(parents=True)
        (complete / "result.json").write_text(json.dumps({"setting": "coop"}))

        # Incomplete run - no result.json
        incomplete = base / "f3_f4"
        incomplete.mkdir(parents=True)

        runs = discover_runs(run_name="run")
        assert len(runs) == 1
        assert runs[0]["features"] == [1, 2]


class TestDiscoverRunsInDir:
    """Tests for _discover_runs_in_dir helper function."""

    def test_basic_discovery(self, tmp_path):
        """Test basic run discovery in a directory."""
        base = tmp_path / "repo_task" / "123" / "f5_f6"
        base.mkdir(parents=True)
        (base / "result.json").write_text(json.dumps({"setting": "coop"}))

        runs = _discover_runs_in_dir(
            base_dir=tmp_path,
            setting="coop",
            subset_data=None,
            repo_filter=None,
            task_filter=None,
            features_filter=None,
        )
        assert len(runs) == 1
        assert runs[0]["repo"] == "repo_task"
        assert runs[0]["task_id"] == 123
        assert runs[0]["features"] == [5, 6]

    def test_infers_setting_from_result_json(self, tmp_path):
        """Test setting is inferred from result.json when not provided."""
        base = tmp_path / "repo_task" / "1" / "f1_f2"
        base.mkdir(parents=True)
        (base / "result.json").write_text(json.dumps({"setting": "solo"}))

        runs = _discover_runs_in_dir(
            base_dir=tmp_path,
            setting=None,  # Should infer from result.json
            subset_data=None,
            repo_filter=None,
            task_filter=None,
            features_filter=None,
        )
        assert len(runs) == 1
        assert runs[0]["setting"] == "solo"

    def test_infers_solo_from_patch_file(self, tmp_path):
        """Test setting inferred as solo when setting is explicitly null and solo.patch exists."""
        base = tmp_path / "repo_task" / "1" / "f1_f2"
        base.mkdir(parents=True)
        # Setting must be explicitly null (not missing) for solo.patch check
        (base / "result.json").write_text(json.dumps({"setting": None}))
        (base / "solo.patch").write_text("diff...")

        runs = _discover_runs_in_dir(
            base_dir=tmp_path,
            setting=None,
            subset_data=None,
            repo_filter=None,
            task_filter=None,
            features_filter=None,
        )
        assert len(runs) == 1
        assert runs[0]["setting"] == "solo"

    def test_skips_invalid_task_id(self, tmp_path):
        """Test that non-numeric task IDs are skipped."""
        # Valid numeric task ID
        valid = tmp_path / "repo_task" / "123" / "f1_f2"
        valid.mkdir(parents=True)
        (valid / "result.json").write_text(json.dumps({"setting": "coop"}))

        # Invalid - non-numeric task ID
        invalid = tmp_path / "repo_task" / "abc" / "f1_f2"
        invalid.mkdir(parents=True)
        (invalid / "result.json").write_text(json.dumps({"setting": "coop"}))

        runs = _discover_runs_in_dir(
            base_dir=tmp_path,
            setting="coop",
            subset_data=None,
            repo_filter=None,
            task_filter=None,
            features_filter=None,
        )
        assert len(runs) == 1
        assert runs[0]["task_id"] == 123

    def test_skips_invalid_feature_format(self, tmp_path):
        """Test that invalid feature directory names are skipped."""
        # Valid feature format
        valid = tmp_path / "repo_task" / "1" / "f1_f2"
        valid.mkdir(parents=True)
        (valid / "result.json").write_text(json.dumps({"setting": "coop"}))

        # Invalid - bad feature format
        invalid = tmp_path / "repo_task" / "1" / "invalid"
        invalid.mkdir(parents=True)
        (invalid / "result.json").write_text(json.dumps({"setting": "coop"}))

        # Invalid - only one feature (need at least 2)
        single = tmp_path / "repo_task" / "1" / "f1"
        single.mkdir(parents=True)
        (single / "result.json").write_text(json.dumps({"setting": "coop"}))

        runs = _discover_runs_in_dir(
            base_dir=tmp_path,
            setting="coop",
            subset_data=None,
            repo_filter=None,
            task_filter=None,
            features_filter=None,
        )
        assert len(runs) == 1
        assert runs[0]["features"] == [1, 2]

    def test_skips_files_not_directories(self, tmp_path):
        """Test that files (not directories) are skipped at each level."""
        # Valid directory structure
        valid = tmp_path / "repo_task" / "1" / "f1_f2"
        valid.mkdir(parents=True)
        (valid / "result.json").write_text(json.dumps({"setting": "coop"}))

        # File that looks like a repo
        (tmp_path / "fake_task").write_text("not a directory")

        runs = _discover_runs_in_dir(
            base_dir=tmp_path,
            setting="coop",
            subset_data=None,
            repo_filter=None,
            task_filter=None,
            features_filter=None,
        )
        assert len(runs) == 1
