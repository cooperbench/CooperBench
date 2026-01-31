"""Unit tests for cooperbench.runner.tasks module."""

import json
import os

import pytest

from cooperbench.runner import discover_tasks
from cooperbench.runner.tasks import load_subset


class TestDiscoverTasks:
    """Tests for discover_tasks function."""

    def test_discover_all_tasks(self):
        """Test discovering all tasks returns non-empty list."""
        tasks = discover_tasks()
        assert len(tasks) > 0
        assert all("repo" in t and "task_id" in t and "features" in t for t in tasks)

    def test_discover_by_repo(self):
        """Test filtering tasks by repository name."""
        tasks = discover_tasks(repo_filter="llama_index_task")
        assert len(tasks) > 0
        assert all(t["repo"] == "llama_index_task" for t in tasks)

    def test_discover_by_task_id(self):
        """Test filtering tasks by specific task ID."""
        tasks = discover_tasks(repo_filter="llama_index_task", task_filter=17244)
        assert len(tasks) > 0
        assert all(t["task_id"] == 17244 for t in tasks)

    def test_discover_specific_features(self):
        """Test filtering tasks by specific feature pair."""
        tasks = discover_tasks(
            repo_filter="llama_index_task",
            task_filter=17244,
            features_filter=[1, 2],
        )
        assert len(tasks) == 1
        assert tasks[0]["features"] == [1, 2]

    def test_discover_generates_feature_pairs(self):
        """Test that discovery generates all pairwise feature combinations."""
        tasks = discover_tasks(repo_filter="llama_index_task", task_filter=17244)
        # Should have nC2 pairs for tasks with >2 features
        features_found = set()
        for t in tasks:
            features_found.add(tuple(sorted(t["features"])))
        # At least some pairs generated
        assert len(features_found) >= 1

    def test_discover_nonexistent_repo(self):
        """Test that nonexistent repository returns empty list."""
        tasks = discover_tasks(repo_filter="nonexistent_repo_xyz")
        assert tasks == []

    def test_discover_nonexistent_task_id(self):
        """Test that nonexistent task ID returns empty list."""
        tasks = discover_tasks(repo_filter="llama_index_task", task_filter=99999999)
        assert tasks == []

    def test_discover_invalid_feature_filter(self):
        """Test that invalid feature filter returns empty list."""
        tasks = discover_tasks(
            repo_filter="llama_index_task",
            task_filter=17244,
            features_filter=[999, 998],  # Features that don't exist
        )
        assert tasks == []

    def test_task_structure(self):
        """Test that discovered tasks have correct structure."""
        tasks = discover_tasks(repo_filter="llama_index_task", task_filter=17244)
        assert len(tasks) > 0

        task = tasks[0]
        assert isinstance(task["repo"], str)
        assert isinstance(task["task_id"], int)
        assert isinstance(task["features"], list)
        assert len(task["features"]) == 2
        assert all(isinstance(f, int) for f in task["features"])

    def test_features_are_sorted(self):
        """Test that feature pairs are in sorted order."""
        tasks = discover_tasks()
        for task in tasks:
            features = task["features"]
            assert features == sorted(features), f"Features not sorted: {features}"


class TestLoadSubset:
    """Tests for load_subset function."""

    def test_load_lite_subset(self):
        """Test loading the lite subset."""
        tasks = load_subset("lite")
        assert len(tasks) == 7
        assert all(isinstance(t, tuple) and len(t) == 2 for t in tasks)
        # Check some expected tasks
        assert ("pillow_task", 25) in tasks
        assert ("dspy_task", 8394) in tasks

    def test_load_nonexistent_subset_raises(self):
        """Test that loading nonexistent subset raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            load_subset("nonexistent_subset_xyz")

    def test_subset_returns_repo_task_tuples(self):
        """Test that subset returns (repo, task_id) tuples."""
        tasks = load_subset("lite")
        for repo, task_id in tasks:
            assert isinstance(repo, str)
            assert isinstance(task_id, int)
            assert repo.endswith("_task") or repo == "typst"


class TestDiscoverTasksWithSubset:
    """Tests for discover_tasks with subset filtering."""

    def test_discover_with_lite_subset(self):
        """Test discovering tasks with lite subset."""
        tasks = discover_tasks(subset="lite")
        # Lite has 100 pairs
        assert len(tasks) == 100

    def test_subset_filters_repos(self):
        """Test that subset filters to only repos in subset."""
        tasks = discover_tasks(subset="lite")
        repos = {t["repo"] for t in tasks}
        # Lite subset has 7 repos
        assert len(repos) == 7
        assert "pillow_task" in repos
        assert "dspy_task" in repos

    def test_subset_with_repo_filter(self):
        """Test combining subset with repo filter."""
        tasks = discover_tasks(subset="lite", repo_filter="pillow_task")
        assert len(tasks) > 0
        assert all(t["repo"] == "pillow_task" for t in tasks)

    def test_subset_with_task_filter(self):
        """Test combining subset with task filter."""
        tasks = discover_tasks(subset="lite", repo_filter="pillow_task", task_filter=25)
        assert len(tasks) > 0
        assert all(t["task_id"] == 25 for t in tasks)

    def test_subset_excludes_non_subset_tasks(self):
        """Test that subset excludes tasks not in subset."""
        # Get all tasks
        all_tasks = discover_tasks()
        # Get lite tasks
        lite_tasks = discover_tasks(subset="lite")

        # Lite should be smaller
        assert len(lite_tasks) < len(all_tasks)

        # Tasks in lite should be subset of all tasks
        lite_pairs = {(t["repo"], t["task_id"], tuple(t["features"])) for t in lite_tasks}
        all_pairs = {(t["repo"], t["task_id"], tuple(t["features"])) for t in all_tasks}
        assert lite_pairs.issubset(all_pairs)


class TestDiscoverTasksWithMockDataset:
    """Tests for discover_tasks with temporary dataset directory."""

    def test_skips_non_task_directories(self, tmp_path):
        """Test that directories not starting with 'task' are skipped."""
        os.chdir(tmp_path)

        # Create mock dataset
        dataset = tmp_path / "dataset" / "test_task"
        dataset.mkdir(parents=True)

        # Valid task directory
        task1 = dataset / "task1"
        task1.mkdir()
        (task1 / "feature1").mkdir()
        (task1 / "feature2").mkdir()

        # Invalid - doesn't start with 'task'
        invalid = dataset / "notask1"
        invalid.mkdir()
        (invalid / "feature1").mkdir()
        (invalid / "feature2").mkdir()

        tasks = discover_tasks(repo_filter="test_task")
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == 1

    def test_requires_minimum_two_features(self, tmp_path):
        """Test that tasks with <2 features are skipped."""
        os.chdir(tmp_path)

        dataset = tmp_path / "dataset" / "test_task"
        dataset.mkdir(parents=True)

        # Task with only 1 feature - should be skipped
        task1 = dataset / "task1"
        task1.mkdir()
        (task1 / "feature1").mkdir()

        # Task with 2 features - should be included
        task2 = dataset / "task2"
        task2.mkdir()
        (task2 / "feature1").mkdir()
        (task2 / "feature2").mkdir()

        tasks = discover_tasks(repo_filter="test_task")
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == 2

    def test_generates_all_pairs_for_multiple_features(self, tmp_path):
        """Test nC2 pair generation for tasks with >2 features."""
        os.chdir(tmp_path)

        dataset = tmp_path / "dataset" / "test_task"
        dataset.mkdir(parents=True)

        # Task with 3 features -> should generate 3 pairs: (1,2), (1,3), (2,3)
        task1 = dataset / "task1"
        task1.mkdir()
        (task1 / "feature1").mkdir()
        (task1 / "feature2").mkdir()
        (task1 / "feature3").mkdir()

        tasks = discover_tasks(repo_filter="test_task", task_filter=1)
        assert len(tasks) == 3

        feature_pairs = {tuple(t["features"]) for t in tasks}
        assert feature_pairs == {(1, 2), (1, 3), (2, 3)}

    def test_feature_filter_validates_existence(self, tmp_path):
        """Test that feature filter validates features exist in task."""
        os.chdir(tmp_path)

        dataset = tmp_path / "dataset" / "test_task"
        dataset.mkdir(parents=True)

        task1 = dataset / "task1"
        task1.mkdir()
        (task1 / "feature1").mkdir()
        (task1 / "feature2").mkdir()

        # Filter for features that exist
        tasks = discover_tasks(
            repo_filter="test_task",
            task_filter=1,
            features_filter=[1, 2],
        )
        assert len(tasks) == 1

        # Filter for features that don't exist
        tasks = discover_tasks(
            repo_filter="test_task",
            task_filter=1,
            features_filter=[1, 99],
        )
        assert len(tasks) == 0
