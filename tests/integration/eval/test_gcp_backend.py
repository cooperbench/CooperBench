"""Integration tests for GCP Batch backend.

These tests require GCP credentials and a configured project.
Run with: pytest tests/integration/eval/test_gcp_backend.py --run-gcp

For faster execution, run tests in parallel:
    pytest tests/integration/eval/test_gcp_backend.py --run-gcp -n auto

Setup:
    1. Enable Batch API: gcloud services enable batch.googleapis.com
    2. Enable Storage API: gcloud services enable storage.googleapis.com
    3. Authenticate: gcloud auth application-default login
    4. Set project: export GOOGLE_CLOUD_PROJECT=your-project-id

GCP Batch has two modes:
1. GCPBatchBackend - EvalBackend interface (one sandbox at a time, ~90s startup each)
2. GCPBatchEvaluator - Batch mode (submit all tasks at once, much faster for scale)

For large-scale evaluation, always use GCPBatchEvaluator.
"""

import os
from pathlib import Path

import pytest

# Mark all tests in this module as requiring GCP
pytestmark = pytest.mark.gcp


class TestGCPBatchBackend:
    """Tests for GCPBatchBackend (EvalBackend interface)."""

    def test_create_backend(self):
        """Test creating a GCP Batch backend."""
        from cooperbench.eval.backends.gcp import GCPBatchBackend

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        backend = GCPBatchBackend(project_id=project_id)
        assert backend._project_id == project_id
        assert backend._region == "us-central1"

    def test_create_backend_custom_region(self):
        """Test creating backend with custom region."""
        from cooperbench.eval.backends.gcp import GCPBatchBackend

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        backend = GCPBatchBackend(project_id=project_id, region="us-west1")
        assert backend._region == "us-west1"

    def test_create_backend_requires_project(self):
        """Test that backend requires project ID."""
        from cooperbench.eval.backends.gcp import GCPBatchBackend

        original = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        try:
            with pytest.raises(ValueError, match="project_id required"):
                GCPBatchBackend()
        finally:
            if original:
                os.environ["GOOGLE_CLOUD_PROJECT"] = original


class TestGCPBatchEvaluator:
    """Tests for GCPBatchEvaluator (batch mode)."""

    def test_create_evaluator(self):
        """Test creating a GCP Batch evaluator."""
        from cooperbench.eval.backends.gcp import GCPBatchEvaluator

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        evaluator = GCPBatchEvaluator(project_id=project_id)
        assert evaluator._project_id == project_id

    def test_create_evaluator_requires_project(self):
        """Test that evaluator requires project ID."""
        from cooperbench.eval.backends.gcp import GCPBatchEvaluator

        original = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        try:
            with pytest.raises(ValueError, match="project_id required"):
                GCPBatchEvaluator()
        finally:
            if original:
                os.environ["GOOGLE_CLOUD_PROJECT"] = original


class TestEvalTask:
    """Tests for EvalTask dataclass."""

    def test_create_eval_task(self):
        """Test creating an EvalTask."""
        from cooperbench.eval.backends.gcp import EvalTask

        task = EvalTask(
            task_index=0,
            repo_name="llama_index_task",
            task_id=17070,
            feature1_id=1,
            feature2_id=2,
            setting="solo",
            log_dir="/logs/test",
            patch1="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            tests1_patch="test patch 1",
            tests2_patch="test patch 2",
        )

        assert task.task_index == 0
        assert task.repo_name == "llama_index_task"
        assert task.setting == "solo"
        assert task.patch2 == ""  # Default for solo mode


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_create_eval_result(self):
        """Test creating an EvalResult."""
        from cooperbench.eval.backends.gcp import EvalResult

        result = EvalResult(
            task_index=0,
            repo_name="llama_index_task",
            task_id=17070,
            features=[1, 2],
            setting="solo",
            feature1_passed=True,
            feature2_passed=True,
            both_passed=True,
        )

        assert result.task_index == 0
        assert result.both_passed is True
        assert result.error is None


class TestGetBatchEvaluator:
    """Tests for get_batch_evaluator factory function."""

    def test_get_gcp_evaluator(self):
        """Test getting GCP batch evaluator via factory."""
        from cooperbench.eval.backends import get_batch_evaluator
        from cooperbench.eval.backends.gcp import GCPBatchEvaluator

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        evaluator = get_batch_evaluator("gcp")
        assert isinstance(evaluator, GCPBatchEvaluator)

    def test_invalid_evaluator_name(self):
        """Test that invalid name raises error."""
        from cooperbench.eval.backends import get_batch_evaluator

        with pytest.raises(ValueError, match="Unknown batch evaluator"):
            get_batch_evaluator("invalid")


class TestGetBackend:
    """Tests for get_backend factory function."""

    def test_get_gcp_backend(self):
        """Test getting GCP backend via factory."""
        from cooperbench.eval.backends import get_backend
        from cooperbench.eval.backends.gcp import GCPBatchBackend

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        backend = get_backend("gcp")
        assert isinstance(backend, GCPBatchBackend)

    def test_get_gcp_batch_backend(self):
        """Test getting GCP backend via 'gcp_batch' alias."""
        from cooperbench.eval.backends import get_backend
        from cooperbench.eval.backends.gcp import GCPBatchBackend

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        backend = get_backend("gcp_batch")
        assert isinstance(backend, GCPBatchBackend)


@pytest.mark.slow
class TestGCPBatchE2E:
    """End-to-end tests that actually submit jobs to GCP Batch.

    IMPORTANT: All test cases are submitted in a SINGLE batch job to
    amortize the VM startup cost (~90s). This is the optimal pattern.

    Uses REAL gold patches from the dataset to verify:
    1. GCP Batch infrastructure works
    2. Docker images are pulled correctly
    3. Task image environment is set up correctly
    4. Test runner applies patches and runs tests
    5. Gold solution actually passes tests

    Run with: pytest tests/integration/eval/test_gcp_backend.py --run-gcp -k "E2E" -v -s
    """

    @pytest.mark.timeout(900)  # 15 min timeout (includes image pull ~2-3 min)
    def test_batch_evaluator_fail_to_pass_and_pass_to_pass(self):
        """Test GCPBatchEvaluator with fail-to-pass and pass-to-pass cases.

        Standard evaluation terminology:
        - fail-to-pass: Tests fail before fix, pass after fix
        - pass-to-pass: Tests pass before and after (regression tests)

        This test runs TWO cases in ONE batch job:
        - Task 0 (fail-to-pass check): No patch → tests FAIL (proves tests catch issues)
        - Task 1 (pass-to-pass check): Gold patch → tests PASS (proves fix works)

        This proves:
        1. The test framework catches actual failures (fail-to-pass baseline)
        2. The gold solution actually fixes the tests (pass-to-pass)
        3. The entire GCP Batch pipeline works correctly

        All cases run in ONE batch job = ONE VM startup cost.
        """
        from cooperbench.eval.backends.gcp import EvalTask, GCPBatchEvaluator

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        # Check if dataset exists
        task_dir = Path("dataset/llama_index_task/task17244")
        if not task_dir.exists():
            pytest.skip("Dataset not found - run from repo root with dataset/")

        # Load gold patch and test patch for feature 1
        feature1_gold = (task_dir / "feature1/feature.patch").read_text()
        tests1 = (task_dir / "feature1/tests.patch").read_text()

        print(f"\nLoaded patches from {task_dir}")
        print(f"  feature1 gold patch: {len(feature1_gold)} chars")
        print(f"  tests1 patch: {len(tests1)} chars")

        # Two test cases in ONE batch job
        tasks = [
            # Task 0: NO code patch, just run tests → should FAIL
            EvalTask(
                task_index=0,
                repo_name="llama_index_task",
                task_id=17244,
                feature1_id=1,
                feature2_id=1,  # Same feature, only testing feature1
                setting="solo",
                log_dir="/tmp/gcp_test/no_patch",
                patch1="",  # Empty patch = no code changes
                tests1_patch=tests1,
                tests2_patch=tests1,  # Same test for both (only care about f1)
            ),
            # Task 1: WITH gold patch, run tests → should PASS
            EvalTask(
                task_index=1,
                repo_name="llama_index_task",
                task_id=17244,
                feature1_id=1,
                feature2_id=1,  # Same feature, only testing feature1
                setting="solo",
                log_dir="/tmp/gcp_test/with_gold",
                patch1=feature1_gold,  # Gold solution
                tests1_patch=tests1,
                tests2_patch=tests1,  # Same test for both
            ),
        ]

        print(f"\nSubmitting {len(tasks)} tasks in ONE batch job...")
        print("Image: akhatua/cooperbench-llama-index:task17244")
        print("Task 0: NO patch → expect FAIL")
        print("Task 1: GOLD patch → expect PASS")

        evaluator = GCPBatchEvaluator(project_id=project_id)
        results = evaluator.run_batch(tasks, parallelism=2)

        # Verify we got results for both tasks
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # Task 0: Should FAIL (no code patch)
        result0 = results[0]
        print("\nTask 0 (NO patch):")
        print(f"  feature1_passed: {result0.feature1_passed}")
        print(f"  error: {result0.error}")
        assert result0.error is None, f"Task 0 unexpected error: {result0.error}"
        assert not result0.feature1_passed, "Task 0 should FAIL without gold patch"

        # Task 1: Should PASS (with gold patch)
        result1 = results[1]
        print("\nTask 1 (GOLD patch):")
        print(f"  feature1_passed: {result1.feature1_passed}")
        print(f"  error: {result1.error}")
        assert result1.error is None, f"Task 1 unexpected error: {result1.error}"
        assert result1.feature1_passed, "Task 1 should PASS with gold patch"

        print("\n SUCCESS!")
        print("  - Task 0 (fail-to-pass baseline): correctly FAILED without patch")
        print("  - Task 1 (pass-to-pass): correctly PASSED with gold patch")
