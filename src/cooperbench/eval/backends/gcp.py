"""GCP Batch backend for evaluation.

Provides two modes:
1. GCPBatchBackend - EvalBackend interface (one sandbox at a time, slow startup)
2. GCPBatchEvaluator - Batch mode (submit all evals at once, much faster for scale)

For large-scale evaluation, use GCPBatchEvaluator which submits ALL tasks as
a single Batch job with parallel task arrays. This amortizes the ~90s VM startup
across all evaluations.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from cooperbench.eval.backends.base import ExecResult, Sandbox

# =============================================================================
# Data Classes for Batch Evaluation
# =============================================================================


@dataclass
class EvalTask:
    """A single evaluation task to run in batch mode."""

    task_index: int
    repo_name: str
    task_id: int
    feature1_id: int
    feature2_id: int
    setting: str  # "solo" or "coop"
    log_dir: str  # Where to save results
    # Patches (content, not paths)
    patch1: str = ""
    patch2: str = ""  # Only for coop mode
    # Test patches (read from dataset)
    tests1_patch: str = ""
    tests2_patch: str = ""


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    task_index: int
    repo_name: str
    task_id: int
    features: list[int]
    setting: str
    feature1_passed: bool
    feature2_passed: bool
    both_passed: bool
    merge_status: str | None = None  # For coop mode
    merge_strategy: str | None = None
    error: str | None = None
    feature1_output: str = ""
    feature2_output: str = ""


# =============================================================================
# GCPBatchEvaluator - Batch mode for large-scale evaluation
# =============================================================================


class GCPBatchEvaluator:
    """Batch evaluator using GCP Batch task arrays.

    Submits ALL evaluation tasks as a single Batch job with parallel tasks.
    Each task runs the complete evaluation pipeline (apply patches, run tests).

    This is much more efficient than GCPBatchBackend for large-scale evaluation
    because it amortizes the ~90s VM startup across all tasks.

    Example:
        evaluator = GCPBatchEvaluator(project_id="my-project")

        # Collect all tasks
        tasks = [
            EvalTask(task_index=0, repo_name="llama_index_task", ...),
            EvalTask(task_index=1, repo_name="dspy_task", ...),
            ...
        ]

        # Submit and wait for results
        results = evaluator.run_batch(tasks, parallelism=50)
    """

    # The eval script that runs inside each Batch task
    EVAL_SCRIPT = """#!/bin/bash
set -e

# Task index from Batch
TASK_INDEX=$BATCH_TASK_INDEX

echo "Task $TASK_INDEX starting..."

# Use cloud-sdk container for gsutil (faster than installing SDK)
# NOTE: COS root filesystem is read-only, so we use /home/workspace which is writable
WORKSPACE_ROOT=/home/workspace
mkdir -p $WORKSPACE_ROOT
GSUTIL="docker run --rm -v $WORKSPACE_ROOT:$WORKSPACE_ROOT -v /tmp:/tmp gcr.io/google.com/cloudsdktool/cloud-sdk:slim gsutil"

# Pull cloud-sdk image first (will be cached for subsequent tasks on same VM)
docker pull gcr.io/google.com/cloudsdktool/cloud-sdk:slim

# Download manifest
$GSUTIL cp gs://$BUCKET_NAME/$MANIFEST_PATH /tmp/manifest.json

# Extract job_id from manifest path (format: {job_id}/manifest.json)
JOB_ID=$(dirname $MANIFEST_PATH)

# Extract this task's config using Python
CONFIG=$(python3 -c "
import json
import sys
with open('/tmp/manifest.json') as f:
    manifest = json.load(f)
task = manifest['tasks'][$TASK_INDEX]
print(json.dumps(task))
")

echo "Task config: $CONFIG"

# Parse config
REPO_NAME=$(echo $CONFIG | python3 -c "import json,sys; print(json.load(sys.stdin)['repo_name'])")
TASK_ID=$(echo $CONFIG | python3 -c "import json,sys; print(json.load(sys.stdin)['task_id'])")
FEATURE1_ID=$(echo $CONFIG | python3 -c "import json,sys; print(json.load(sys.stdin)['feature1_id'])")
FEATURE2_ID=$(echo $CONFIG | python3 -c "import json,sys; print(json.load(sys.stdin)['feature2_id'])")
SETTING=$(echo $CONFIG | python3 -c "import json,sys; print(json.load(sys.stdin)['setting'])")
IMAGE=$(echo $CONFIG | python3 -c "import json,sys; print(json.load(sys.stdin)['image'])")

echo "Repo: $REPO_NAME, Task: $TASK_ID, Features: $FEATURE1_ID,$FEATURE2_ID, Setting: $SETTING"

# Pull the task image (skip if already cached on this VM)
if ! docker image inspect $IMAGE &> /dev/null; then
    echo "Pulling image: $IMAGE"
    docker pull $IMAGE
else
    echo "Image already cached: $IMAGE"
fi

# Create workspace in writable path
WORKSPACE=$WORKSPACE_ROOT/eval_$TASK_INDEX
mkdir -p $WORKSPACE
cd $WORKSPACE

# Download patches from GCS (path includes job_id)
$GSUTIL cp gs://$BUCKET_NAME/$JOB_ID/tasks/$TASK_INDEX/patch1.patch $WORKSPACE/ 2>/dev/null || touch $WORKSPACE/patch1.patch
$GSUTIL cp gs://$BUCKET_NAME/$JOB_ID/tasks/$TASK_INDEX/patch2.patch $WORKSPACE/ 2>/dev/null || touch $WORKSPACE/patch2.patch
$GSUTIL cp gs://$BUCKET_NAME/$JOB_ID/tasks/$TASK_INDEX/tests1.patch $WORKSPACE/
$GSUTIL cp gs://$BUCKET_NAME/$JOB_ID/tasks/$TASK_INDEX/tests2.patch $WORKSPACE/

# Run evaluation in Docker container
echo "Running evaluation in Docker..."

# Create the eval script to run inside container
cat > $WORKSPACE/run_eval.sh << 'EVALSCRIPT'
#!/bin/bash
set -e
cd /workspace/repo

# Configure git
git config user.email "eval@cooperbench.local"
git config user.name "CooperBench Eval"

BASE_SHA=$(git rev-parse HEAD)
SETTING="$1"
RESULT_FILE="$2"

# Initialize result
echo '{"feature1_passed": false, "feature2_passed": false, "error": null}' > $RESULT_FILE

if [ "$SETTING" = "solo" ]; then
    # Solo mode: apply one patch, test both features
    if [ -s /patches/patch1.patch ]; then
        git apply /patches/patch1.patch 2>&1 || git apply --3way /patches/patch1.patch 2>&1 || true
    fi

    # Test feature 1
    git checkout --force $BASE_SHA 2>&1
    if [ -s /patches/patch1.patch ]; then
        git apply /patches/patch1.patch 2>&1 || git apply --3way /patches/patch1.patch 2>&1 || true
    fi
    bash /usr/local/bin/runner.sh tests1.patch patch1.patch > /tmp/test1.log 2>&1 && F1_PASS=true || F1_PASS=false

    # Test feature 2
    git checkout --force $BASE_SHA 2>&1
    if [ -s /patches/patch1.patch ]; then
        git apply /patches/patch1.patch 2>&1 || git apply --3way /patches/patch1.patch 2>&1 || true
    fi
    bash /usr/local/bin/runner.sh tests2.patch patch1.patch > /tmp/test2.log 2>&1 && F2_PASS=true || F2_PASS=false

else
    # Coop mode: merge patches, then test both features
    # Create agent1 branch
    git checkout -b agent1 2>&1
    if [ -s /patches/patch1.patch ]; then
        git apply /patches/patch1.patch 2>&1 || git apply --3way /patches/patch1.patch 2>&1 || true
    fi
    git add -A && git commit -m "Agent 1" --allow-empty 2>&1

    # Create agent2 branch
    git checkout $BASE_SHA 2>&1
    git checkout -b agent2 2>&1
    if [ -s /patches/patch2.patch ]; then
        git apply /patches/patch2.patch 2>&1 || git apply --3way /patches/patch2.patch 2>&1 || true
    fi
    git add -A && git commit -m "Agent 2" --allow-empty 2>&1

    # Try merge
    MERGE_STATUS="clean"
    if ! git merge agent1 --no-commit --no-ff 2>&1; then
        MERGE_STATUS="conflicts"
        git merge --abort 2>/dev/null || true
        # Try union merge
        echo "* merge=union" >> .gitattributes
        if git merge agent1 --no-commit --no-ff 2>&1; then
            MERGE_STATUS="union"
        else
            echo '{"feature1_passed": false, "feature2_passed": false, "error": "merge_failed"}' > $RESULT_FILE
            exit 0
        fi
    fi
    git commit -m "Merged" --allow-empty 2>&1
    git diff $BASE_SHA HEAD > /patches/merged.patch

    # Test feature 1
    git checkout --force $BASE_SHA 2>&1
    git apply /patches/merged.patch 2>&1 || true
    bash /usr/local/bin/runner.sh tests1.patch merged.patch > /tmp/test1.log 2>&1 && F1_PASS=true || F1_PASS=false

    # Test feature 2
    git checkout --force $BASE_SHA 2>&1
    git apply /patches/merged.patch 2>&1 || true
    bash /usr/local/bin/runner.sh tests2.patch merged.patch > /tmp/test2.log 2>&1 && F2_PASS=true || F2_PASS=false
fi

# Write result
python3 -c "
import json
import os

def read_log(path, max_len=10000):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()[:max_len]
    except Exception:
        pass
    return ''

result = {
    'feature1_passed': $([[ $F1_PASS == true ]] && echo 'True' || echo 'False'),
    'feature2_passed': $([[ $F2_PASS == true ]] && echo 'True' || echo 'False'),
    'merge_status': '${MERGE_STATUS:-null}',
    'feature1_output': read_log('/tmp/test1.log'),
    'feature2_output': read_log('/tmp/test2.log'),
    'error': None
}
with open('$RESULT_FILE', 'w') as f:
    json.dump(result, f)
"
EVALSCRIPT

chmod +x $WORKSPACE/run_eval.sh

# Run in Docker
# NOTE: CooperBench images have ENTRYPOINT set to runner.sh, so we must override it
docker run --rm \
    --entrypoint /bin/bash \
    -v $WORKSPACE/patch1.patch:/patches/patch1.patch \
    -v $WORKSPACE/patch2.patch:/patches/patch2.patch \
    -v $WORKSPACE/tests1.patch:/patches/tests1.patch \
    -v $WORKSPACE/tests2.patch:/patches/tests2.patch \
    -v $WORKSPACE/run_eval.sh:/run_eval.sh \
    -v $WORKSPACE:/output \
    $IMAGE \
    /run_eval.sh "$SETTING" /output/result.json

# Upload result to GCS (path includes job_id)
$GSUTIL cp $WORKSPACE/result.json gs://$BUCKET_NAME/$JOB_ID/results/$TASK_INDEX/result.json

echo "Task $TASK_INDEX completed"
"""

    def __init__(
        self,
        project_id: str | None = None,
        region: str = "us-central1",
        bucket_name: str | None = None,
        vm_image: str | None = None,
    ) -> None:
        """Initialize GCP Batch evaluator.

        Args:
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            region: GCP region for Batch jobs (default: us-central1)
            bucket_name: GCS bucket for patches/results (default: cooperbench-eval-{project})
            vm_image: Custom VM image with pre-pulled Docker images.
                Can be:
                - None: Use default batch-cos image (pulls Docker images at runtime)
                - "cooperbench-eval": Use image from cooperbench-eval family
                - "projects/PROJECT/global/images/IMAGE": Full image path
                Defaults to COOPERBENCH_VM_IMAGE env var if set.

        Environment variables:
            GOOGLE_CLOUD_PROJECT: Default project ID
            COOPERBENCH_VM_IMAGE: Default VM image (e.g., "cooperbench-eval")
        """
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self._project_id:
            raise ValueError("project_id required")

        self._region = region
        self._bucket_name = bucket_name or f"cooperbench-eval-{self._project_id}"
        self._vm_image = vm_image or os.environ.get("COOPERBENCH_VM_IMAGE")
        self._logger = logging.getLogger("cooperbench.eval.backends.gcp_batch")
        self._storage_client = None
        self._batch_client = None

    def _get_storage_client(self):
        if self._storage_client is None:
            from google.cloud import storage

            self._storage_client = storage.Client(project=self._project_id)
        return self._storage_client

    def _get_batch_client(self):
        if self._batch_client is None:
            from google.cloud import batch_v1

            self._batch_client = batch_v1.BatchServiceClient()
        return self._batch_client

    def _ensure_bucket(self):
        client = self._get_storage_client()
        bucket = client.bucket(self._bucket_name)
        if not bucket.exists():
            self._logger.info(f"Creating bucket {self._bucket_name}")
            bucket = client.create_bucket(self._bucket_name, location=self._region)
            bucket.add_lifecycle_delete_rule(age=7)
            bucket.patch()
        return bucket

    def run_batch(
        self,
        tasks: list[EvalTask],
        parallelism: int = 50,
        timeout: int = 1800,
        on_progress: Callable | None = None,
        group_by_image: bool = True,
    ) -> list[EvalResult]:
        """Submit all tasks as Batch job(s) and wait for results.

        Args:
            tasks: List of evaluation tasks
            parallelism: Max concurrent tasks (VMs)
            timeout: Max runtime per task in seconds
            on_progress: Optional callback(status: str, completed: int, total: int)
            group_by_image: If True, group tasks by Docker image into separate
                jobs for optimal caching (default: True)

        Returns:
            List of EvalResult for each task
        """
        from cooperbench.utils import get_image_name

        if not tasks:
            return []

        # Group tasks by image for optimal Docker layer caching
        if group_by_image:
            tasks_by_image: dict[str, list[EvalTask]] = {}
            for task in tasks:
                image = get_image_name(task.repo_name, task.task_id)
                tasks_by_image.setdefault(image, []).append(task)

            if len(tasks_by_image) > 1:
                self._logger.info(f"Grouping {len(tasks)} tasks into {len(tasks_by_image)} jobs by image")
                return self._run_multiple_jobs(tasks_by_image, parallelism, timeout, on_progress)

        # Single image or grouping disabled - run as single job
        return self._run_single_job(tasks, parallelism, timeout, on_progress)

    def _run_single_job(
        self,
        tasks: list[EvalTask],
        parallelism: int,
        timeout: int,
        on_progress: Callable | None,
    ) -> list[EvalResult]:
        """Run tasks as a single Batch job."""
        from google.cloud import batch_v1

        job_id = f"eval-batch-{uuid.uuid4().hex[:12]}"
        self._logger.info(f"Submitting batch job {job_id} with {len(tasks)} tasks")

        if on_progress:
            on_progress("submitting", 0, len(tasks))

        # Upload manifest and patches to GCS
        bucket = self._ensure_bucket()
        manifest_path = self._upload_manifest(bucket, job_id, tasks)

        # Create Batch job
        job = self._create_batch_job(
            job_id=job_id,
            task_count=len(tasks),
            parallelism=parallelism,
            timeout=timeout,
            manifest_path=manifest_path,
        )

        # Submit job
        client = self._get_batch_client()
        parent = f"projects/{self._project_id}/locations/{self._region}"
        request = batch_v1.CreateJobRequest(parent=parent, job_id=job_id, job=job)

        try:
            client.create_job(request=request)
            self._logger.info(f"Job {job_id} submitted")
        except Exception as e:
            self._logger.error(f"Failed to submit job: {e}")
            raise

        # Wait for completion
        job_name = f"{parent}/jobs/{job_id}"
        self._wait_for_job(job_name, timeout * len(tasks) // parallelism + 300, len(tasks), on_progress)

        # Collect results
        if on_progress:
            on_progress("collecting", len(tasks), len(tasks))
        results = self._collect_results(bucket, job_id, tasks)

        # Cleanup
        self._cleanup(bucket, job_id, client, job_name)

        return results

    def _run_multiple_jobs(
        self,
        tasks_by_image: dict[str, list[EvalTask]],
        parallelism: int,
        timeout: int,
        on_progress: Callable | None,
    ) -> list[EvalResult]:
        """Run tasks grouped by image as separate parallel jobs.

        This optimizes Docker layer caching by ensuring tasks with the same
        image are in the same job, so each VM only pulls one image.
        """
        import concurrent.futures

        total_tasks = sum(len(tasks) for tasks in tasks_by_image.values())
        completed = 0
        all_results: list[EvalResult] = []

        if on_progress:
            on_progress("submitting", 0, total_tasks)

        # Submit all jobs in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks_by_image)) as executor:
            futures = {}
            for image, tasks in tasks_by_image.items():
                # Each job gets parallelism proportional to its task count
                job_parallelism = min(parallelism, len(tasks))
                future = executor.submit(self._run_single_job, tasks, job_parallelism, timeout, None)
                futures[future] = (image, tasks)

            # Collect results as jobs complete
            for future in concurrent.futures.as_completed(futures):
                image, tasks = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    completed += len(tasks)
                    if on_progress:
                        on_progress("running", completed, total_tasks)
                except Exception as e:
                    self._logger.error(f"Job for {image} failed: {e}")
                    # Add error results for failed tasks
                    for task in tasks:
                        all_results.append(
                            EvalResult(
                                task_index=task.task_index,
                                repo_name=task.repo_name,
                                task_id=task.task_id,
                                features=[task.feature1_id, task.feature2_id],
                                setting=task.setting,
                                feature1_passed=False,
                                feature2_passed=False,
                                both_passed=False,
                                error=str(e),
                            )
                        )
                    completed += len(tasks)

        # Sort results by original task_index
        all_results.sort(key=lambda r: r.task_index)

        return all_results

    def _upload_manifest(self, bucket, job_id: str, tasks: list[EvalTask]) -> str:
        """Upload manifest and patches to GCS."""
        from cooperbench.utils import get_image_name

        manifest = {
            "job_id": job_id,
            "task_count": len(tasks),
            "tasks": [],
        }

        for task in tasks:
            image = get_image_name(task.repo_name, task.task_id)
            manifest["tasks"].append(
                {
                    "task_index": task.task_index,
                    "repo_name": task.repo_name,
                    "task_id": task.task_id,
                    "feature1_id": task.feature1_id,
                    "feature2_id": task.feature2_id,
                    "setting": task.setting,
                    "image": image,
                }
            )

            # Upload patches (ensure proper newline endings)
            prefix = f"{job_id}/tasks/{task.task_index}"

            def ensure_newline(content: str) -> str:
                """Ensure patch content ends with newline (required by git)."""
                if content and not content.endswith("\n"):
                    return content + "\n"
                return content

            bucket.blob(f"{prefix}/patch1.patch").upload_from_string(ensure_newline(task.patch1 or ""))
            bucket.blob(f"{prefix}/patch2.patch").upload_from_string(ensure_newline(task.patch2 or ""))
            bucket.blob(f"{prefix}/tests1.patch").upload_from_string(ensure_newline(task.tests1_patch))
            bucket.blob(f"{prefix}/tests2.patch").upload_from_string(ensure_newline(task.tests2_patch))

        # Upload manifest
        manifest_path = f"{job_id}/manifest.json"
        bucket.blob(manifest_path).upload_from_string(json.dumps(manifest, indent=2))

        return manifest_path

    def _create_batch_job(
        self,
        job_id: str,
        task_count: int,
        parallelism: int,
        timeout: int,
        manifest_path: str,
    ):
        from google.cloud import batch_v1

        job = batch_v1.Job()

        # Script runnable - runs directly on VM which has Docker pre-installed
        runnable = batch_v1.Runnable()
        runnable.script = batch_v1.Runnable.Script()
        runnable.script.text = self.EVAL_SCRIPT

        # Task spec
        task_spec = batch_v1.TaskSpec()
        task_spec.runnables = [runnable]
        task_spec.max_run_duration = f"{timeout}s"

        # Environment variables
        env = batch_v1.Environment()
        env.variables = {
            "MANIFEST_PATH": manifest_path,
            "BUCKET_NAME": self._bucket_name,
        }
        task_spec.environment = env

        # Compute resources
        resources = batch_v1.ComputeResource()
        resources.cpu_milli = 4000  # 4 vCPUs
        resources.memory_mib = 16384  # 16 GB
        task_spec.compute_resource = resources

        # Task group with parallelism
        task_group = batch_v1.TaskGroup()
        task_group.task_spec = task_spec
        task_group.task_count = task_count
        task_group.parallelism = min(parallelism, task_count)

        job.task_groups = [task_group]

        # Allocation policy - use custom VM image or default batch-cos
        allocation_policy = batch_v1.AllocationPolicy()
        allocation_policy.location = batch_v1.AllocationPolicy.LocationPolicy()
        allocation_policy.location.allowed_locations = [f"regions/{self._region}"]

        instance_policy = batch_v1.AllocationPolicy.InstancePolicy()
        boot_disk = batch_v1.AllocationPolicy.Disk()

        if self._vm_image:
            # Use custom VM image with pre-pulled Docker images
            if self._vm_image.startswith("projects/"):
                # Full image path: projects/PROJECT/global/images/IMAGE
                boot_disk.image = self._vm_image
            else:
                # Image family name: use latest from family
                boot_disk.image = f"projects/{self._project_id}/global/images/family/{self._vm_image}"
            boot_disk.size_gb = 200  # Larger disk for pre-pulled images
            self._logger.info(f"Using custom VM image: {boot_disk.image}")
        else:
            # Default: batch-cos (Container-Optimized OS)
            boot_disk.image = "batch-cos"
            boot_disk.size_gb = 100

        instance_policy.boot_disk = boot_disk

        instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
        instances.policy = instance_policy
        allocation_policy.instances = [instances]

        job.allocation_policy = allocation_policy

        # Logs
        job.logs_policy = batch_v1.LogsPolicy()
        job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

        return job

    def _wait_for_job(
        self,
        job_name: str,
        max_wait: int,
        total_tasks: int = 0,
        on_progress: Callable | None = None,
    ):
        from google.cloud import batch_v1

        client = self._get_batch_client()
        start = time.time()
        last_status = None

        while time.time() - start < max_wait:
            job = client.get_job(name=job_name)
            state = job.status.state

            self._logger.debug(f"Job state: {state.name}")

            # Count completed tasks from task groups
            completed = 0
            if job.status.task_groups:
                for tg in job.status.task_groups.values():
                    completed += tg.counts.get("SUCCEEDED", 0) + tg.counts.get("FAILED", 0)

            # Determine status string
            if state == batch_v1.JobStatus.State.QUEUED:
                status = "queued"
            elif state == batch_v1.JobStatus.State.SCHEDULED:
                status = "provisioning"
            elif state == batch_v1.JobStatus.State.RUNNING:
                status = "running"
            else:
                status = state.name.lower()

            # Report progress if callback provided and status changed
            if on_progress and (status != last_status or completed > 0):
                on_progress(status, completed, total_tasks)
                last_status = status

            if state == batch_v1.JobStatus.State.SUCCEEDED:
                self._logger.info("Job completed successfully")
                return
            elif state == batch_v1.JobStatus.State.FAILED:
                raise RuntimeError(
                    f"Job failed: {job.status.status_events[-1].description if job.status.status_events else 'unknown'}"
                )

            time.sleep(10)

        raise TimeoutError(f"Job did not complete within {max_wait}s")

    def _collect_results(self, bucket, job_id: str, tasks: list[EvalTask]) -> list[EvalResult]:
        results = []

        for task in tasks:
            result_blob = bucket.blob(f"{job_id}/results/{task.task_index}/result.json")

            try:
                if result_blob.exists():
                    data = json.loads(result_blob.download_as_text())
                    results.append(
                        EvalResult(
                            task_index=task.task_index,
                            repo_name=task.repo_name,
                            task_id=task.task_id,
                            features=[task.feature1_id, task.feature2_id],
                            setting=task.setting,
                            feature1_passed=data.get("feature1_passed", False),
                            feature2_passed=data.get("feature2_passed", False),
                            both_passed=data.get("feature1_passed", False) and data.get("feature2_passed", False),
                            merge_status=data.get("merge_status"),
                            merge_strategy=data.get("merge_strategy"),
                            error=data.get("error"),
                            feature1_output=data.get("feature1_output", ""),
                            feature2_output=data.get("feature2_output", ""),
                        )
                    )
                else:
                    results.append(
                        EvalResult(
                            task_index=task.task_index,
                            repo_name=task.repo_name,
                            task_id=task.task_id,
                            features=[task.feature1_id, task.feature2_id],
                            setting=task.setting,
                            feature1_passed=False,
                            feature2_passed=False,
                            both_passed=False,
                            error="Result not found",
                        )
                    )
            except Exception as e:
                results.append(
                    EvalResult(
                        task_index=task.task_index,
                        repo_name=task.repo_name,
                        task_id=task.task_id,
                        features=[task.feature1_id, task.feature2_id],
                        setting=task.setting,
                        feature1_passed=False,
                        feature2_passed=False,
                        both_passed=False,
                        error=str(e),
                    )
                )

        return results

    def _cleanup(self, bucket, job_id: str, client, job_name: str):
        """Clean up GCS data and delete job."""
        try:
            blobs = list(bucket.list_blobs(prefix=f"{job_id}/"))
            for blob in blobs:
                blob.delete()
        except Exception as e:
            self._logger.warning(f"Failed to cleanup GCS: {e}")

        try:
            client.delete_job(name=job_name)
        except Exception:
            pass


# =============================================================================
# GCPBatchBackend - EvalBackend interface (one sandbox at a time)
# =============================================================================


class GCPBatchExecResult:
    """Result from a GCP Batch job execution."""

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    @property
    def returncode(self) -> int:
        return self._returncode

    def stdout_read(self) -> str:
        return self._stdout

    def stderr_read(self) -> str:
        return self._stderr


class GCPBatchSandbox:
    """GCP Batch sandbox - single sandbox mode (slower, for compatibility)."""

    WRAPPER_IMAGE = "gcr.io/google.com/cloudsdktool/cloud-sdk:slim"

    def __init__(
        self,
        project_id: str,
        region: str,
        image: str,
        workdir: str,
        timeout: int,
        sandbox_id: str,
        bucket_name: str,
        logger: logging.Logger,
    ) -> None:
        self._project_id = project_id
        self._region = region
        self._image = image
        self._workdir = workdir
        self._timeout = timeout
        self._sandbox_id = sandbox_id
        self._bucket_name = bucket_name
        self._logger = logger
        self._job_counter = 0
        self._terminated = False
        self._batch_client = None
        self._storage_client = None

    def _get_batch_client(self):
        if self._batch_client is None:
            from google.cloud import batch_v1

            self._batch_client = batch_v1.BatchServiceClient()
        return self._batch_client

    def _get_storage_client(self):
        if self._storage_client is None:
            from google.cloud import storage

            self._storage_client = storage.Client(project=self._project_id)
        return self._storage_client

    def exec(self, *args: str) -> ExecResult:
        """Execute command - creates a new Batch job each time (slow)."""
        if self._terminated:
            return GCPBatchExecResult(-1, "", "Sandbox terminated")

        from google.cloud import batch_v1

        self._job_counter += 1
        job_id = f"{self._sandbox_id}-exec-{self._job_counter}"

        # Escape command
        import base64

        command_json = json.dumps(list(args))
        command_b64 = base64.b64encode(command_json.encode()).decode()

        workspace_path = f"gs://{self._bucket_name}/{self._sandbox_id}/workspace"
        output_path = f"gs://{self._bucket_name}/{self._sandbox_id}/output/{self._job_counter}"

        script = f"""#!/bin/bash
set -o pipefail
COMMAND_JSON=$(echo "{command_b64}" | base64 -d)
mkdir -p /workspace && cd /workspace
gsutil -m rsync -r {workspace_path}/ /workspace/ 2>/dev/null || true
python3 -c "import json,subprocess,sys; cmd=json.loads('$COMMAND_JSON'); r=subprocess.run(cmd,capture_output=True,text=True); open('/tmp/stdout.txt','w').write(r.stdout); open('/tmp/stderr.txt','w').write(r.stderr); sys.exit(r.returncode)" || EXITCODE=$?
gsutil cp /tmp/stdout.txt {output_path}/stdout.txt
gsutil cp /tmp/stderr.txt {output_path}/stderr.txt
echo ${{EXITCODE:-0}} | gsutil cp - {output_path}/returncode.txt
gsutil -m rsync -r /workspace/ {workspace_path}/ 2>/dev/null || true
exit 0
"""

        job = batch_v1.Job()
        runnable = batch_v1.Runnable()
        runnable.container = batch_v1.Runnable.Container()
        runnable.container.image_uri = self.WRAPPER_IMAGE
        runnable.container.entrypoint = "/bin/bash"
        runnable.container.commands = ["-c", script]

        task_spec = batch_v1.TaskSpec()
        task_spec.runnables = [runnable]
        task_spec.max_run_duration = f"{self._timeout}s"

        resources = batch_v1.ComputeResource()
        resources.cpu_milli = 2000
        resources.memory_mib = 4096
        task_spec.compute_resource = resources

        task_group = batch_v1.TaskGroup()
        task_group.task_spec = task_spec
        task_group.task_count = 1
        job.task_groups = [task_group]

        allocation_policy = batch_v1.AllocationPolicy()
        allocation_policy.location = batch_v1.AllocationPolicy.LocationPolicy()
        allocation_policy.location.allowed_locations = [f"regions/{self._region}"]
        job.allocation_policy = allocation_policy

        job.logs_policy = batch_v1.LogsPolicy()
        job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

        client = self._get_batch_client()
        parent = f"projects/{self._project_id}/locations/{self._region}"
        request = batch_v1.CreateJobRequest(parent=parent, job_id=job_id, job=job)

        try:
            client.create_job(request=request)
        except Exception as e:
            return GCPBatchExecResult(-1, "", str(e))

        job_name = f"{parent}/jobs/{job_id}"
        return self._wait_for_job(job_name)

    def _wait_for_job(self, job_name: str) -> GCPBatchExecResult:
        from google.cloud import batch_v1

        client = self._get_batch_client()
        start = time.time()

        while time.time() - start < self._timeout:
            job = client.get_job(name=job_name)
            state = job.status.state

            if state == batch_v1.JobStatus.State.SUCCEEDED:
                return self._get_output()
            elif state in (batch_v1.JobStatus.State.FAILED, batch_v1.JobStatus.State.DELETION_IN_PROGRESS):
                result = self._get_output()
                if not result.stdout_read() and not result.stderr_read():
                    return GCPBatchExecResult(-1, "", f"Job failed: {state.name}")
                return result

            time.sleep(5)

        return GCPBatchExecResult(-1, "", f"Timeout after {self._timeout}s")

    def _get_output(self) -> GCPBatchExecResult:
        client = self._get_storage_client()
        bucket = client.bucket(self._bucket_name)
        prefix = f"{self._sandbox_id}/output/{self._job_counter}"

        time.sleep(1)

        try:
            stdout = (
                bucket.blob(f"{prefix}/stdout.txt").download_as_text()
                if bucket.blob(f"{prefix}/stdout.txt").exists()
                else ""
            )
        except Exception:
            stdout = ""

        try:
            stderr = (
                bucket.blob(f"{prefix}/stderr.txt").download_as_text()
                if bucket.blob(f"{prefix}/stderr.txt").exists()
                else ""
            )
        except Exception:
            stderr = ""

        try:
            rc_blob = bucket.blob(f"{prefix}/returncode.txt")
            returncode = int(rc_blob.download_as_text().strip()) if rc_blob.exists() else 0
        except Exception:
            returncode = 0

        return GCPBatchExecResult(returncode, stdout, stderr)

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True

        try:
            client = self._get_storage_client()
            bucket = client.bucket(self._bucket_name)
            for blob in bucket.list_blobs(prefix=f"{self._sandbox_id}/"):
                blob.delete()
        except Exception:
            pass

        try:
            client = self._get_batch_client()
            parent = f"projects/{self._project_id}/locations/{self._region}"
            for i in range(1, self._job_counter + 1):
                try:
                    client.delete_job(name=f"{parent}/jobs/{self._sandbox_id}-exec-{i}")
                except Exception:
                    pass
        except Exception:
            pass


class GCPBatchBackend:
    """GCP Batch backend - EvalBackend interface.

    Note: This creates a new Batch job for each exec() call, which is slow
    (~90s startup per call). For large-scale evaluation, use GCPBatchEvaluator
    instead, which submits all tasks at once.
    """

    def __init__(
        self,
        project_id: str | None = None,
        region: str = "us-central1",
        bucket_name: str | None = None,
    ) -> None:
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self._project_id:
            raise ValueError("project_id required")

        self._region = region
        self._bucket_name = bucket_name or f"cooperbench-eval-{self._project_id}"
        self._logger = logging.getLogger("cooperbench.eval.backends.gcp")
        self._storage_client = None

    def _get_storage_client(self):
        if self._storage_client is None:
            from google.cloud import storage

            self._storage_client = storage.Client(project=self._project_id)
        return self._storage_client

    def _ensure_bucket(self):
        client = self._get_storage_client()
        bucket = client.bucket(self._bucket_name)
        if not bucket.exists():
            bucket = client.create_bucket(self._bucket_name, location=self._region)
            bucket.add_lifecycle_delete_rule(age=7)
            bucket.patch()

    def create_sandbox(
        self,
        image: str,
        timeout: int = 600,
        workdir: str = "/workspace",
    ) -> Sandbox:
        self._ensure_bucket()
        sandbox_id = f"sandbox-{uuid.uuid4().hex[:12]}"
        self._logger.info(f"Creating sandbox {sandbox_id}")

        sandbox = GCPBatchSandbox(
            project_id=self._project_id,
            region=self._region,
            image=image,
            workdir=workdir,
            timeout=timeout,
            sandbox_id=sandbox_id,
            bucket_name=self._bucket_name,
            logger=self._logger,
        )

        sandbox.exec("mkdir", "-p", "/patches")
        return sandbox
