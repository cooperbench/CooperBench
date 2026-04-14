"""Convert SWE-smith Docker images to CooperBench format.

Two-tier conversion:

  Tier 1 (per-repo, done once):
      Pulls a SWE-smith image, clones the real GitHub repo with full git
      history into ``/workspace/repo``, injects a conda-aware runner.sh,
      commits as ``<registry>/cooperbench-<repo>:base``, and optionally
      pushes to Docker Hub.

  Tier 2 (per-task, seconds):
      Starts from the base image, checks out the PR's base commit,
      re-installs the package, and commits as
      ``<registry>/cooperbench-<repo>:task<id>``.

Usage::

    # Tier 1: create + push a base image for Flask
    python -m cooperbench.generation.convert_swesmith base \\
        --swesmith-image jyangballin/swesmith.x86_64.pallets_1776_flask.bc098406 \\
        --github-url https://github.com/pallets/flask.git \\
        --repo-name flask_task --push

    # Tier 2: create a per-task image
    python -m cooperbench.generation.convert_swesmith task \\
        --repo-name flask_task --task-id 5939 --base-commit c34d6e81

    # Batch: convert multiple repos from a JSON config
    python -m cooperbench.generation.convert_swesmith batch \\
        --config repos.json --push
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CONDA_ACTIVATE = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"


# ---------------------------------------------------------------------------
# runner.sh template (conda-aware, auto-detecting test files)
# ---------------------------------------------------------------------------

RUNNER_SH_CONDA = textwrap.dedent("""\
    #!/bin/bash
    set -e

    # Activate the conda environment used by SWE-smith images
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate testbed

    cleanup() {
        echo "Cleaning up repository..."
        cd /workspace/repo 2>/dev/null || true
        git reset --hard HEAD 2>/dev/null || true
        git clean -fdx 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    TEST_PATCH="$1"
    FEATURE_PATCH="$2"

    if [[ -z "$TEST_PATCH" ]]; then
        echo "Usage: runner.sh <test_patch> [feature_patch]"
        exit 1
    fi

    cd /workspace/repo
    git reset --hard HEAD
    git clean -fdx

    # Apply feature patch if provided
    if [[ -n "$FEATURE_PATCH" ]]; then
        echo "Applying feature patch: $FEATURE_PATCH"
        if [[ -f "/patches/$FEATURE_PATCH" ]]; then
            git apply --ignore-whitespace --ignore-space-change "/patches/$FEATURE_PATCH" \\
                || git apply --3way "/patches/$FEATURE_PATCH"
        else
            echo "Error: Feature patch not found at /patches/$FEATURE_PATCH"
            exit 1
        fi
    fi

    # Apply test patch
    echo "Applying test patch: $TEST_PATCH"
    if [[ -f "/patches/$TEST_PATCH" ]]; then
        git apply --ignore-whitespace --ignore-space-change "/patches/$TEST_PATCH" \\
            || git apply --3way "/patches/$TEST_PATCH"
    else
        echo "Error: Test patch not found at /patches/$TEST_PATCH"
        exit 1
    fi

    # Re-install package (editable) so any code changes are picked up
    pip install -e . 2>/dev/null || true

    # Auto-detect test files from the test patch
    TEST_FILES=$(grep '^diff --git a/' "/patches/$TEST_PATCH" 2>/dev/null \\
        | sed 's|diff --git a/\\([^ ]*\\) b/.*|\\1|')
    if [[ -z "$TEST_FILES" ]]; then
        echo "Warning: Could not detect test files from patch, running tests/"
        TEST_FILES="tests/"
    fi

    echo "Running tests: $TEST_FILES"
    python -m pytest $TEST_FILES -x -v

    echo "Test execution completed!"
""")


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _docker(*args: str, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    """Run a docker CLI command."""
    cmd = ["docker", *args]
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def _image_exists(image: str) -> bool:
    r = _docker("image", "inspect", image, check=False)
    return r.returncode == 0


def _exec_in_container(container_id: str, cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Execute a bash command inside a running container."""
    return _docker(
        "exec", container_id, "bash", "-lc", cmd,
        timeout=timeout, check=False,
    )


# ---------------------------------------------------------------------------
# Tier 1: Per-repo base image
# ---------------------------------------------------------------------------

def convert_repo_base(
    swesmith_image: str,
    github_url: str,
    repo_name: str,
    *,
    push: bool = False,
    registry: str | None = None,
) -> str:
    """Create a CooperBench base image from a SWE-smith image (Tier 1).

    Steps:
      1. Pull the SWE-smith image
      2. Start a container
      3. Clone the real GitHub repo to /workspace/repo (full history)
      4. Inject the conda-aware runner.sh
      5. docker commit as <registry>/cooperbench-<repo>:base
      6. Optionally push

    Returns the base image name.
    """
    from cooperbench.utils import get_base_image_name

    base_image = get_base_image_name(repo_name, registry=registry)
    logger.info("Converting %s → %s", swesmith_image, base_image)

    # Pull SWE-smith image if not present
    if not _image_exists(swesmith_image):
        logger.info("Pulling %s …", swesmith_image)
        _docker("pull", swesmith_image, timeout=1200)

    # Start container
    r = _docker(
        "run", "-d",
        "--entrypoint", "/bin/bash",
        swesmith_image,
        "-c", "sleep infinity",
    )
    container_id = r.stdout.strip()[:12]
    logger.info("Container started: %s", container_id)

    try:
        # Clone the real repo with full history
        logger.info("Cloning %s into /workspace/repo …", github_url)
        result = _exec_in_container(
            container_id,
            f"mkdir -p /workspace && git clone {github_url} /workspace/repo",
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr}")

        # Inject runner.sh
        logger.info("Injecting runner.sh …")
        _inject_runner_sh(container_id)

        # Create /patches directory
        _exec_in_container(container_id, "mkdir -p /patches")

        # Commit with correct entrypoint
        logger.info("Committing as %s …", base_image)
        _docker(
            "commit",
            "--change", 'ENTRYPOINT ["/usr/local/bin/runner.sh"]',
            "--change", "WORKDIR /workspace/repo",
            container_id, base_image,
        )

    finally:
        _docker("rm", "-f", container_id, check=False)

    if push:
        logger.info("Pushing %s …", base_image)
        _docker("push", base_image, timeout=1200)
        logger.info("Pushed %s", base_image)

    logger.info("Base image ready: %s", base_image)
    return base_image


def _inject_runner_sh(container_id: str) -> None:
    """Write runner.sh into a running container."""
    import base64
    encoded = base64.b64encode(RUNNER_SH_CONDA.encode()).decode()
    _exec_in_container(
        container_id,
        f"echo '{encoded}' | base64 -d > /usr/local/bin/runner.sh && chmod +x /usr/local/bin/runner.sh",
    )


# ---------------------------------------------------------------------------
# Tier 2: Per-task image
# ---------------------------------------------------------------------------

def convert_task(
    repo_name: str,
    task_id: int,
    base_commit: str,
    *,
    registry: str | None = None,
    install_cmd: str = "pip install -e .",
) -> str:
    """Create a per-task image from the base image (Tier 2).

    Checks out the correct commit and re-installs the package.
    Takes ~5-10 seconds.

    Returns the task image name.
    """
    from cooperbench.utils import get_base_image_name, get_image_name

    base_image = get_base_image_name(repo_name, registry=registry)
    task_image = get_image_name(repo_name, task_id)

    if _image_exists(task_image):
        logger.info("Task image already exists: %s", task_image)
        return task_image

    if not _image_exists(base_image):
        raise RuntimeError(
            f"Base image not found: {base_image}\n"
            f"Run 'python -m cooperbench.generation.convert_swesmith base' first."
        )

    logger.info("Creating task image %s (commit %s) …", task_image, base_commit[:10])
    t0 = time.time()

    r = _docker(
        "run", "-d",
        "--entrypoint", "/bin/bash",
        base_image,
        "-c", "sleep infinity",
    )
    container_id = r.stdout.strip()[:12]

    try:
        # Checkout the specific commit
        result = _exec_in_container(
            container_id,
            f"cd /workspace/repo && git checkout {base_commit}",
        )
        if result.returncode != 0:
            raise RuntimeError(f"git checkout failed: {result.stderr}")

        # Re-install package in conda env
        result = _exec_in_container(
            container_id,
            f"{CONDA_ACTIVATE} && cd /workspace/repo && {install_cmd}",
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning("pip install returned %d: %s", result.returncode, result.stderr[-500:])

        # Commit with correct entrypoint
        _docker(
            "commit",
            "--change", 'ENTRYPOINT ["/usr/local/bin/runner.sh"]',
            "--change", "WORKDIR /workspace/repo",
            container_id, task_image,
        )

    finally:
        _docker("rm", "-f", container_id, check=False)

    elapsed = time.time() - t0
    logger.info("Task image ready: %s (%.1fs)", task_image, elapsed)
    return task_image


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def convert_batch_repos(
    repo_configs: list[dict],
    *,
    push: bool = False,
    concurrency: int = 2,
    registry: str | None = None,
) -> list[dict]:
    """Convert multiple repos in parallel (Tier 1 batch).

    Each entry in *repo_configs* should have:
      - swesmith_image: str
      - github_url: str
      - repo_name: str

    Returns a list of result dicts.
    """
    results = []

    def _do_one(cfg: dict) -> dict:
        try:
            img = convert_repo_base(
                swesmith_image=cfg["swesmith_image"],
                github_url=cfg["github_url"],
                repo_name=cfg["repo_name"],
                push=push,
                registry=registry,
            )
            return {"repo_name": cfg["repo_name"], "status": "ok", "image": img}
        except Exception as e:
            logger.error("Failed to convert %s: %s", cfg["repo_name"], e)
            return {"repo_name": cfg["repo_name"], "status": "failed", "error": str(e)}

    if concurrency <= 1:
        for cfg in repo_configs:
            results.append(_do_one(cfg))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_do_one, cfg): cfg for cfg in repo_configs}
            for f in as_completed(futures):
                results.append(f.result())

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convert_swesmith",
        description="Convert SWE-smith Docker images to CooperBench format.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- base ---
    base_p = sub.add_parser("base", help="Tier 1: create a per-repo base image")
    base_p.add_argument("--swesmith-image", required=True)
    base_p.add_argument("--github-url", required=True)
    base_p.add_argument("--repo-name", required=True)
    base_p.add_argument("--push", action="store_true")
    base_p.add_argument("--registry")

    # --- task ---
    task_p = sub.add_parser("task", help="Tier 2: create a per-task image from base")
    task_p.add_argument("--repo-name", required=True)
    task_p.add_argument("--task-id", type=int, required=True)
    task_p.add_argument("--base-commit", required=True)
    task_p.add_argument("--registry")
    task_p.add_argument("--install-cmd", default="pip install -e .")

    # --- batch ---
    batch_p = sub.add_parser("batch", help="Tier 1 batch: convert multiple repos")
    batch_p.add_argument("--config", required=True, help="JSON file with list of repo configs")
    batch_p.add_argument("--push", action="store_true")
    batch_p.add_argument("--concurrency", type=int, default=2)
    batch_p.add_argument("--registry")

    # Add --verbose to each subparser so it works positionally
    for p in (base_p, task_p, batch_p):
        p.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("urllib3", "docker", "httpcore", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if args.command == "base":
        convert_repo_base(
            swesmith_image=args.swesmith_image,
            github_url=args.github_url,
            repo_name=args.repo_name,
            push=args.push,
            registry=args.registry,
        )
    elif args.command == "task":
        convert_task(
            repo_name=args.repo_name,
            task_id=args.task_id,
            base_commit=args.base_commit,
            registry=args.registry,
            install_cmd=args.install_cmd,
        )
    elif args.command == "batch":
        with open(args.config) as f:
            configs = json.load(f)
        results = convert_batch_repos(
            configs, push=args.push,
            concurrency=args.concurrency, registry=args.registry,
        )
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"\nBatch complete: {ok}/{len(results)} repos converted")
        for r in results:
            status = "OK" if r["status"] == "ok" else f"FAILED: {r.get('error', '')[:80]}"
            print(f"  {r['repo_name']}: {status}")


if __name__ == "__main__":
    main()
