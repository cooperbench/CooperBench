# Data Creation Pipeline

End-to-end pipeline for creating new CooperBench benchmark tasks from any GitHub repository, using pre-built Docker images from [SWE-smith](https://github.com/SWE-smith/SWE-smith).

## Overview

The pipeline has two phases: a **one-time image conversion** (per repo) and a **four-stage generation pipeline** (per run).

```
SWE-smith image ──► Base image (Tier 1, once per repo)
                        │
                        ▼
  Collect PRs ──► Filter ──► Onboard (Tier 2, per task) ──► Controller
                                                              │
                                                        CooperBench tasks
                                                        with feature pairs
```

**Tier 1** converts a SWE-smith Docker image into a CooperBench-compatible "base" image by cloning the full GitHub repo with git history and injecting a conda-aware test runner. This is done once per repository and pushed to Docker Hub.

**Tier 2** creates lightweight per-task images (~5 seconds each) by checking out the PR's base commit from the base image and reinstalling the package. This happens automatically during the pipeline's onboard stage.

## Prerequisites

- Docker running locally
- A [GitHub token](https://github.com/settings/tokens) with `repo` scope (set as `GITHUB_TOKEN` in `.env`)
- An LLM API key (set as `GEMINI_API_KEY` in `.env`)
- Docker Hub login (for pushing images): `docker login`

## Step 1: Find the SWE-smith Image

SWE-smith publishes pre-built Docker images for 128+ repos on Docker Hub under the `jyangballin` namespace:

```bash
docker search jyangballin/swesmith --limit 25
```

The naming convention is `jyangballin/swesmith.x86_64.<github_org>_<fork_id>_<repo>.<hash>`.

For example, the Flask image is `jyangballin/swesmith.x86_64.pallets_1776_flask.bc098406`.

## Step 2: Convert to CooperBench Base Image (Tier 1)

Run this **once per repository**. It pulls the SWE-smith image, clones the real GitHub repo with full git history into `/workspace/repo`, injects a conda-aware `runner.sh`, and commits the result.

### Single repo

```bash
python -m cooperbench.generation.convert_swesmith base \
    --swesmith-image jyangballin/swesmith.x86_64.pallets_1776_flask.bc098406 \
    --github-url https://github.com/pallets/flask.git \
    --repo-name flask_task \
    --push
```

This produces `priyank0003/cooperbench-flask:base`. Takes 2-5 minutes depending on repo size.

### Batch (multiple repos)

Create a `repos.json` config file:

```json
[
  {
    "swesmith_image": "jyangballin/swesmith.x86_64.pallets_1776_flask.bc098406",
    "github_url": "https://github.com/pallets/flask.git",
    "repo_name": "flask_task"
  },
  {
    "swesmith_image": "jyangballin/swesmith.x86_64.pallets_1776_jinja.a1b2c3d4",
    "github_url": "https://github.com/pallets/jinja.git",
    "repo_name": "jinja_task"
  }
]
```

Then run:

```bash
python -m cooperbench.generation.convert_swesmith batch \
    --config repos.json --push --concurrency 2
```

### CLI reference

```
convert_swesmith base  --swesmith-image IMG --github-url URL --repo-name NAME [--push] [--registry REG]
convert_swesmith task  --repo-name NAME --task-id ID --base-commit SHA [--registry REG] [--install-cmd CMD]
convert_swesmith batch --config FILE [--push] [--concurrency N] [--registry REG]
```

## Step 3: Run the Generation Pipeline

With the base image ready, the four-stage pipeline creates benchmark tasks automatically:

```bash
python -m cooperbench.generation.pipeline \
    --repo pallets/flask \
    --repo-name flask_task \
    --repo-url https://github.com/pallets/flask.git \
    --use-swesmith \
    --max-tasks 10 \
    --concurrency 5 \
    --model gemini/gemini-3-flash-preview
```

### Pipeline stages

| Stage | Name | What it does |
|-------|------|-------------|
| 1 | **Collect** | Fetches merged PRs from GitHub and builds SWE-bench-style instances |
| 2 | **Filter** | Applies quality filters (min test lines, max patch size, etc.) |
| 3 | **Onboard** | Converts each PR to a task directory with patches, feature descriptions, and Docker images. With `--use-swesmith`, creates per-task images from the base in ~5 seconds each (Tier 2) |
| 4 | **Controller** | Runs the LLM controller to expand/decompose features, check for conflicts, and classify hard vs. trivial datapoints |

### Key flags

| Flag | Description | Default |
|------|-------------|---------|
| `--use-swesmith` | Use SWE-smith converted base images instead of building from Dockerfiles | off |
| `--max-tasks N` | Stop after N validated tasks (0 = unlimited) | `0` |
| `--concurrency N` | Parallel workers for the controller stage | `1` |
| `--stage NAME` | Run only one stage (`collect`, `filter`, `onboard`, `controller`) | all |
| `--from-run DIR` | Fork earlier stages from a previous run directory | - |
| `--skip-build` | Skip Docker image building (use existing images) | off |
| `--skip-validate` | Skip Docker validation of onboarded tasks | off |
| `--model MODEL` | LLM model for feature.md generation and controller | `gemini/gemini-3-flash-preview` |
| `--max-pulls N` | Max PRs to collect from GitHub | `200` |
| `--target-features N` | Target number of features per task | `8` |
| `--max-cost F` | Max LLM cost per task in USD | `3.0` |

### Running a single stage

You can resume from a previous run by forking earlier stages:

```bash
# Run only the controller stage, reusing collect/filter/onboard from a previous run
python -m cooperbench.generation.pipeline \
    --repo pallets/flask --repo-name flask_task \
    --repo-url https://github.com/pallets/flask.git \
    --use-swesmith \
    --stage controller \
    --from-run logs/pipeline_runs/flask_task_20260409_134500 \
    --concurrency 5
```

## Step 4: Inspect Results

### Pipeline outputs

```
logs/pipeline_runs/<repo_name>_<timestamp>/
    pipeline_config.json
    stage1_collect/result.json      # PRs fetched, instances built
    stage2_filter/result.json       # candidates after filtering
    stage3_onboard/result.json      # task dirs created, validation results
    stage4_controller/
        result.json                 # aggregate: features, hard datapoints, cost
        task<id>_result.json        # per-task controller output
```

### Dataset outputs

Each validated task gets a directory in `dataset/`:

```
dataset/<repo_name>/task<id>/
    runner.sh                       # conda-aware test runner (auto-detects test files)
    feature1/
        feature.md                  # LLM-generated feature description
        feature.patch               # golden implementation patch
        tests.patch                 # test cases
    feature2/ ...                   # additional features from the controller
    controller_log.json             # conflict analysis and datapoint classification
```

## Manual Tier 2 Conversion

To create a per-task image outside the pipeline (e.g., for debugging):

```bash
python -m cooperbench.generation.convert_swesmith task \
    --repo-name flask_task \
    --task-id 5939 \
    --base-commit c34d6e81fd8e405e6d4178bf24b364918811ef17
```

## How SWE-smith Images Differ from CooperBench

| Aspect | SWE-smith | CooperBench (after conversion) |
|--------|-----------|-------------------------------|
| Repo location | `/testbed/` | `/workspace/repo` |
| Python env | conda `testbed` | conda `testbed` (preserved) |
| Git history | Single squashed "Initial commit" | Full history (cloned from GitHub) |
| `runner.sh` | Not present | Injected during Tier 1 |
| Entrypoint | `/bin/bash` | `/usr/local/bin/runner.sh` |

The two-tier conversion bridges these differences. After conversion, the images are fully compatible with all CooperBench modules (agents, eval, generation).

---

## Feature Generation (Single Task)

For generating new features on an **existing** task (not creating tasks from scratch), use the generation CLI:

### Quick Start

```bash
# Generate a single feature
python -m cooperbench.generation --task dspy_task/task8394

# Just see the prompt (no agent run)
python -m cooperbench.generation --task dspy_task/task8394 --prompt-only

# Validate existing patches
python -m cooperbench.generation --task dspy_task/task8394 --validate feature.patch tests.patch
```

### Usage

```bash
# Single attempt with Gemini 3 Flash (default)
python -m cooperbench.generation --task dspy_task/task8394

# Multiple attempts with output directory
python -m cooperbench.generation --task dspy_task/task8394 --attempts 5 --output ./generated

# Use different model
python -m cooperbench.generation --task dspy_task/task8394 --model claude-3-opus

# Use local Docker instead of Modal
python -m cooperbench.generation --task dspy_task/task8394 --backend docker
```

### How it works

1. **Prompt Building** (`prompt.py`) — analyzes existing features to build a generation prompt targeting conflicting code regions
2. **Agent Execution** (`generator.py`) — runs `mini_swe_agent` in a sandbox to implement the feature and write tests
3. **Patch Splitting** (`splitter.py`) — separates output into `feature.patch`, `tests.patch`, and `feature.md`
4. **Validation** (`validator.py`) — verifies tests pass and the feature conflicts with at least one existing feature

### Programmatic usage

```python
from cooperbench.generation import generate_feature, validate_generated_feature

result = generate_feature(
    task_dir="dataset/dspy_task/task8394",
    model_name="gpt-4o",
    backend="modal",
)

if result.success:
    print(f"Feature patch:\n{result.feature_patch}")
    print(f"Cost: ${result.agent_cost:.4f}")

validation = validate_generated_feature(
    repo_name="dspy_task",
    task_id=8394,
    feature_patch=result.feature_patch,
    tests_patch=result.tests_patch,
)
print(f"Valid: {validation['valid']}")
```

## Module Structure

```
generation/
├── __init__.py              # Package exports
├── __main__.py              # CLI entry point (single-task feature generation)
├── README.md                # This file
├── collect.py               # Stage 1: PR collection from GitHub
├── onboard.py               # Stage 3: PR-to-task conversion, Docker build, validation
├── pipeline.py              # End-to-end pipeline orchestrator (stages 1-4)
├── convert_swesmith.py      # SWE-smith → CooperBench image converter (Tier 1 + 2)
├── controller.py            # Stage 4: LLM controller for feature expansion
├── expand.py                # Feature expansion (generate new features)
├── decompose.py             # Feature decomposition
├── resolve.py               # Conflict resolution
├── verify.py                # Deterministic patch verification
├── generator.py             # Single-feature generation orchestrator
├── prompt.py                # Prompt building for feature generation
├── splitter.py              # Patch splitting
└── validator.py             # Sandbox-based validation
```
