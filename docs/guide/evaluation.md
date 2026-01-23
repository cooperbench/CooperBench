# Evaluation Phase

The evaluation phase measures the quality of agent-generated code changes through test execution and merge conflict analysis.

## Overview

Evaluation performs:

1. **Test Execution**: Run test suite against generated patches
2. **Merge Analysis**: Check if patches from multiple agents can merge cleanly
3. **Conflict Scoring**: Quantify the severity of any merge conflicts

## Running Evaluation

### Via CLI

```bash
# Test evaluation (single/solo)
cooperbench evaluate \
    --setting single \
    --repo-name pallets_click_task \
    --task-id 2068 \
    --feature1-id 1 \
    --model1 gpt-5 \
    --eval-type test \
    --patch-location logs \
    --not-save-to-hf

# Merge evaluation (coop/coop_ablation)
cooperbench evaluate \
    --setting coop \
    --repo-name pallets_click_task \
    --task-id 2068 \
    --feature1-id 1 \
    --feature2-id 2 \
    --model1 gpt-5 \
    --model2 gpt-5 \
    --eval-type merge \
    --patch-location logs \
    --not-save-to-hf
```

### Via Python API

```python
import asyncio
from cooperbench import BenchSetting, FileInterface
from cooperbench.evaluation import evaluate

async def run():
    interface = FileInterface(
        setting=BenchSetting.COOP,
        repo_name="pallets_click_task",
        task_id=2068,
        k=1,
        feature1_id=1,
        feature2_id=2,
        model1="gpt-5",
        model2="gpt-5",
        save_to_hf=False,
    )

    results = await evaluate(interface, eval_type="merge", file_location="logs")
    print(f"Conflict score: {results['conflict_score']}")

asyncio.run(run())
```

## Evaluation Types

| Type | Settings | Description |
|------|----------|-------------|
| `test` | single, solo | Run feature tests against patch |
| `merge` | coop, coop_ablation | Merge analysis + tests |

## Metrics

### Test Pass Rate

Whether all feature tests pass after applying the patch.

### Merge Status

Whether two patches can be merged:

- `clean` - No conflicts, git merge succeeds
- `conflicts` - Has merge conflicts

### Conflict Score

Quantifies merge conflict severity:

```python
conflict_score = (conflict_sections * 20) + (conflict_lines * 2)
```

## Patch Locations

Patches can be loaded from:

| Location | Description |
|----------|-------------|
| `logs` | Local logs directory (default) |
| `cache` | Local cache (previously downloaded from HF) |
| `hf` | Download fresh from HuggingFace |

## Output Files

| File | Description |
|------|-------------|
| `test_result_*.json` | Test execution results |
| `merge_report_*.json` | Merge analysis report |

## Merge Report Format

```json
{
  "repo_name": "pallets_click_task",
  "task_id": 2068,
  "timestamp": "2024-01-15 10:30:00",
  "feature1": {
    "number": 1,
    "tests_passed": true,
    "test_output": "..."
  },
  "feature2": {
    "number": 2,
    "tests_passed": true,
    "test_output": "..."
  },
  "merge_status": "conflicts",
  "conflict_score": 42,
  "conflict_details": {
    "conflict_sections": 2,
    "conflict_lines": 11,
    "avg_lines_per_conflict": 5.5
  }
}
```

## Docker-Based Testing

Tests run in isolated Docker containers for reproducibility. Each task has a `Dockerfile` and `runner.sh` that:

1. Start from a base image with required toolchain (Python, Node, Go, Rust)
2. Clone the repository at a specific commit
3. Apply feature and test patches
4. Run the test suite

### Pre-built Images

Pre-built Docker images are hosted on Docker Hub under `akhatua/cooperbench-*`:

```bash
# List all available images
./scripts/list_images.sh

# Example images
# akhatua/cooperbench-pallets-click:task2068
# akhatua/cooperbench-dspy:task8635
# akhatua/cooperbench-react-hook-form:task85
```

The test runner automatically pulls from Docker Hub if available, falling back to local builds.

### Configuration

Control Docker behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `COOPERBENCH_USE_REGISTRY` | `true` | Try pulling from Docker Hub first |
| `COOPERBENCH_REGISTRY` | `akhatua` | Docker Hub username/org |
| `COOPERBENCH_IMAGE_PREFIX` | `cooperbench` | Image name prefix |

```bash
# Force local builds only
COOPERBENCH_USE_REGISTRY=false cooperbench evaluate ...

# Use a different registry
COOPERBENCH_REGISTRY=myorg cooperbench evaluate ...
```

### Building Images Locally

If images aren't available on Docker Hub, they're built automatically. You can also pre-build all images:

```bash
# Build a single image
cd dataset/pallets_click_task/task2068
docker build -t pallets_click_task_2068 .

# Build and push all images to Docker Hub (current arch)
./scripts/push_images.sh

# Build for both amd64 and arm64 (slower, for distribution)
./scripts/push_images.sh --multi-arch
```

### Image Naming

| Source | Format | Example |
|--------|--------|---------|
| Docker Hub | `{registry}/{prefix}-{repo}:{task}` | `akhatua/cooperbench-pallets-click:task2068` |
| Local | `{repo}_{task}` | `pallets_click_task_2068` |
