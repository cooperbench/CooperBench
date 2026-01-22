# Execution Phase

The execution phase takes implementation plans and generates actual code changes.

!!! warning "Not Yet Migrated"
    The execution phase is not yet fully implemented in the new CooperBench package.
    This documentation describes the planned functionality.

## Overview

Execution uses the [OpenHands](https://github.com/All-Hands-AI/OpenHands) agent framework to:

1. Load the implementation plan
2. Execute changes in a sandboxed environment
3. Generate a patch file with the changes

## Planned CLI

```bash
cooperbench execute \
    --setting single \
    --repo-name my_repo \
    --task-id 123 \
    --feature1-id 1 \
    --model gpt-5 \
    --plan-location logs
```

## Planned Python API

```python
from cooperbench import BenchSetting, FileInterface
from cooperbench.execution import execute_plan

interface = FileInterface(
    setting=BenchSetting.SINGLE,
    repo_name="my_repo",
    task_id=123,
    k=1,
    feature1_id=1,
    model1="gpt-5",
)

await execute_plan(interface, plan_location="logs")
```

## Plan Locations

Plans can be loaded from:

- `logs` - Local logs directory
- `cache` - Local cache (previously downloaded from HF)
- `hf` - Download fresh from HuggingFace

## Output

Execution produces:

| File | Description |
|------|-------------|
| `patch_<model>_k<k>_feature<id>.patch` | Git diff of changes |
| `execution_traj_<model>_k<k>.json` | OpenHands trajectory |

## Docker Integration

OpenHands runs in Docker containers. Container naming:

```
openhands-app-{repo_name}-{task_id}-feature{f1}_k{k}
openhands-app-{repo_name}-{task_id}-feature{f1}_feature{f2}_k{k}
```
