# Feature Generation Pipeline

Automated generation of new benchmark features using LLM agents running on Modal.

## Quick Start

```bash
# From project root
cd /path/to/CooperBench

# Generate a single feature
python -m cooperbench.generation --task dspy_task/task8394

# Just see the prompt (no agent run)
python -m cooperbench.generation --task dspy_task/task8394 --prompt-only

# Validate existing patches
python -m cooperbench.generation --task dspy_task/task8394 --validate feature.patch tests.patch
```

## Usage

### Generate Features

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

### Validate Patches

```bash
# Check if patches pass tests and conflict with existing features
python -m cooperbench.generation \
    --task dspy_task/task8394 \
    --validate ./generated/feature.patch ./generated/tests.patch
```

## How It Works

### 1. Prompt Building (`prompt.py`)

Analyzes existing features in a task to build a generation prompt:
- Reads all `feature.md` files to understand the format
- Parses `feature.patch` files to identify "hot spots" (frequently modified files/lines)
- Instructs agent to create conflicting features

### 2. Agent Execution (`generator.py`)

Runs `mini_swe_agent` on Modal with the task's Docker image:
- Agent explores the codebase
- Implements a new feature that modifies similar code regions
- Writes tests
- Verifies tests pass

### 3. Patch Splitting (`splitter.py`)

Separates agent's output into:
- `feature.patch` - Source code changes
- `tests.patch` - Test file changes
- `feature.md` - Feature description extracted from agent output

### 4. Validation (`validator.py`)

All validation runs in Modal sandboxes:
- **Test validation**: Runs tests using existing `runner.sh`
- **Conflict detection**: Applies patches to git branches and attempts merge

A generated feature is **valid** if:
- ✅ All tests pass
- ✅ Conflicts with at least 1 existing feature

## Module Structure

```
generation/
├── __init__.py      # Package exports
├── __main__.py      # CLI entry point
├── generator.py     # Main orchestrator
├── prompt.py        # Prompt building
├── splitter.py      # Patch splitting
├── validator.py     # Modal-based validation
└── README.md        # This file
```

## Programmatic Usage

```python
from cooperbench.generation import generate_feature, validate_generated_feature

# Generate a new feature
result = generate_feature(
    task_dir="dataset/dspy_task/task8394",
    model_name="gpt-4o",
    backend="modal",
)

if result.success:
    print(f"Feature patch:\n{result.feature_patch}")
    print(f"Tests patch:\n{result.tests_patch}")
    print(f"Cost: ${result.agent_cost:.4f}")

# Validate patches
validation = validate_generated_feature(
    repo_name="dspy_task",
    task_id=8394,
    feature_patch=result.feature_patch,
    tests_patch=result.tests_patch,
)

print(f"Valid: {validation['valid']}")
print(f"Conflicts with features: {validation['conflict_result']['conflicts']}")
```

## Success Criteria

A generated feature is considered **successful** if:

1. **Tests Pass**: The feature implementation is correct and all tests (including new tests) pass
2. **Has Conflicts**: The feature conflicts with at least one existing feature when merging

The conflict requirement ensures the generated feature is useful for testing multi-agent coordination - features that merge cleanly don't test the coordination aspects of the benchmark.
