# Evaluation Phase

The evaluation phase measures the quality of agent-generated code changes.

!!! warning "Not Yet Migrated"
    The evaluation phase is not yet fully implemented in the new CooperBench package.
    This documentation describes the planned functionality.

## Overview

Evaluation performs:

1. **Test Execution**: Run test suite against generated patches
2. **Merge Analysis**: Check if patches from multiple agents can merge cleanly
3. **Conflict Scoring**: Quantify the severity of any merge conflicts

## Planned CLI

```bash
cooperbench evaluate \
    --setting coop \
    --repo-name my_repo \
    --task-id 123 \
    --feature1-id 1 \
    --feature2-id 2 \
    --model gpt-5 \
    --patch-location logs
```

## Metrics

### Test Pass Rate

Percentage of tests that pass after applying the patch:

```python
pass_rate = passed_tests / total_tests
```

### Merge Status

Whether two patches can be merged:

- `clean` - No conflicts
- `conflicts` - Has merge conflicts

### Conflict Score

Quantifies merge conflict severity:

```python
conflict_score = (conflict_sections * 20) + (conflict_lines * 2)
```

## Merge Strategies

Three merge strategies are evaluated:

1. **Naive**: Standard git merge
2. **Union**: Concatenates conflicting changes
3. **LLM**: Uses an LLM to resolve conflicts

## Output Files

| File | Description |
|------|-------------|
| `test_result_<model>_<repo>_task<id>_f<f>_k<k>.txt` | Test execution results |
| `merge_report_f<f1>_into_f<f2>_k<k>.json` | Merge analysis report |
| `merge_diff_f<f1>_into_f<f2>_k<k>.diff` | Merged diff file |

## Merge Report Format

```json
{
  "repo_name": "my_repo",
  "task_id": 123,
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
