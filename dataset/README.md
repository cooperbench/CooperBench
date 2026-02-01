---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - code
  - benchmark
  - multi-agent
  - collaboration
  - software-engineering
pretty_name: CooperBench Dataset
size_categories:
  - n<1K
---

# CooperBench Dataset

This dataset contains the benchmark tasks for evaluating multi-agent coordination in code collaboration.

**Paper**: [CooperBench: Why Coding Agents Cannot be Your Teammates Yet](https://arxiv.org/abs/2601.13295)
**Code**: [github.com/cooperbench/CooperBench](https://github.com/cooperbench/CooperBench)
**Website**: [cooperbench.com](https://cooperbench.com)

## Structure

```
dataset/
├── {repo_name}_task/           # Repository-specific tasks
│   └── task{id}/               # Individual task
│       ├── Dockerfile          # Container setup for testing
│       ├── combined.patch      # Full PR patch (all features)
│       ├── setup.sh            # Repository setup script
│       ├── runner.sh           # Test runner wrapper
│       ├── run_tests.sh        # Test execution script
│       └── feature{N}/         # Feature-specific files
│           ├── feature.md          # Feature description
│           ├── feature.patch       # Feature implementation patch
│           └── tests.patch         # Test files patch
└── README.md
```


## Repositories

| Directory | Repository | Tasks | Features |
|-----------|------------|-------|----------|
| `dottxt_ai_outlines_task` | [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines) | 3 | 22 |
| `dspy_task` | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 4 | 23 |
| `go_chi_task` | [go-chi/chi](https://github.com/go-chi/chi) | 3 | 13 |
| `huggingface_datasets_task` | [huggingface/datasets](https://github.com/huggingface/datasets) | 3 | 13 |
| `llama_index_task` | [run-llama/llama_index](https://github.com/run-llama/llama_index) | 3 | 16 |
| `openai_tiktoken_task` | [openai/tiktoken](https://github.com/openai/tiktoken) | 1 | 10 |
| `pallets_click_task` | [pallets/click](https://github.com/pallets/click) | 3 | 27 |
| `pallets_jinja_task` | [pallets/jinja](https://github.com/pallets/jinja) | 3 | 30 |
| `pillow_task` | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) | 3 | 15 |
| `react_hook_form_task` | [react-hook-form/react-hook-form](https://github.com/react-hook-form/react-hook-form) | 2 | 11 |
| `samuelcolvin_dirty_equals_task` | [samuelcolvin/dirty-equals](https://github.com/samuelcolvin/dirty-equals) | 1 | 9 |
| `typst` | [typst/typst](https://github.com/typst/typst) | 1 | 10 |

## Subsets

Pre-defined task subsets are available in `subsets/` for quick evaluation:

| Subset | Tasks | Pairs | Repos | Description |
|--------|-------|-------|-------|-------------|
| `flash` | 20 | 50 | 11 | Dev subset for rapid iteration (sampled from lite) |
| `lite` | 26 | 100 | 12 | Quick evaluation subset |

Both subsets are generated via **uniform pair-level sampling**:

```python
random.seed(42)  # fixed seed for reproducibility

# Generate all possible feature pairs from dataset
all_pairs = []
for task in all_tasks:
    for f1, f2 in combinations(task.features, 2):
        all_pairs.append((task.repo, task.task_id, [f1, f2]))

# Sample 100 pairs uniformly
sampled_pairs = random.sample(all_pairs, 100)
# Result: 23 tasks, 100 pairs, 11 repos
```

Each task in the subset includes a `pairs` field specifying exactly which feature pairs are included:

```json
{
  "repo": "pallets_jinja_task",
  "task_id": 1621,
  "pairs": [[1, 6], [2, 9], [3, 4], ...]
}
```

**Usage:**
```bash
cooperbench run --subset lite --setting coop
```

## Usage

Each task represents a pull request that was split into multiple independent features. Agents are assigned features to implement, and their outputs are evaluated for correctness and merge compatibility.

See the main CooperBench documentation for details on running experiments.
