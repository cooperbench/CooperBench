# CooperBench Dataset

This directory contains the benchmark tasks for evaluating multi-agent coordination in code collaboration.

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

## Statistics

| Metric | Count |
|--------|-------|
| Repositories | 12 |
| Tasks | 30 |
| Features | 199 |

## Repositories

- `dottxt_ai_outlines_task` - 3 tasks, 22 features
- `dspy_task` - 4 tasks, 23 features
- `go_chi_task` - 3 tasks, 13 features
- `huggingface_datasets_task` - 3 tasks, 13 features
- `llama_index_task` - 3 tasks, 16 features
- `openai_tiktoken_task` - 1 task, 10 features
- `pallets_click_task` - 3 tasks, 27 features
- `pallets_jinja_task` - 3 tasks, 30 features
- `pillow_task` - 3 tasks, 15 features
- `react_hook_form_task` - 2 tasks, 11 features
- `samuelcolvin_dirty_equals_task` - 1 task, 9 features
- `typst` - 1 task, 10 features

## Usage

Each task represents a pull request that was split into multiple independent features. Agents are assigned features to implement, and their outputs are evaluated for correctness and merge compatibility.

See the main CooperBench documentation for details on running experiments.
