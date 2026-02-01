# CooperBench

[![arXiv](https://img.shields.io/badge/arXiv-2601.13295-b31b1b.svg)](https://arxiv.org/abs/2601.13295)
[![Website](https://img.shields.io/badge/Website-cooperbench.com-blue.svg)](https://cooperbench.com)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/cooperbench/cooperbench)
[![PyPI](https://img.shields.io/pypi/v/cooperbench.svg)](https://pypi.org/project/cooperbench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Can AI agents work together as teammates?** CooperBench is the first benchmark designed to measure how well AI agents can cooperate when handling individual tasks with potential conflicts.

We find that **coordinating agents perform much worse than a single agent** given the same total workload. This coordination deficit presents a fundamental barrier to deploying AI systems that can work alongside humans or other agents.

## Installation

```bash
pip install cooperbench
```

For development:

```bash
git clone https://github.com/cooperbench/CooperBench.git
cd CooperBench
pip install -e ".[dev]"
```

### Requirements

- Python 3.12+
- [Modal](https://modal.com) account (for sandbox execution)
- Redis (for inter-agent communication in coop mode)

### Setup

1. **Modal**: Sign up at [modal.com](https://modal.com) and run `modal setup`
2. **Redis**: Run locally with `docker run -p 6379:6379 redis:7` or use a cloud provider
3. **LLM API keys**: Set in `.env` file:

```bash
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
```

### Dataset

Download the benchmark dataset:

```bash
git clone https://huggingface.co/datasets/cooperbench/cooperbench dataset/
```

## Quick Start

### CLI

Run agents on a task:

```bash
# Run cooperative agents (2 agents, shared communication)
cooperbench run -n my-experiment -r llama_index_task -m gpt-4o

# Run solo agent (1 agent handling both features)
cooperbench run -n my-experiment -r llama_index_task -m gpt-4o --setting solo

# Evaluate results
cooperbench eval -n my-experiment
```

### Python API

```python
from cooperbench import run, evaluate

# Run agents
run(
    run_name="my-experiment",
    repo="llama_index_task",
    model_name="gpt-4o",
    setting="coop",  # or "solo"
)

# Evaluate patches
evaluate(run_name="my-experiment")
```

## CLI Reference

### `cooperbench run`

Run agents on benchmark tasks.

```bash
cooperbench run -n NAME [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --name` | Experiment name (required) | - |
| `-r, --repo` | Filter by repository | all |
| `-t, --task` | Filter by task ID | all |
| `-f, --features` | Feature pair (e.g., `1,2`) | all pairs |
| `-m, --model` | LLM model | `gemini/gemini-3-flash-preview` |
| `-a, --agent` | Agent framework | `mini_swe_agent` |
| `-c, --concurrency` | Parallel tasks | `20` |
| `--setting` | `coop` or `solo` | `coop` |
| `--redis` | Redis URL | `redis://localhost:6379` |
| `--git` | Enable git collaboration | disabled |
| `--no-messaging` | Disable agent messaging | enabled |
| `--force` | Rerun existing results | skip |

### `cooperbench eval`

Evaluate completed runs.

```bash
cooperbench eval -n NAME [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --name` | Experiment name (required) | - |
| `-r, --repo` | Filter by repository | all |
| `-t, --task` | Filter by task ID | all |
| `-f, --features` | Feature pair (e.g., `1,2`) | all pairs |
| `-c, --concurrency` | Parallel evaluations | `10` |
| `--backend` | `modal`, `docker`, or `gcp` | `modal` |
| `--force` | Re-evaluate existing | skip |

## Experiment Settings

| Setting | Agents | Description |
|---------|--------|-------------|
| `coop` | 2 | Two agents with Redis messaging, each handles one feature |
| `solo` | 1 | Single agent handles both features sequentially |

## Dataset Structure

```
dataset/
  <repo_name>/
    task<id>/
      setup.sh          # Repository setup script
      run_tests.sh      # Test runner script
      feature1/
        feature.md      # Feature description
        feature.patch   # Golden implementation
        tests.patch     # Test cases
      feature2/
        ...
```

## Output Structure

Results are saved to `logs/`:

```
logs/<run_name>/<repo>/task<id>/features_<i>_<j>/
  agent1/
    trajectory.json     # Full agent trajectory
    patch.diff          # Generated patch
  agent2/
    ...
  eval.json             # Evaluation results
```

## Benchmark Statistics

| Metric | Value |
|--------|-------|
| Tasks | 652 |
| Repositories | 12 |
| Languages | Python, TypeScript, Go, Rust |

## Key Findings

1. **Agents perform worse together than alone** — GPT-5 and Claude Sonnet 4.5 achieve only 25% success with two-agent cooperation, roughly 50% lower than when a single agent handles both tasks.

2. **Communication reduces conflicts but not failures** — Agents spend up to 20% of their budget on communication, reducing merge conflicts but not improving overall success.

3. **Three capability gaps underlie coordination failures**:
   - **Expectation failures (42%)** — agents fail to integrate partner state information
   - **Communication failures (26%)** — questions go unanswered, breaking decision loops
   - **Commitment failures (32%)** — agents break promises or make unverifiable claims

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run integration tests (requires Modal)
pytest tests/ -v --run-modal

# Lint
ruff check src/
ruff format src/

# Type check
mypy src/cooperbench/
```

## Citation

```bibtex
@article{cooperbench2026,
  title={CooperBench: Why Coding Agents Cannot be Your Teammates Yet},
  author={Khatua*, Arpandeep and Zhu*, Hao and Tran†, Peter and Prabhudesai†, Arya
          and Sadrieh†, Frederic and Lieberwirth†, Johann K. and Yu, Xinkai
          and Fu, Yicheng and Ryan, Michael J. and Pei, Jiaxin and Yang, Diyi},
  journal={arXiv preprint},
  year={2026},
  url={https://arxiv.org/abs/2601.13295},
  note={*Equal contribution (Stanford) · †Equal contribution (SAP Labs)}
}
```

## License

MIT
