# CooperBench

[![PyPI version](https://badge.fury.io/py/cooperbench.svg)](https://badge.fury.io/py/cooperbench)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2601.13295-b31b1b.svg)](https://arxiv.org/abs/2601.13295)

**Can AI agents work together as teammates?** CooperBench is the first benchmark designed to measure how well AI agents can cooperate when handling individual tasks with potential conflicts.

We find that **coordinating agents perform much worse than a single agent** given the same total workload. This coordination deficit presents a fundamental barrier to deploying AI systems that can work alongside humans or other agents.

📄 [Paper](https://arxiv.org/abs/2601.13295) · 🌐 [Website](https://cooperbench.com) · 🤗 [Dataset](https://huggingface.co/datasets/cooperbench/cooperbench) · 💻 [GitHub](https://github.com/cooperbench/CooperBench)

## Installation

```bash
pip install cooperbench
```

### With Optional Dependencies

```bash
# LLM support (litellm, anthropic, openai)
pip install cooperbench[llm]

# Execution support (openhands-ai)
pip install cooperbench[execution]

# Cloud deployment (modal)
pip install cooperbench[serve]

# Development tools
pip install cooperbench[dev]

# Everything
pip install cooperbench[all]
```

### From Source

```bash
git clone https://github.com/cooperbench/cooperbench.git
cd cooperbench
pip install -e ".[all]"
```

## Experiment Settings

CooperBench supports four experiment modes:

| Setting | Agents | Features | Communication | Description |
|---------|--------|----------|---------------|-------------|
| `single` | 1 | 1 | N/A | Baseline single-task performance |
| `solo` | 1 | 2 | N/A | Single agent handling multiple tasks |
| `coop` | 2 | 2 | Yes | Full multi-agent coordination |
| `coop_ablation` | 2 | 2 | Planning only | Ablation study |

## CLI Usage

### Planning Phase

Generate implementation plans from feature descriptions:

```bash
# Single agent, single feature
cooperbench plan \
    --setting single \
    --repo-name pallets_jinja_task \
    --task-id 1621 \
    --feature1-id 1 \
    --model1 anthropic/claude-sonnet-4-5-20250929 \
    --not-save-to-hf

# Cooperative planning (two agents)
cooperbench plan \
    --setting coop \
    --repo-name pallets_jinja_task \
    --task-id 1621 \
    --feature1-id 1 \
    --feature2-id 2 \
    --model1 anthropic/claude-sonnet-4-5-20250929 \
    --model2 anthropic/claude-sonnet-4-5-20250929 \
    --not-save-to-hf
```

### Execution Phase

Execute plans to generate code changes:

```bash
cooperbench execute \
    --setting single \
    --repo-name pallets_jinja_task \
    --task-id 1621 \
    --feature1-id 1 \
    --model1 anthropic/claude-sonnet-4-5-20250929 \
    --plan-location logs \
    --not-save-to-hf
```

### Evaluation Phase

Test generated patches:

```bash
cooperbench evaluate \
    --setting single \
    --repo-name pallets_jinja_task \
    --task-id 1621 \
    --feature1-id 1 \
    --model1 anthropic/claude-sonnet-4-5-20250929 \
    --eval-type test \
    --patch-location logs \
    --not-save-to-hf
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--setting, -s` | Experiment mode: `single`, `solo`, `coop`, `coop_ablation` |
| `--repo-name` | Repository/task name |
| `--task-id` | Task number |
| `--model1, -m1` | Model for first agent |
| `--model2, -m2` | Model for second agent (coop modes) |
| `--feature1-id, -i` | First feature ID |
| `--feature2-id, -j` | Second feature ID (non-single modes) |
| `--k` | Experiment run identifier (default: 1) |
| `--not-save-to-hf` | Don't save results to HuggingFace |
| `--create-pr` | Create PR when saving to HF |

## Python API

### Basic Usage

```python
from cooperbench import BenchSetting, FileInterface

# Create interface for single-agent experiment
interface = FileInterface(
    setting=BenchSetting.SINGLE,
    repo_name="pallets_jinja_task",
    task_id=1621,
    k=1,
    feature1_id=1,
    model1="anthropic/claude-sonnet-4-5-20250929",
    save_to_hf=False,
)

print(interface.setting.value)  # "single"
```

### Running Planning

```python
import asyncio
from cooperbench import BenchSetting, FileInterface
from cooperbench.planning import create_plan

async def run_planning():
    interface = FileInterface(
        setting=BenchSetting.SINGLE,
        repo_name="pallets_jinja_task",
        task_id=1621,
        k=1,
        feature1_id=1,
        model1="anthropic/claude-sonnet-4-5-20250929",
        save_to_hf=False,
    )
    
    await create_plan(interface, max_iterations=25)

asyncio.run(run_planning())
```

### Multi-Agent Coordination

```python
import asyncio
from cooperbench import BenchSetting, FileInterface
from cooperbench.planning import create_plan

async def run_coop_planning():
    interface = FileInterface(
        setting=BenchSetting.COOP,
        repo_name="pallets_jinja_task",
        task_id=1621,
        k=1,
        feature1_id=1,
        feature2_id=2,
        model1="anthropic/claude-sonnet-4-5-20250929",
        model2="anthropic/claude-sonnet-4-5-20250929",
        save_to_hf=False,
    )
    
    await create_plan(interface, max_iterations=25)

asyncio.run(run_coop_planning())
```

### Running Evaluation

```python
import asyncio
from cooperbench import BenchSetting, FileInterface
from cooperbench.evaluation import evaluate

async def run_evaluation():
    interface = FileInterface(
        setting=BenchSetting.SINGLE,
        repo_name="pallets_jinja_task",
        task_id=1621,
        k=1,
        feature1_id=1,
        model1="anthropic/claude-sonnet-4-5-20250929",
        save_to_hf=False,
    )
    
    # eval_type: "test" for single/solo, "merge" for coop
    await evaluate(interface, eval_type="test", patch_location="logs")

asyncio.run(run_evaluation())
```

## Dataset Structure

CooperBench expects tasks organized as:

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
        feature.md
        feature.patch
        tests.patch
```

## Output Structure

Results are saved to:

```
logs/<setting>/<repo_name>/task<task_id>/
  plan_<model>_k<k>_feature<id>.md      # Implementation plan
  patch_<model>_k<k>_feature<id>.patch  # Generated code
  planning_traj_<model>_k<k>.json       # Full trajectory
```

## Environment Setup

Create a `.env` file:

```bash
# Required for LLM calls
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Optional: HuggingFace for result storage
HF_TOKEN=your_token_here
```

## Requirements

- Python 3.12+
- Docker (for execution phase)
- Git

## Benchmark Statistics

| Metric | Value |
|--------|-------|
| Tasks | 652 |
| Repositories | 12 |
| Languages | Python, TypeScript, Go, Rust |
| Annotators | 8 |

Each task assigns two agents different features that can be implemented independently but may conflict without proper coordination.

## Key Findings

1. **Agents perform worse together than alone** — GPT-5 and Claude Sonnet 4.5 achieve only 25% success with two-agent cooperation, roughly 50% lower than when a single agent handles both tasks.

2. **Communication reduces conflicts but not failures** — Agents spend up to 20% of their budget on communication, reducing merge conflicts but not improving overall success.

3. **Three capability gaps underlie coordination failures**:
   - **Expectation failures (42%)** — agents fail to integrate partner state information
   - **Communication failures (26%)** — questions go unanswered, breaking decision loops
   - **Commitment failures (32%)** — agents break promises or make unverifiable claims

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
mypy src/
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
