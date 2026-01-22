# CooperBench

**Multi-agent coordination benchmark for collaborative code generation.**

CooperBench evaluates how AI agents collaborate on software engineering tasks, specifically measuring their ability to coordinate changes across shared codebases without creating merge conflicts.

## Key Features

- **Multiple Experiment Settings**: Compare single-agent vs multi-agent approaches
- **Planning Phase**: Agents explore codebases and create implementation plans
- **Execution Phase**: Plans are executed using OpenHands agent framework
- **Evaluation Phase**: Automated testing and merge conflict analysis

## Experiment Settings

| Setting | Description |
|---------|-------------|
| `single` | Single agent implements one feature |
| `solo` | Single agent implements two features simultaneously |
| `coop` | Two agents coordinate via communication to implement separate features |
| `coop_ablation` | Two agents with planning communication but no execution communication (ablation study) |

## Quick Example

```python
from cooperbench import BenchSetting, FileInterface
from cooperbench.planning import create_plan

# Create experiment configuration
interface = FileInterface(
    setting=BenchSetting.COOP,
    repo_name="example_repo",
    task_id=123,
    k=1,
    feature1_id=1,
    feature2_id=2,
    model1="anthropic/claude-sonnet-4-5-20250929",
    model2="anthropic/claude-sonnet-4-5-20250929",
)

# Run planning phase
await create_plan(interface, max_iterations=25)
```

## CLI Usage

```bash
# Single agent planning
cooperbench plan --setting single --repo-name my_repo --task-id 123 \
    --feature1-id 1 --model1 gpt-5

# Cooperative planning with two agents
cooperbench plan --setting coop --repo-name my_repo --task-id 123 \
    --feature1-id 1 --feature2-id 2 \
    --model1 claude-sonnet --model2 claude-sonnet
```

## Installation

```bash
pip install cooperbench

# With LLM support
pip install cooperbench[llm]

# With all dependencies
pip install cooperbench[all]
```

## Documentation

- [Getting Started](getting-started/installation.md) - Installation and setup
- [Concepts](concepts/overview.md) - Understanding CooperBench architecture
- [User Guide](guide/planning.md) - Detailed usage instructions
- [API Reference](api/core.md) - Python API documentation
