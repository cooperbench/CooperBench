# Experiment Settings

CooperBench supports four experiment settings that represent different approaches to multi-agent coordination.

## Settings Overview

| Setting | Agents | Features | Planning Comm | Execution Comm | Use Case |
|---------|--------|----------|---------------|----------------|----------|
| `single` | 1 | 1 | N/A | N/A | Baseline single-task performance |
| `solo` | 1 | 2 | N/A | N/A | Single agent handling multiple tasks |
| `coop` | 2 | 2 | Yes | Yes | Full multi-agent coordination |
| `coop_ablation` | 2 | 2 | Yes | No | Ablation study (no execution coordination) |

## Single Mode

**One agent, one feature.**

The simplest setting where a single agent implements one feature. Used as a baseline to measure individual agent capability.

```python
from cooperbench import BenchSetting, FileInterface

interface = FileInterface(
    setting=BenchSetting.SINGLE,
    repo_name="my_repo",
    task_id=123,
    k=1,
    feature1_id=1,
    model1="gpt-5",
)
```

**Outputs:**
- One implementation plan
- One patch file

## Solo Mode

**One agent, two features.**

A single agent receives both feature descriptions and must create a unified plan that implements both without conflicts. Tests an agent's ability to self-coordinate.

```python
interface = FileInterface(
    setting=BenchSetting.SOLO,
    repo_name="my_repo",
    task_id=123,
    k=1,
    feature1_id=1,
    feature2_id=2,
    model1="gpt-5",
)
```

**Outputs:**
- One unified implementation plan
- One combined patch file

## Coop Mode

**Two agents, two features, with communication.**

The primary multi-agent setting. Two agents each handle one feature but can communicate to coordinate their changes and avoid conflicts.

```python
interface = FileInterface(
    setting=BenchSetting.COOP,
    repo_name="my_repo",
    task_id=123,
    k=1,
    feature1_id=1,
    feature2_id=2,
    model1="claude-sonnet",
    model2="claude-sonnet",
)
```

**Communication Protocol:**

Agents use the `communicate_with_agent` tool:

```python
communicate_with_agent(
    message="I'll modify src/auth.py lines 50-80. Which files do you need?",
    message_type="question"
)
```

Message types:
- `analysis` - Sharing findings about the codebase
- `proposal` - Suggesting coordination strategies
- `question` - Asking for information
- `concern` - Raising potential conflicts
- `agreement` - Confirming coordination plans

**Outputs:**
- Two implementation plans (one per agent)
- Two patch files
- Conversation trajectory

## Coop Ablation Mode

**Two agents, two features, no runtime communication.**

Uses the same planning phase as `coop` (with full agent communication during planning), but during execution, agents work independently without runtime coordination. This setting measures the value of execution-phase communication.

**Planning**: Identical to `coop` - agents communicate and coordinate their plans
**Execution**: Independent - agents execute without runtime communication

```python
interface = FileInterface(
    setting=BenchSetting.COOP_ABLATION,
    repo_name="my_repo",
    task_id=123,
    k=1,
    feature1_id=1,
    feature2_id=2,
    model1="claude-sonnet",
    model2="claude-sonnet",
)
```

**Outputs:**
- Two coordinated implementation plans (from shared planning)
- Two patch files (potentially diverging during execution)

## Choosing a Setting

| Goal | Recommended Setting |
|------|---------------------|
| Baseline performance | `single` |
| Self-coordination ability | `solo` |
| Multi-agent coordination | `coop` |
| Measure execution-time coordination value | Compare `coop` vs `coop_ablation` |

## Path Conventions

Output files follow predictable naming:

```
logs/<setting>/<repo_name>/task<task_id>/feature<i>[_feature<j>]/
  plan_<model>_k<k>_feature<id>.md
  patch_<model>_k<k>_feature<id>.patch
  planning_traj_<model>_k<k>.json
```

Examples:
- `logs/single/my_repo/task123/feature1/plan_gpt5_k1_feature1.md`
- `logs/coop/my_repo/task123/feature1_feature2/plan_claude_k1_feature1.md`
