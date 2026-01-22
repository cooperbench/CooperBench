# Planning Phase

The planning phase generates implementation plans for each feature before code execution begins.

The planning phase is where agents explore a codebase and create implementation plans for features.

## How It Works

1. **Workspace Setup**: A git worktree is created from the base repository
2. **Agent Initialization**: Agent receives feature description and system prompt
3. **Exploration Loop**: Agent uses tools to understand the codebase
4. **Plan Creation**: Agent submits implementation plan via `agreement_reached`

## Running Planning

### Via CLI

```bash
# Single agent
cooperbench plan \
    --setting single \
    --repo-name my_repo \
    --task-id 123 \
    --feature1-id 1 \
    --model1 anthropic/claude-sonnet-4-5-20250929

# Cooperative agents
cooperbench plan \
    --setting coop \
    --repo-name my_repo \
    --task-id 123 \
    --feature1-id 1 \
    --feature2-id 2 \
    --model1 anthropic/claude-sonnet-4-5-20250929 \
    --model2 anthropic/claude-sonnet-4-5-20250929
```

### Via Python API

```python
import asyncio
from cooperbench import BenchSetting, FileInterface
from cooperbench.planning import create_plan

async def run():
    interface = FileInterface(
        setting=BenchSetting.COOP,
        repo_name="my_repo",
        task_id=123,
        k=1,
        feature1_id=1,
        feature2_id=2,
        model1="anthropic/claude-sonnet-4-5-20250929",
        model2="anthropic/claude-sonnet-4-5-20250929",
        save_to_hf=False,
    )
    
    await create_plan(interface, max_iterations=25)

asyncio.run(run())
```

## Available Tools

### list_files

List directory contents:

```json
{
  "explanation": "Exploring project structure",
  "path": "src"
}
```

### read_file

Read file with line numbers:

```json
{
  "explanation": "Understanding auth implementation",
  "filename": "src/auth.py",
  "start_line": 1,
  "end_line": 50
}
```

### grep_search

Search for patterns:

```json
{
  "explanation": "Finding all authentication functions",
  "query": "def authenticate",
  "file_pattern": "*.py",
  "case_sensitive": false
}
```

### communicate_with_agent (coop only)

Send message to other agent:

```json
{
  "message": "I'll handle auth.py, can you take user.py?",
  "message_type": "proposal"
}
```

### agreement_reached

Submit final plan:

```json
{
  "plan": "# FEATURE IMPLEMENTATION PLAN\n\n## CONTEXT\n..."
}
```

## Plan Format

Agents are prompted to create detailed plans:

```markdown
# FEATURE IMPLEMENTATION PLAN

## CONTEXT
Problem: [What needs to be built/fixed]
Goal: [End state after implementation]
Requirements: [Key functional requirements]

## TECHNICAL DETAILS
Architecture: [How this fits in existing codebase]
Critical Notes: [Important gotchas, formatting rules]

## STEP-BY-STEP IMPLEMENTATION

### Step 1: Add authentication middleware
File: `src/app.py`
Location: Line 15 (after imports)
Action: Add
Before Code: [none - new addition]
After Code:
```python
def authenticate_user(request):
    return validate_token(request.headers.get('auth'))
```
Critical Details: Leave line 16-17 blank, add import at top
```

## Coordination in Coop Mode

Both `coop` and `coop_ablation` settings use identical planning logic with full agent communication. The difference between these modes is only in the execution phase (where `coop_ablation` runs agents independently without runtime communication).

In coop/coop_ablation planning, agents communicate to:

1. **Identify overlaps**: Which files both need to modify
2. **Establish ownership**: Who handles which files/sections
3. **Coordinate positions**: Line numbers for shared files
4. **Confirm agreement**: Verify coordination before finalizing

Example conversation:

```
Agent 1: [QUESTION] I need to modify src/models/User.js and 
         controllers/auth.js. What files do you need?

Agent 2: [ANALYSIS] I need User.js for profile fields and a new 
         ProfileController.js. We overlap on User.js.

Agent 1: [PROPOSAL] I'll add auth fields at lines 20-35. Can you 
         add profile fields after line 35?

Agent 2: [AGREEMENT] Agreed. I'll start my changes at line 36.
```

## Output Files

Planning creates:

| File | Description |
|------|-------------|
| `plan_<model>_k<k>_feature<id>.md` | Implementation plan |
| `planning_traj_<model>_k<k>.json` | Full trajectory with LLM calls |

## Trajectory Format

```json
{
  "metadata": {
    "planning_mode": "coop",
    "agents": [
      {"id": "agent1", "model": "claude-sonnet", "role": "feature1"},
      {"id": "agent2", "model": "claude-sonnet", "role": "feature2"}
    ],
    "start_time": "2024-01-15T10:00:00Z",
    "end_time": "2024-01-15T10:15:00Z"
  },
  "trajectory": [
    {
      "timestamp": "2024-01-15T10:00:05Z",
      "agent_id": "agent1",
      "type": "llm_call",
      "data": {
        "input": {"messages": [...], "tools": [...]},
        "output": {"content": "...", "tool_calls": [...]}
      }
    }
  ],
  "final_plans": {
    "plan1": "# FEATURE IMPLEMENTATION PLAN...",
    "plan2": "# FEATURE IMPLEMENTATION PLAN..."
  }
}
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_iterations` | 25 | Maximum planning iterations per agent |
| `save_to_hf` | True | Upload results to HuggingFace |
| `create_pr` | False | Create PR instead of direct commit |
| `k` | 1 | Experiment run identifier |
