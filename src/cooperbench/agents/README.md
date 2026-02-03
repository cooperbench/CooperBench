# CooperBench Agent Frameworks

This directory contains agent framework adapters for CooperBench. Each adapter wraps an agent implementation to conform to the `AgentRunner` interface.

## Available Agents

| Agent | Directory | Description |
|-------|-----------|-------------|
| `mini_swe_agent` | `mini_swe_agent/` | Lightweight SWE agent with bash, file editing, and messaging tools. Uses Modal sandboxes. |
| `swe_agent` | `swe_agent/` | Full SWE-agent implementation with SWE-ReX deployment. Supports advanced tooling and review. |
| `openhands_sdk` | `openhands_agent_sdk/` | OpenHands Software Agent SDK. Runs agent-server in Modal with full SDK toolset. |

## Usage

```python
from cooperbench.agents import get_runner, list_agents

# List available agents
print(list_agents())  # ['mini_swe_agent', 'swe_agent']

# Get an agent runner
runner = get_runner("mini_swe_agent")

# Run on a task
result = runner.run(
    task="Implement feature X",
    image="cooperbench/task-image:latest",
    agent_id="agent1",
    model_name="gpt-4o",
)
```

## Registering External Agents

External packages can register custom agent implementations using the `COOPERBENCH_EXTERNAL_AGENTS` environment variable. This is useful for research projects or custom agent frameworks.

### Steps

1. Create an adapter that conforms to the `AgentRunner` protocol
2. Register it using the `@register` decorator:

```python
# myproject/agents/custom_agent/adapter.py
from cooperbench.agents.registry import register
from cooperbench.agents import AgentResult

@register("my_custom_agent")
class CustomAgentRunner:
    def run(self, task, image, **kwargs) -> AgentResult:
        # Your implementation
        ...
```

3. Set the environment variable to the module path(s):

```bash
export COOPERBENCH_EXTERNAL_AGENTS="myproject.agents.custom_agent.adapter"

# Multiple agents (comma-separated)
export COOPERBENCH_EXTERNAL_AGENTS="myproject.agents.agent1.adapter,myproject.agents.agent2.adapter"
```

4. Use with cooperbench CLI:

```bash
cooperbench run -a my_custom_agent -r repo_name -t task_id
```

The external adapter modules must be importable (installed in the same Python environment).

## AgentRunner Interface

All agents must implement the `AgentRunner` protocol:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class AgentResult:
    """Result from an agent run."""
    status: str           # "Submitted", "Error", "LimitsExceeded"
    patch: str            # Generated diff/patch
    cost: float           # Total LLM cost in USD
    steps: int            # Number of agent steps/actions
    messages: list[dict]  # Conversation history
    error: str | None     # Error message if failed

class AgentRunner(Protocol):
    """Protocol for agent framework adapters."""

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "gpt-4o",
        agents: list[str] | None = None,
        comm_url: str | None = None,
        git_server_url: str | None = None,
        git_enabled: bool = False,
        messaging_enabled: bool = True,
        config: dict | None = None,
    ) -> AgentResult:
        ...
```

## Adding a New Agent

### 1. Create the adapter directory

```
agents/
  my_agent/
    __init__.py
    adapter.py
    ... (other files)
```

### 2. Implement the adapter

```python
# agents/my_agent/adapter.py

from cooperbench.agents import AgentResult
from cooperbench.agents.registry import register

@register("my_agent")
class MyAgentRunner:
    """Adapter for MyAgent framework."""

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "gpt-4o",
        agents: list[str] | None = None,
        comm_url: str | None = None,
        git_server_url: str | None = None,
        git_enabled: bool = False,
        messaging_enabled: bool = True,
        config: dict | None = None,
    ) -> AgentResult:
        """Run the agent on a task."""

        # Your implementation here
        # - Start sandbox/environment with the given image
        # - Run agent with the task description
        # - Capture the generated patch
        # - Return AgentResult

        return AgentResult(
            status="Submitted",
            patch="diff --git ...",
            cost=0.05,
            steps=10,
            messages=[...],
            error=None,
        )
```

### 3. Register the adapter

Add auto-import in `registry.py`:

```python
def _auto_register():
    """Import all adapter modules to register them."""
    try:
        import cooperbench.agents.mini_swe_agent.adapter  # noqa: F401
    except ImportError:
        pass
    try:
        import cooperbench.agents.swe_agent.adapter  # noqa: F401
    except ImportError:
        pass
    try:
        import cooperbench.agents.my_agent.adapter  # noqa: F401  # Add this
    except ImportError:
        pass
```

### 4. Add dependencies (optional)

If your agent requires additional dependencies, add them as an optional dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
my-agent = [
    "my-agent-package>=1.0",
]
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | Task description (feature to implement) |
| `image` | `str` | Docker/Modal image with the codebase |
| `agent_id` | `str` | Unique identifier for this agent instance |
| `model_name` | `str` | LLM model (e.g., `gpt-4o`, `claude-3-5-sonnet`) |
| `agents` | `list[str]` | All agent IDs in the session (for collaboration) |
| `comm_url` | `str` | Redis URL for inter-agent messaging |
| `git_server_url` | `str` | Git server URL for code collaboration |
| `git_enabled` | `bool` | Enable git push/pull/merge |
| `messaging_enabled` | `bool` | Enable send_message tool |
| `config` | `dict` | Agent-specific configuration |

## Collaboration Features

### Messaging

Agents can communicate via Redis:

```python
from cooperbench.agents.mini_swe_agent.connectors.messaging import MessagingConnector

comm = MessagingConnector(
    agent_id="agent1",
    agents=["agent1", "agent2"],
    url="redis://localhost:6379",
)

# Send message
comm.send("agent2", "I'm working on feature X")

# Receive messages
messages = comm.receive()
```

### Git Collaboration

Agents can share code via a git server:

```python
from cooperbench.agents.mini_swe_agent.connectors.git import GitConnector, GitServer

# Start git server (coordinator side)
server = GitServer(sandbox=modal_sandbox)
server_url = server.start()

# Connect agent (agent side)
git = GitConnector(agent_id="agent1", agents=["agent1", "agent2"], server_url=server_url)
git.setup(env)

# Push/pull changes
git.push()
git.pull()
```

## Environment Requirements

- **Modal**: Both agents use Modal for sandbox execution. Run `modal setup` to configure.
- **Redis**: Required for inter-agent messaging. Run `docker run -p 6379:6379 redis:7`.
- **LLM API Keys**: Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. in environment.
