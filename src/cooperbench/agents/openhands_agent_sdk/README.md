# OpenHands SDK Adapter for CooperBench

This directory contains the OpenHands Software Agent SDK integration for CooperBench. It runs the full OpenHands agent-server remotely in Modal sandboxes, preserving all SDK capabilities while enabling isolated code execution.

## What We Built

We built a **minimal adapter** that leverages the OpenHands SDK's native architecture:

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Local Machine                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   CooperBench CLI                           │    │
│  │    cooperbench run -a openhands_sdk -m vertex_ai/...        │    │
│  └────────────────────────────┬────────────────────────────────┘    │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐    │
│  │               OpenHandsSDKRunner (adapter.py)               │    │
│  │  - Creates Modal sandbox with agent-server image            │    │
│  │  - Connects via SDK's RemoteWorkspace                       │    │
│  │  - Uses RemoteConversation to run agent loop                │    │
│  └────────────────────────────┬────────────────────────────────┘    │
└───────────────────────────────│─────────────────────────────────────┘
                                │ HTTP (via Modal Tunnel)
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                     Modal Sandbox (Cloud)                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           akhatua/cooperbench-<repo>:<task>-oh              │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  Python 3.12 venv (/opt/agent-server-venv)          │    │    │
│  │  │    - openhands-agent-server                         │    │    │
│  │  │    - openhands-sdk (local copy with customizations) │    │    │
│  │  │    - openhands-tools (terminal, file_editor, etc.)  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  Task Environment (original Python/runtime)         │    │    │
│  │  │    - /workspace/repo (the codebase to modify)       │    │    │
│  │  │    - Task-specific dependencies intact              │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| File | Purpose |
|------|---------|
| `adapter.py` | CooperBench adapter using `RemoteConversation` + `RemoteWorkspace` |
| `connectors/` | Infrastructure components (`ModalRedisServer`, `ModalGitServer`) |
| `docker/Dockerfile.agent-server` | Layers agent-server onto CooperBench task images |
| `docker/build_agent_image.sh` | Builds multi-platform images (`linux/amd64`, `linux/arm64`) |
| `openhands-sdk/` | Local copy of SDK (for customizations) |
| `openhands-tools/` | Local copy of tools (incl. `SendMessageTool`, `ReceiveMessageTool`) |
| `openhands-workspace/` | Local copy of workspace implementations |

### How It Works

1. **Image Layering**: CooperBench task images (e.g., `akhatua/cooperbench-llama-index:task17244`) are extended with the OpenHands agent-server. The `-oh` suffix denotes these combined images.

2. **Python Isolation**: The agent-server runs in an isolated Python 3.12 venv (`/opt/agent-server-venv`), while the task's original Python/runtime remains untouched for code execution.

3. **Remote Execution**: The adapter uses the SDK's native `RemoteConversation` and `RemoteWorkspace` classes—no custom agent loop or tool reimplementation needed.

4. **Credential Handling**: API keys and Google Cloud ADC are securely passed to Modal via `modal.Secret`.

## Usage

### Running with CLI

```bash
# Run on a single task
cooperbench run \
  --setting solo \
  -a openhands_sdk \
  -m vertex_ai/gemini-2.5-flash \
  -r llama_index_task \
  -t 17244 \
  -f 1,2

# Run on a subset
cooperbench run \
  --setting solo \
  -a openhands_sdk \
  -m anthropic/claude-sonnet-4-5-20250929 \
  -s lite
```

### Building Agent Images

Before running, you need to build the `-oh` images for your tasks:

```bash
cd src/cooperbench/agents/openhands_agent_sdk/docker

# Build for a single task
./build_agent_image.sh llama-index task17244

# This creates: akhatua/cooperbench-llama-index:task17244-oh
```

The script:
1. Takes the base CooperBench task image
2. Installs Python 3.12 via `uv` in an isolated venv
3. Installs `openhands-agent-server` and local SDK packages
4. Pushes multi-platform image to Docker Hub

### Available Tools

The adapter uses OpenHands SDK's default toolset via `get_default_agent(cli_mode=True)`:

- **TerminalTool** - Execute shell commands
- **FileEditorTool** - View, create, edit files with str_replace
- **GlobTool** - Find files by pattern
- **GrepTool** - Search file contents
- **TaskTrackerTool** - Track implementation progress
- **SendMessageTool** - Send messages to teammates (coop mode only)
- **ReceiveMessageTool** - Check for messages from teammates (coop mode only)

Browser tools are disabled since tasks run headless.

Messaging tools are only registered when `REDIS_URL` and multiple `AGENTS` are configured (coop mode).

## Customization

### Adding New Tools

1. Add/modify tools in `openhands-tools/openhands/tools/`
2. Rebuild the `-oh` images to include your changes
3. Update `adapter.py` if the tool needs special registration

### Modifying the Agent

The agent behavior comes from the SDK. To customize:

1. Modify prompts in `openhands-sdk/openhands/sdk/agent/prompts/`
2. Adjust condenser settings in `openhands-sdk/openhands/sdk/context/condenser/`
3. Rebuild images to apply changes

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google AI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `VERTEX_PROJECT` | GCP project for Vertex AI |
| `VERTEX_LOCATION` | GCP region for Vertex AI |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON |

The adapter automatically reads Google Cloud ADC from `~/.config/gcloud/application_default_credentials.json` if `GOOGLE_APPLICATION_CREDENTIALS` is not set.

## Experiment Naming

The CLI uses agent shorthand `oh` in experiment names:

```
solo-oh-gemini-2-5-flash-llama-index-17244
│    │  │                 │           │
│    │  │                 │           └── task ID
│    │  │                 └── repo name
│    │  └── model name (cleaned)
│    └── agent shorthand (openhands_sdk → oh)
└── setting
```

## Collaboration Support

The adapter supports multi-agent collaboration via Redis messaging and Git code sharing. **Fully decoupled**: the adapter manages its own infrastructure internally—the CooperBench orchestrator just passes flags.

### Architecture

Everything runs in Modal - no local Redis or Git server needed:

```
┌─────────────────────────────────────────────────────────────────────┐
│                             Modal                                   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │            ModalRedisServer (shared sandbox)               │     │
│  │  - Single Redis instance for all agents in the coop run    │     │
│  │  - First agent to run() creates it, others reuse           │     │
│  │  - Cleanup when last agent finishes                        │     │
│  └───────────▲─────────────────────────────▲──────────────────┘     │
│              │                             │                        │
│  ┌───────────┴───────────┐   ┌─────────────┴─────────────┐          │
│  │  Agent-Server Sandbox │   │  Agent-Server Sandbox     │          │
│  │  (agent_0)            │   │  (agent_1)                │          │
│  │                       │   │                           │          │
│  │  - SendMessageTool    │   │  - SendMessageTool        │          │
│  │    pushes to Redis    │   │    pushes to Redis        │          │
│  │                       │   │                           │          │
│  │  - ReceiveMessageTool │   │  - ReceiveMessageTool     │          │
│  │    pops from Redis    │   │    pops from Redis        │          │
│  └───────────┬───────────┘   └─────────────┬─────────────┘          │
│              │                             │                        │
│              │  git push/pull              │  git push/pull         │
│              ▼                             ▼                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │            ModalGitServer (per-run sandbox)                │     │
│  │  - git-daemon with receive-pack enabled                    │     │
│  │  - Each run gets isolated bare repo                        │     │
│  │  - Agents push to their branch, fetch teammates' branches  │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘

Coordination via module-level registry with thread-safe locking.
```

### Decoupled Design

The adapter handles its own infrastructure:

```python
# CooperBench orchestrator just passes flags:
agent.run(
    task=task,
    image=image,
    agents=["agent_0", "agent_1"],
    messaging_enabled=True,
    git_enabled=True,
    # comm_url/git_server_url can be None - adapter creates Modal servers
)

# Internally, adapter:
# 1. Detects coop mode (multiple agents + messaging_enabled or git_enabled)
# 2. Creates ModalRedisServer for messaging (first agent, thread-safe)
# 3. Creates ModalGitServer for code sharing (per-run, thread-safe)
# 4. Sets up 'team' remote in each agent's sandbox
# 5. Passes URLs to sandbox via env vars
# 6. Releases references when done (last agent cleans up)
```

This means:
- **No external Redis or Git server needed** - adapter creates its own
- **No orchestrator changes** - just pass standard flags
- **Self-contained** - ignores `git_server_url` from coop.py (manages its own)
- **Portable** - can switch to different backends later

### Running Coop Mode

```bash
# Run with 2 agents on collaborative tasks (messaging only)
cooperbench run \
  --setting coop \
  -a openhands_sdk \
  -m vertex_ai/gemini-2.5-flash \
  -r llama_index_task \
  -t 17244

# Run with git collaboration enabled
cooperbench run \
  --setting coop \
  -a openhands_sdk \
  -m vertex_ai/gemini-2.5-flash \
  -r llama_index_task \
  -t 17244 \
  --git
```

### How Messaging Works

1. **Sending Messages**: Agent uses `SendMessageTool` to push messages to teammate's Redis inbox.

2. **Receiving Messages**: Agent uses `ReceiveMessageTool` to pop messages from their own inbox. Messages are formatted as: `[Message from agent_0]: ...`

3. **Collaboration Prompt**: Task prompt is augmented with instructions explaining the messaging tools.

### How Git Collaboration Works

1. **Git Server Creation**: When `git_enabled=True`, the adapter creates a `ModalGitServer` sandbox running `git-daemon` with receive-pack enabled.

2. **Remote Setup**: Each agent's sandbox is configured with a `team` remote pointing to the shared git server. Each agent gets their own branch (e.g., `agent_0`, `agent_1`).

3. **Code Sharing**: Agents can push their changes and fetch teammates' branches:
   ```bash
   git push team agent_0        # Push your changes
   git fetch team               # Fetch all branches
   git diff HEAD..team/agent_1  # See teammate's changes
   git merge team/agent_1       # Merge teammate's work
   ```

4. **System Prompt**: Agents are instructed about the `team` remote and their branch name.

### Environment Variables in Sandbox

The adapter passes these to each sandbox for collaboration tools:

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Redis URL (tunnel to ModalRedisServer) |
| `GIT_URL` | Git URL (tunnel to ModalGitServer) |
| `AGENT_ID` | This agent's ID |
| `AGENTS` | Comma-separated list of all agent IDs |

## Upstream SDK

This integration is based on the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk). See the upstream README for SDK documentation:

- [Getting Started](https://docs.openhands.dev/sdk/getting-started)
- [Architecture](https://docs.openhands.dev/sdk/arch/overview)
- [API Reference](https://docs.openhands.dev/sdk/guides/agent-server/api-reference)
