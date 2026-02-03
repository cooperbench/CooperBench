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
| `docker/Dockerfile.agent-server` | Layers agent-server onto CooperBench task images |
| `docker/build_agent_image.sh` | Builds multi-platform images (`linux/amd64`, `linux/arm64`) |
| `openhands-sdk/` | Local copy of SDK (for customizations) |
| `openhands-tools/` | Local copy of tools (for customizations) |
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

Browser tools are disabled since tasks run headless.

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

**Not yet implemented.** The adapter signature includes collaboration parameters (`agents`, `comm_url`, `git_server_url`, etc.) for future multi-agent support, but these are currently ignored.

## Upstream SDK

This integration is based on the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk). See the upstream README for SDK documentation:

- [Getting Started](https://docs.openhands.dev/sdk/getting-started)
- [Architecture](https://docs.openhands.dev/sdk/arch/overview)
- [API Reference](https://docs.openhands.dev/sdk/guides/agent-server/api-reference)
