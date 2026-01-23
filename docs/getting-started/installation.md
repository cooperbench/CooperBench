# Installation

## Requirements

- Python 3.11 or higher
- Git (for repository operations)

## Install from PyPI

```bash
pip install cooperbench
```

## Install with Optional Dependencies

CooperBench has several optional dependency groups:

```bash
# LLM support (litellm, anthropic, openai)
pip install cooperbench[llm]

# Execution support (openhands-ai)
pip install cooperbench[execution]

# Cloud deployment (modal)
pip install cooperbench[serve]

# Development tools (pytest, mypy, ruff)
pip install cooperbench[dev]

# Everything
pip install cooperbench[all]
```

## Install from Source

```bash
# Clone with submodules (required for coop execution)
git clone --recurse-submodules https://github.com/cooperbench/CooperBench.git
cd CooperBench
pip install -e ".[all]"
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## For Execution Phase

Execution requires Docker:

```bash
# Verify Docker is running
docker info
```

### For `coop` Mode Only

The `coop` setting requires custom OpenHands Docker images:

```bash
cd src/cooperbench/execution/openhands_colab
./build
```

This builds:
- `colab/openhands_colab:latest`
- `colab/openhands_runtime_colab:latest`

**Note:** Building requires `poetry` (`pip install poetry`).

## Environment Setup

Create a `.env` file with your API keys:

```bash
# Required for LLM calls
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Optional: HuggingFace for experiment storage
HF_TOKEN=your_token_here
```

## Verify Installation

```bash
cooperbench --help
```

You should see:

```
usage: cooperbench [-h] {plan,execute,evaluate} ...

CooperBench: Multi-agent coordination benchmark for code collaboration

positional arguments:
  {plan,execute,evaluate}
                        Available commands
    plan                Run planning phase
    execute             Run execution phase
    evaluate            Run evaluation phase
```
