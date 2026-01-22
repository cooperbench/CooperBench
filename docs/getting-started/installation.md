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
git clone https://github.com/cooperbench/cooperbench.git
cd cooperbench
pip install -e ".[all]"
```

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
