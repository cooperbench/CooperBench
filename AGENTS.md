# CooperBench Agent Rules

## Python Environment

**Always use the `cooper` conda environment.** Never install packages into or run commands with the global/base Python.

```bash
# Run any Python command:
conda run -n cooper python -m cooperbench.generation.verify ...

# Install packages:
conda run -n cooper pip install <package>

# Reinstall cooperbench in editable mode after code changes:
conda run -n cooper pip install -e .
```

The `cooper` env uses Python 3.12 and has all CooperBench dependencies pre-installed.

## Running Long Commands

`conda run` buffers stdout. For unbuffered output (needed for monitoring long-running scripts), activate the env directly:

```bash
eval "$(conda shell.zsh hook 2>/dev/null)" && conda activate cooper && PYTHONUNBUFFERED=1 python -u -m cooperbench.generation.controller ...
```

Key: `PYTHONUNBUFFERED=1` + `python -u` ensures real-time streaming.

## LLM Model

Default model for all LLM calls (generation, resolution, feature.md, etc.):

```
gemini/gemini-3-flash-preview
```

Use this as the `--model` flag value and as the default `model_name` parameter.
