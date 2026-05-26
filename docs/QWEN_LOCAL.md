# Running CooperBench against a self-hosted (Qwen / Llama / etc.) endpoint

CooperBench's `claude_code` adapter drives the official `claude-code`
CLI, which only speaks Anthropic's `/v1/messages` API. To run it
against any other model you put a translation proxy in between:

```
claude-code (Anthropic format)
       │
       ▼
   LiteLLM proxy   ←  you run this; it translates Anthropic ↔ OpenAI
       │
       ▼
your OpenAI-compatible inference server (vLLM, llama.cpp, ...)
```

This document covers the canonical reproducible setup using only the
PyPI distribution — no repo checkout required.

## Prerequisites

- Docker (CooperBench runs each task in a container)
- Redis on `localhost:6379` for coop messaging:
  ```
  docker run -d --name cb-redis -p 6379:6379 redis:7-alpine
  ```
- An OpenAI-compatible endpoint URL serving your model
- Python ≥ 3.12

## Install

```bash
pip install cooperbench           # adapter + CLI
pip install 'litellm[proxy]'      # translation proxy (used internally)
```

## Canonical single-command run (Qwen3.5-9B on Modal as the example)

```bash
cooperbench run \
  --openai-base-url https://cooperbench--qwen35-9b-128k-serve.modal.run/v1 \
  --openai-model Qwen/Qwen3.5-9B \
  -m Qwen/Qwen3.5-9B \
  -a claude_code \
  --setting coop \
  -s lite \
  -r dspy_task -t 8394 -f 3,4 \
  -c 2 \
  --no-auto-eval
```

Logs land in `./logs/<run-name>/coop/<repo>/<task>/<features>/`.

### What that does under the hood

- Picks a free local port.
- Spawns `litellm --model openai/Qwen/Qwen3.5-9B --api_base <openai-base-url> ...`
  bound to that port, with `OPENAI_API_KEY=dummy` in the child env.
- Polls `/health/liveliness` until the proxy is up.
- Sets `ANTHROPIC_BASE_URL=http://localhost:<port>` and a placeholder
  `ANTHROPIC_AUTH_TOKEN` for the duration of the run.
- Tears down the proxy subprocess when the run exits (also on Ctrl-C).

### Why those flags

- `--openai-base-url` — the OpenAI-compatible endpoint (vLLM, llama.cpp, ...).
- `--openai-model` — the model name sent to that endpoint. Defaults to
  the value of `-m` if omitted.
- `-m Qwen/Qwen3.5-9B` — model name sent to claude-code (must contain
  `qwen` so the adapter's model registry picks the small-context
  profile).
- `-a claude_code` — selects the Claude Code adapter.

## Manual-proxy escape hatch

If you already have an Anthropic-format proxy running (or want to share
one across multiple `cooperbench run` invocations), use `--base-url` /
`--auth-token` instead of `--openai-base-url`:

```bash
# Start your own proxy somewhere
litellm --model openai/Qwen/Qwen3.5-9B \
  --api_base https://cooperbench--qwen35-9b-128k-serve.modal.run/v1 \
  --port 4000 ...

# Point cooperbench at it (no auto-spawn)
cooperbench run --base-url http://localhost:4000 --auth-token any \
  -m Qwen/Qwen3.5-9B ...
```

`--openai-base-url` and `--base-url` are mutually exclusive.

## How the adapter behaves with a custom endpoint

When `--base-url` is set, `src/cooperbench/agents/claude_code/adapter.py`:

1. Forwards `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` into the task
   container (rewriting `localhost` → `host.docker.internal`).
2. Adds `--add-host=host.docker.internal:host-gateway` so the container
   can reach the host proxy.
3. Preserves the model name verbatim (the proxy controls naming).
4. Writes `~/.claude/settings.json` with
   `CLAUDE_CODE_ATTRIBUTION_HEADER=0` — that header otherwise busts the
   KV cache on vLLM/llama.cpp (~90% slowdown).
5. Looks up the model name (case-insensitive substring) in
   `_MODEL_PROFILES`. For `qwen`, applies:
   - `max_output_tokens=4096`
   - `file_read_max_tokens=4000`
   - `mcp_max_output_tokens=2000`
   - `disallowed_tools=SMALL_CONTEXT_DISALLOWED_TOOLS`

Profile values fill defaults; explicit `config` keys override.
A model name without a registry match (e.g. `gpt-5.5`) still gets
routing + attribution-header fix but keeps Claude Code's stock tool
surface and budgets.

## Adding another small-context model

Edit `_MODEL_PROFILES` in
`src/cooperbench/agents/claude_code/adapter.py`:

```python
_MODEL_PROFILES = {
    "qwen": {...},
    "llama": {
        "max_output_tokens": 4096,
        "file_read_max_tokens": 4000,
        "mcp_max_output_tokens": 2000,
        "disallowed_tools": SMALL_CONTEXT_DISALLOWED_TOOLS,
    },
}
```

The key is matched as a case-insensitive substring against the model
name passed via `-m`. Cut a release after merging so PyPI users pick it
up.

## Inspecting a run

```
logs/<run-name>/coop/<repo>/<task>/<features>/
├── agent1_traj.json          # parsed trajectory + status + cost
├── agent2_traj.json
├── agent{N}.patch            # diff each agent produced (N = feature_id)
├── agent1_stream.jsonl       # raw claude-code stream events
├── agent2_stream.jsonl
├── agent1_session.jsonl      # claude-code session JSONL (tool calls, messages)
├── agent2_session.jsonl
├── agent1_sent.jsonl         # per-agent coop messaging log
├── agent2_sent.jsonl
├── conversation.json         # combined inter-agent messages
└── result.json               # both agents' summary
```

The `*_session.jsonl` files are the most useful — one JSON line per
tool call, tool result, or assistant message.

## Local-dev shortcuts (optional)

For convenience when working out of a repo checkout there are two
helper files that bundle the proxy invocation:

- `scripts/qwen_proxy.yaml` — equivalent to the inline `litellm` flags
  above
- `scripts/serve_qwen_proxy.sh` — `litellm --config <yaml> --port ...`
  wrapper

Neither is required for PyPI users — they're just easier to edit than
a long CLI invocation when you're iterating on the proxy config.
