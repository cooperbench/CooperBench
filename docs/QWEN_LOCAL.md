# Running CooperBench against a self-hosted Qwen (or any vLLM endpoint)

CooperBench's `claude_code` adapter wraps the official `claude-code` CLI,
which speaks Anthropic's `/v1/messages` API. vLLM v0.17.1+ implements
that same API natively — so claude-code can talk to a vLLM server
**directly, with no translation proxy in between**.

```
claude-code (Anthropic /v1/messages) ───► vLLM /v1/messages
```

## Prerequisites

- Docker (CooperBench runs each task in a container)
- Redis on `localhost:6379` for coop messaging:
  ```
  docker run -d --name cb-redis -p 6379:6379 redis:7-alpine
  ```
- A vLLM (v0.17.1+) endpoint serving your model with tool-calling
  enabled. Reference serve flags (Qwen3.5-9B at 128k):
  ```
  vllm serve Qwen/Qwen3.5-9B \
    --max-model-len 131072 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
  ```

## Install

```bash
pip install cooperbench
```

That's it. No `litellm[proxy]`, no extras.

## Run

```bash
cooperbench run \
  --base-url https://your-vllm-host.example.com \
  --auth-token dummy \
  -m Qwen/Qwen3.5-9B \
  -a claude_code \
  --setting coop \
  -s lite -c 2 \
  --no-auto-eval
```

That's the whole flow. claude-code (inside the task container) issues
`POST /v1/messages` to your vLLM, and vLLM responds in Anthropic format
directly.

### What each flag does

- `--base-url` — vLLM endpoint. Bare host or host+`/v1`; claude-code
  appends `/v1/messages` itself. Auto-rewritten to
  `host.docker.internal` for container reachability when it's a local URL.
- `--auth-token` — placeholder for vLLM (no auth needed); claude-code
  requires *some* credential env var to start.
- `-m Qwen/Qwen3.5-9B` — model name sent to vLLM. Must match
  vLLM's `--served-model-name`. The substring `qwen` (case-insensitive)
  is also how the adapter's `_MODEL_PROFILES` picks the small-context
  profile (tighter Read/MCP budgets + stripped tool surface).
- `-a claude_code` — selects the Claude Code adapter.

## What the adapter does for you

When `--base-url` is set, `src/cooperbench/agents/claude_code/adapter.py`:

1. Forwards `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` into the task
   container, rewriting `localhost` / `127.0.0.1` →
   `host.docker.internal` so the container can reach a host-side endpoint.
2. Adds `--add-host=host.docker.internal:host-gateway` to make that
   rewrite resolve.
3. Preserves the model name verbatim (no provider-prefix strip — vLLM
   controls naming via `--served-model-name`).
4. Injects a placeholder auth token if `--base-url` is set without one
   (claude-code refuses to start without a credential env var).
5. Writes `~/.claude/settings.json` inside the container with
   `CLAUDE_CODE_ATTRIBUTION_HEADER=0` — that header otherwise busts the
   KV cache on vLLM/llama.cpp backends (~90% slowdown).
6. Looks up the model name (case-insensitive substring) in
   `_MODEL_PROFILES`. For `qwen`, applies:
   - `max_output_tokens=4096`
   - `file_read_max_tokens=4000`
   - `mcp_max_output_tokens=2000`
   - `disallowed_tools=SMALL_CONTEXT_DISALLOWED_TOOLS`

Real Anthropic runs (i.e. no `--base-url`) are unaffected by any of this.

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

The key matches as a case-insensitive substring against `-m`. Cut a
release after merging.

## Inspecting a run

```
logs/<run-name>/coop/<repo>/<task>/<features>/
├── agent1_traj.json          # parsed trajectory + status + cost
├── agent2_traj.json
├── agent{N}.patch            # diff each agent produced (N = feature_id)
├── agent1_stream.jsonl       # raw claude-code stream events
├── agent2_stream.jsonl
├── agent1_session.jsonl      # session JSONL (tool calls, messages)
├── agent2_session.jsonl
├── agent1_sent.jsonl         # per-agent coop messaging log
├── agent2_sent.jsonl
├── conversation.json         # combined inter-agent messages
└── result.json               # both agents' summary
```

The `*_session.jsonl` files are the most useful — one JSON line per
tool call, tool result, or assistant message.
