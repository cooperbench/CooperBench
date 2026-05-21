# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Team-mode primitives moved to `cooperbench.team_harness`** — extracted from `cooperbench/agents/_team` (private) to `cooperbench/team_harness` (public, library-shaped) so other benchmarks can consume the coordination algorithm without depending on CooperBench's task layout.  Adds a `TeamSession` facade that bundles per-run state (run_id, Redis URL, agents, scratchpad volume) and exposes adapter-facing factories (`env_for`, `scratchpad_mount_args`, `mcp_config`, `prompt_for`, `prompt_section`, `loop_poller`, `task_list_client`, `harvest_metrics`).  Adapters consume the session instead of calling loose helpers; the runner constructs one per run.

### Added

- **Ablation flags for team mode** — five `--team-no-*` CLI flags (`--team-no-task-list`, `--team-no-scratchpad`, `--team-no-mcp`, `--team-no-auto-refresh`, `--team-no-protocol`) gate each coordination feature independently.  Each flag flips one boolean on `TeamHarnessConfig`; the session's factory methods return `None`/`[]`/`{}` for disabled features so adapter blocks skip cleanly.  The lead/member role split stays on as the always-on baseline (without it team collapses to coop).  `result.json` now surfaces `team_features: {...}` so post-hoc analysis can attribute pass-rate deltas to the specific feature that was off.  End-to-end smoke on `dottxt_ai_outlines/1371 [1,2]` with codex: default writes `task_log.json`+`tasks.json`+metrics; `--team-no-task-list --team-no-scratchpad --team-no-mcp` drops all three and produces no `cb-team-*` Docker volume.

### Fixed

- **`swe_agent` now supports `--backend docker` too** — adapter was hardcoded to `swerex.ModalDeploymentConfig`.  Added a backend dispatch that picks `DockerDeploymentConfig` when `config["backend"] == "docker"`.  Required two follow-on fixes: (a) `docker_args=["--entrypoint", ""]` to clear the task image's `ENTRYPOINT=runner.sh` (otherwise swerex's `sh -c "..."` becomes a `runner.sh sh -c "..."` and the first positional arg gets interpreted as a feature-patch path); (b) monkey-patch swerex's `DockerDeployment._get_swerex_start_cmd` to invoke `pipx run --spec swe-rex swerex-remote ...` instead of `pipx run swe-rex ...` — upstream swerex assumes the package's executable matches the package name, but the published `swe-rex` package provides an executable named `swerex-remote`.  Smoke-tested with `dottxt_ai_outlines/1655 [1,3]` solo: 1/1 pass in 2m 53s.
- **codex `exec` hung indefinitely in Modal sandbox** — codex's `exec` mode tries to read additional input from stdin and blocks until EOF.  Docker's non-tty `docker exec` gives it EOF immediately, but Modal sandbox `exec` keeps stdin open, so codex sat for the full sandbox lifetime (~2h) printing "Reading additional input from stdin..." and producing zero stream output.  Fix: add `</dev/null` to the codex invocation in `codex/adapter.py::_build_codex_command`.  Smoke-tested with a solo run on `dottxt_ai_outlines/1655 [1,3]`: 1/1 pass in 1m 48s.
- **`cooperbench eval` now refuses to eval an openhands_sdk run with a non-modal backend** — openhands_sdk produces patches that include a committed `patch.txt` artifact in the working tree and relies on Modal-hosted Redis for inter-agent coordination; running the eval through Docker silently changed the test environment.  The eval now reads the run's `config.json`, detects `agent_framework == "openhands_sdk"`, and exits with a clear warning if `--backend != modal`.
- **`normalize_patch` was eating trailing blank context lines** — the helper used `text.strip()` to collapse trailing whitespace before re-appending a single newline.  A unified-diff blank-context line is encoded as `" \n"` (one space + LF), which `strip()` consumes along with the terminator.  Result: the last hunk's body became one (or two, for consecutive blank lines) shorter than its `@@ -L,N @@` header claimed, and `git apply` rejected it.  In a 10-task team-mode run on `claude_code` this manifested as every "merge conflicts / both features fail" task — the malformed patches couldn't 3-way-merge cleanly even though they applied.  Now uses `text.lstrip("\n").rstrip("\n") + "\n"` so blank context lines are preserved.
- **`mini_swe_agent_v2` adapter wasn't normalizing patches at all** — `adapter.py:252` did `(r.get("output") or "").strip()` and returned the raw stripped string.  Every msa patch on disk ended in a non-newline byte, so `git apply` rejected them outright (msa accidentally hid behind the same blank-context bug as `normalize_patch`, just one layer deeper).  Now routes through the fixed `normalize_patch` helper, matching `claude_code` and `codex`.
- **`mini_swe_agent_v2` Modal sandbox died on first exec** — `environments/modal.py::_start_sandbox` called `modal.Sandbox.create(image=..., ...)` with no command.  Combined with `image.entrypoint([])` clearing the image's entrypoint, the sandbox ran the image's `CMD` (e.g. `/bin/bash`) which exited immediately on stdin-less startup, so the next `exec()` hit "Sandbox not found".  Across a 10-task run every agent failed at step 1 (~$0.015 each) with 260 retried-sandbox-died errors.  Now passes `"sleep", "infinity"` as the positional command, matching the eval backend's existing fix.
- **`claude_code` and `codex` adapters silently ignored `--backend modal`** — both call shared `_coop.runtime.build_environment`, which was hardcoded to `DockerEnvironment` regardless of the `config["backend"]` the runner threaded through.  At concurrency=10 in team mode this surfaced as ~20 simultaneous `docker run` attempts on the host, several of which hit the 120s startup timeout (~half of `cx` member agents died at container creation).  `build_environment` now takes a `backend` kwarg; both adapters pass `config.get("backend", "docker")`.

### Changed

- **Stronger team-mode lead prompt** — `_team/prompt.py::_lead_block` previously buried the integration step at the bottom of a long workflow list, and Claude/Codex consistently exited after their own feature without ever reading `/workspace/shared/<agent>.patch`.  Replaced with an opening hard-rule ("the integration step is the WHOLE point of your job — DO NOT SKIP IT") and a 5-point pre-submission checklist (tasks done → member patches present → patches applied → tree compiles → both features visible in `git diff`).  Also asks the lead to drop a `PLAN.md` upfront that divides the file regions per feature.  Member block now opens with "stay in your lane" and points at that plan.
- **Eval `test_merged` policy: `identical → naive → lead-when-naive-conflicts`**.  The previous chain tried `identical → naive → union → solo-fallback`; both `union` and the member-fallback were dropped.  Rationale: union merge concatenates conflicting hunks, which usually produces syntactically broken code and rewards lucky non-overlap rather than real coordination.  The lead is the team's designated integrator; if naive merge conflicts, the lead's `patch.txt` alone is the team's deliverable, and it must pass both feature suites unaided.  Member fallback removed for the same reason — if the lead didn't integrate, the team's coordination failed regardless of what the member produced.  Surfaced in `eval.json` as `merge.strategy = "solo-agent1"` when the lead-alone path succeeded.

### Added

- **`dataset/subsets/core.json`** — 10-pair core subset for quick agent comparisons.  Stratified sampling: largest-remainder proportional allocation by repo's full-dataset pair count, with a one-slot floor per primary language (Python / Go / Rust / TS).  Within each repo, pairs are sampled by spreading across tasks before any repeat.  Reproducible via `python scripts/generate_core_subset.py` (seed=42).  Pass-rate on `core` tracks the overall dataset's shape without per-task eval data.
- **`docs/BENCHMARK_RESULTS.md`** — horizontal team-mode comparison of `claude_code` / `codex` / `mini_swe_agent_v2` / `openhands_sdk` on the `core` subset.  Per-task pass/fail matrix with merge strategy used, framework totals (`msa` 6/10, `oh` 5/10, `cc` 5/10, `cx` 5/10), and a narrative of the reruns that surfaced the bugs in the unreleased Fixed/Changed sections.

## [0.0.15] - 2026-05-18

### Added

- **Claude Code adapter** (`cooperbench.agents.claude_code`).  Wraps the
  Claude Code CLI inside the task's Docker container with coop messaging
  and git-collaboration support.  Authenticates via the host's
  `ANTHROPIC_API_KEY`, registers the shared `coop-*` shell wrappers, and
  participates in solo / coop / coop+git / team runs alongside the
  existing `mini_swe_agent_v2`, `swe_agent`, and `openhands_sdk`
  adapters.

- **Codex adapter** (`cooperbench.agents.codex`).  Wraps the OpenAI
  Codex CLI (`codex exec --json --sandbox danger-full-access
  --skip-git-repo-check`).  Writes `${CODEX_HOME}/auth.json` inside the
  container so the CLI authenticates without prompts, parses Codex's
  JSONL event stream for status / token totals / messages, and reports
  cost as `0.0` because Codex does not emit a cost field.  Includes a
  one-shot model-name fallback (retries without `--model` if Codex
  rejects the requested model) and a preflight `OPENAI_API_KEY` check.

- **`team` setting** alongside `solo` and `coop`.  N agents organized
  as one lead + N-1 members, with a Redis-backed shared task list
  (atomic claim via `coop-task-claim`), a shared scratchpad volume
  mounted at `/workspace/shared`, and role-specific prompt blocks.
  All five adapters (`mini_swe_agent_v2`, `swe_agent`, `openhands_sdk`,
  `claude_code`, `codex`) accept the new `team_role` / `team_id` /
  `task_list_url` kwargs; CLI adapters install `coop-task-create` /
  `coop-task-claim` / `coop-task-update` / `coop-task-list` shell
  wrappers next to the existing `coop-*` messaging tools.  Post-run,
  the task-list audit log is used to compute coordination metrics
  (`time_to_first_claim_seconds`, `claims_per_agent`,
  `updates_per_agent`, `tasks_done`, `unowned_at_end`) saved in
  `result.json`.

- **Team-mode filesystem mirror** of the task list at
  `/workspace/shared/tasks/`.  One `<id>.json` per task plus
  `_index.json` (cheap `ls` target) and `_log.jsonl` (audit trail),
  written via tempfile+replace so readers never observe partial state.
  Lets agents `ls` and `cat` tasks with their existing tools instead of
  going through the `coop-task-list` CLI.

- **Typed `coop-request` / `coop-respond` protocol**
  (`cooperbench.agents._team.protocol`) layered on plain Redis
  messaging.  `coop-request <peer> <kind> <body>` returns a request_id
  and optionally blocks via `--wait N`; `coop-respond <request_id>
  <body>` writes back.  The sender's `await_response` uses BLPOP so it
  sleeps instead of busy-polling, and both events flow into the shared
  task-log so coordination metrics include protocol events.

- **MCP long-poll server** (`cooperbench.agents._team.mcp_server`).
  Stdio JSON-RPC server exposing a single `wait_for_message` tool
  backed by BLPOP on the agent's inbox.  Registered automatically for
  CLI adapters: Claude Code writes to `$CLAUDE_CONFIG_DIR/.claude.json`,
  Codex writes to `$CODEX_HOME/config.toml`.  Gives opaque CLI agent
  loops the closest thing to push-style inbox delivery.

- **In-loop task-list auto-refresh** for `mini_swe_agent_v2`
  (`cooperbench.agents._team.loop_refresh`).  `TeamPoller` runs between
  LLM queries — same hook as the existing inbox poll — and prepends a
  compact `[Team task list] open: 1, in_progress: 2, ...` summary so
  the LLM doesn't need to remember to call `coop-task-list`.

- **Shared `cooperbench.agents._coop` module** holding the
  agent-agnostic coop primitives that the CLI adapters previously
  duplicated: `coop_msg.py` (Redis-backed messaging CLI installed as
  `coop-send` / `coop-recv` / `coop-broadcast` / `coop-peek` /
  `coop-agents`), `prompt.py` (solo / coop / coop+git prompt assembly),
  `runtime.py` (`ContainerEnv` protocol, environment assembly,
  in-container file I/O helpers, git-setup command construction,
  sent-messages log parsing, and `normalize_patch`), and
  `install_snippet.sh` (sourced from each adapter's `setup.sh`).

### Fixed

- **`mini_swe_agent_v2`: prevent re-exploration loops after compaction**
  (PR #54).  The default compaction summarizer was producing terse,
  generic summaries that lost the file contents agents had already
  read.  After compaction the agent would re-cat/grep the same files,
  hit the 100-step limit, and fail to submit — in one full coop run,
  91% of agents (63/72 sampled trajectories) hit `LimitsExceeded` with
  mean ~99 of 100 steps used.  Two targeted changes: (1) `summarize_context`
  now serializes prior turns into a single user message as a tagged
  transcript, preventing the model from role-playing as the next
  assistant turn; (2) the default `compaction_summary_prompt` is now
  a structured template with explicit headings (FILE MAP, RELEVANT CODE
  READ, KEY SYMBOLS, SEARCH RESULTS, EDITS, BUILD/TEST OUTPUT, COLLEAGUE
  MESSAGES, OPEN QUESTIONS, CURRENT PLAN), asks for verbatim quoting
  with `file:line` citations, and includes an anti-hallucination guard
  for line counts.  Measured on an A/B replay of 9 real solver segments,
  coverage of post-compaction re-explores went from 40% to 57% on a
  heuristic file+symbol scorer.

- **`runner/coop`: coerce message timestamps to float before sorting**
  (PR #49).  `execute_coop` crashed mid-rollout with `TypeError: '<'
  not supported between instances of 'int' and 'str'` when one agent
  adapter reported numeric timestamps (`mini_swe_agent` uses
  `time.time()` floats) and another reported ISO strings (OpenHands
  SDK).  The sort fired before `agent{fid}_traj.json` was written, so
  callers relying on the structured trajectory got nothing even though
  the rollout completed.  Added `_message_timestamp_key` that
  best-effort coerces to float (None / non-numeric → 0.0).

- **CLI adapter patch normalization** (PR #51).  The previous Claude
  Code adapter's `.strip()` on `patch.txt` was eating the trailing
  newline that `git apply` requires, producing "corrupt patch at line
  N" errors.  Replaced with `normalize_patch()` (one trailing newline,
  no leading whitespace) in the shared `_coop` module.

- **Team-mode prompt: explicit `patch.txt` submission step**.  The
  initial team-mode end-to-end had members writing diffs to
  `/workspace/shared/<id>.patch` only and never to
  `/workspace/repo/patch.txt`, scoring 0/2 despite great coordination.
  Both lead and member prompts now have an explicit
  `### Final submission — REQUIRED` section that names `patch.txt` as
  the only file the bench evaluates and provides the exact
  `git diff > patch.txt` command.  Verified by a follow-up run on the
  same task that scored 2/2.

- **Result-table rendering for team mode**.  Cosmetic fix to
  `runner/core._print_single_result` — team mode's per-agent dicts
  carry `patch_lines: int`, but the previous code tried
  `len(r.get("patch", "").splitlines())` and showed 0.

### Changed

- **`fakeredis` is now a dev dependency** (and `uv.lock` is regenerated
  to match).  Six team-mode test modules import `fakeredis` to mock
  Redis; without it pinned, CI's `uv pip install -e ".[dev]"` left it
  missing and pytest collection failed.

## [0.0.14] - 2026-04-30

### Changed

- **`mini_swe_agent_v2` patch is now read directly from `patch.txt` in the agent's container.**  Previously the adapter parsed the patch out of the agent's stdout (whatever followed the `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` sentinel).  In a real coop+git run with GPT-5.4 against the v0.0.13 prompts, ~50% of agents still emitted the bare sentinel without `&& cat patch.txt` — but **all 18/18 agents still wrote a `patch.txt` file** in their working directory.  Reading the file directly via `docker exec cat patch.txt` after `agent.run()` returns is deterministic regardless of which submit-command variant the agent picked.  Submit step in `coop.yaml` / `solo.yaml` simplified to the bare `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (still framed as `EXACT command required` so models follow it consistently).  No fallback to stdout submission text — `patch.txt` is the only source of truth.

## [0.0.13] - 2026-04-30

### Fixed

- **`mini_swe_agent_v2` Submit step now uses upstream's prescriptive wording.**  In a real coop+git run with GPT-5.4 against the v0.0.12 prompts, ~50% of agents reverted to the bare `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (without `&& cat patch.txt`), producing empty patches even when they had edited files.  Strong-prior models trained on upstream mini-swe-agent's older `swebench.yaml` recognise that pattern from training and override the prompt.  Restore upstream's exact framing — `Submit (EXACT command required) / You MUST use this EXACT command to submit:` — so the prompt reads as prescriptive rather than as one example among many.

## [0.0.12] - 2026-04-30

### Changed

- **`mini_swe_agent_v2` patch is now the agent's `submission`** — the adapter no longer captures `base_commit` and runs `git diff <base>` at end-of-run. Instead the patch comes directly from `result['submission']`, which the env populates with everything the agent emits after `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Mirrors upstream mini-swe-agent's SWE-bench config. The `coop.yaml` / `solo.yaml` prompts now instruct the agent to curate via `git diff -- file1 file2 > patch.txt`, verify with `cat patch.txt`, and submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`. No working-tree-extraction fallback — if the agent didn't submit, there is no patch.
- **`config/mini.yaml` split into `config/solo.yaml` + `config/coop.yaml`** — the previous single file conditioned everything on `{% if agent_id %}` blocks. The adapter now selects which file to load via `is_coop = len(agents) > 1`. While splitting, fixed a leak in the solo branch where the `CRITICAL REQUIREMENTS` section still mentioned `send_message to your colleague` for an agent with no colleague.
- **Shared singleton git server for `--git` coop runs** — replaces the previous design that spun up a fresh `debian:bookworm-slim` container per run, ran `apt-get install git`, slept 3s, and returned (resulting in race conditions where agents' initial `git push` beat the daemon to startup). The new design auto-creates one image (`cooperbench-git-server:local`), one network (`cooperbench`), and one container (`cooperbench-git`) on first use; per-run isolation comes from path namespacing under `/git/<run_id>/repo.git`. Idempotent — first run pays a ~30s image-build cost, subsequent runs reuse the singleton in ~140ms. Mirrors the Redis-style "one daemon, many namespaces" pattern.
- **Submission prompts simplified + `.git` footgun warnings** — the `## Submission` section in `coop.yaml` / `solo.yaml` is now ~5 lines (write a `git diff` to `patch.txt`, `cat` it, submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`).  Adds an explicit `<CRITICAL>` block forbidding `rm -rf .git`, `git init`, `git rm -rf .`, and `git reset --hard` inside `/workspace/repo` — these are easy footguns for small models, observed in the wild causing patches to come out as malformed `new file mode` diffs that fail to apply.

### Fixed

- **Eval surfaces `git apply` failures instead of silently masking them.**  `_setup_branches` now emits explicit `PATCH<N>_APPLIED` / `_SKIPPED` / `_FAILED` markers per agent and returns an `apply_status` dict.  `test_merged` writes that to the result and refuses to call merge `clean` when any input patch failed to apply — instead reporting `merge.status: "missing_input"`.  Previously, an agent submitting a malformed patch (e.g. a `new file mode` diff against an existing file) would have its branch silently end up empty, and the subsequent merge against the other agent's branch would report `clean` despite the missing input — making the eval lie about partial success.
- **Per-feature eval result schema enriched.**  `feature1` / `feature2` now carry `feature_id`, `exit_code`, `tests_passed`, and `tests_failed` (was just `passed: bool` + a 50KB `test_output` blob).  Lets consumers reason about results without grepping raw pytest output.

- **`mini_swe_agent_v2` adapter no longer crashes on `content=None`** — tool-calling assistant turns leave `content=None` (the body lives in `tool_calls`), and CooperBench's downstream `_extract_conversation` does `"send_message" in content`, which raises `TypeError` on None. The adapter now coerces to `""` before populating `AgentResult.messages`.
- **`mini_swe_agent_v2` adapter wires up the `agent_config` flag** — previously listed in `MiniSweAgentV2Runner.run`'s signature but never read from. Now loads the YAML and deep-merges its `config:` block over the defaults. Forward-compatible: `**kwargs` accepted so unknown caller-side args don't crash `run()`.
- **`mini_swe_agent_v2` adapter drops the dead `SEND_MESSAGE_TOOL` import** — only `BASH_TOOL` is registered with the model; `send_message` is intercepted from inside the bash command string. The leftover import was confusing.
- **`DockerEnvironmentConfig.network`** — added a typed `network` field so the `--network <name>` flag reaches `docker run`. Previously the adapter passed `network=...` as a kwarg, but Pydantic silently dropped it (no such field), and agent containers ended up on the default bridge with no route to the per-run git server's IP. With the new shared-singleton git server design, agent containers must join the shared `cooperbench` network for DNS-by-name resolution to work.
- **`DefaultAgent.serialize()` no longer mutates `_segments`** — `run()` calls `save()` in its finally clause every step, and once compaction had fired, each `serialize()` call appended another snapshot of the current live messages as a fresh solver segment (and reset the buffer, which the next `query()` repopulated). Net effect: one compaction produced N+1 overlapping post-compaction solver segments instead of 1. Fix: `serialize()` builds the snapshot list locally without touching `self._segments`.

### Added

- **`cooperbench` CLI auto-loads `./.env`** — `cli.py` now calls `dotenv.load_dotenv()` at module load, so project-local `OPENAI_API_KEY` etc. is picked up without users having to `set -a && source .env` ahead of every invocation. Matches the convention used elsewhere in the codebase.

## [0.0.11] - 2026-04-18

### Added

- **Context compaction (summarization) for `mini_swe_agent_v2`** — long-running agents no longer exhaust the model context window. When the previous LLM response's `prompt_tokens` reaches `compaction_token_trigger` (default `28000`), the agent calls the model a second time (without tools) to summarize old turns and replaces the live history with `[system, task, summary] + recent_turns` (last `compaction_keep_recent_turns=2` assistant turns kept verbatim). Repeated compactions naturally fold the previous summary back into `old_turns`. Configurable via `agent.compaction_enabled` / `compaction_token_trigger` / `compaction_keep_recent_turns` / `compaction_summary_prompt`; **enabled by default**.
- **Full-trajectory artifact** — when any compaction occurs, the adapter writes `{log_dir}/{agent_id}_full_traj.json` containing a `segments` list (alternating `solver` / `summarizer` blocks) so the unabridged pre-compaction history is preserved for analysis even though the live `messages` list has been shortened.
- **`LitellmModel.summarize_context(messages, summary_prompt)`** — separate completion call (no tools) used by the agent's compaction step; tracks cost via `GLOBAL_MODEL_STATS` and tags the resulting message with `extra.summary=True`.

### Changed

- **`mini_swe_agent_v2` adapter config merge** — switched from shallow `dict.update` to `recursive_merge`, so partial overrides (e.g. only `agent.compaction_enabled`) no longer clobber sibling keys from the default YAML.

## [0.0.10] - 2026-04-18

### Fixed

- **Docker eval backend entrypoint handling** - `eval/backends/docker.py` started its sandbox container with `command="sleep infinity"` but no `entrypoint` override, so images that set an `ENTRYPOINT` (e.g. the benchmark dataset images with `/usr/local/bin/runner.sh`) would consume `sleep infinity` as entrypoint arguments and exit immediately. Every subsequent `docker exec` then hit "container is not running". Clear the entrypoint on startup (`entrypoint=""`) so `sleep infinity` runs as PID 1; `runner.sh` is still invoked explicitly via `docker exec` during evaluation. Matches the Modal and GCP eval backends and completes the fix started in 0.0.9 for the agent-side environments.

## [0.0.9] - 2026-04-17

### Fixed

- **Docker backend entrypoint handling for `mini_swe_agent` and `mini_swe_agent_v2`** - Containers whose images set an `ENTRYPOINT` (e.g. the benchmark dataset images that use `/usr/local/bin/runner.sh` as their entrypoint) were exiting immediately because `sleep infinity` / `sleep <timeout>` was passed as arguments to the entrypoint rather than as the container command. Both docker environments now explicitly clear the entrypoint (`--entrypoint ""` / `/bin/bash -c`), matching the behaviour of the Modal and GCP backends.

## [0.0.8] - 2026-04-17

### Fixed

- **`git index.lock` in coop eval** - Stale lock file left by `runner.sh` after the first feature test no longer blocks the second feature's `git checkout`/`reset` in `_run_tests`

## [0.0.7] - 2026-04-17

### Changed

- **Docker is now the default backend** for both `cooperbench run` and `cooperbench eval`, as well as every helper API (`evaluate`, `_evaluate_single`, `run_patch_test`, `test_merged`, `test_solo`, `get_backend`, `get_environment`, agent adapter config fallbacks, and the coop/solo runners). Previously defaulted to `modal`.

### Fixed

- **Auto-eval now honours `--backend`** - `cooperbench run --backend <X>` no longer silently falls back to modal during the inline evaluation phase; the value is threaded through both the single-task and multi-task auto-eval paths (`runner/core.py`).

## [0.0.6] - 2026-04-17

### Added

- **`cooperbench prepare`** - New CLI subcommand that downloads the benchmark dataset from HuggingFace (`CodeConflict/cooperbench-dataset`) into `./dataset`, so PyPI users don't need to clone the GitHub repo
- **`scripts/upload_dataset_to_hf.py`** - Maintainer script to sync the local `dataset/` tree up to the HuggingFace dataset repo

### Changed

- **README** - Replaced `git clone` dataset instructions with `cooperbench prepare`; fixed stale HuggingFace URL
- Added `huggingface-hub>=0.24` as a core dependency

## [0.0.5] - 2026-02-14

### Added

- **mini_swe_agent_v2** - New agent framework with improved tool-call based architecture, litellm model integration, cache control, multimodal support, and retry logic

### Changed

- **Python 3.10 support** - Lowered minimum Python version from 3.12 to 3.10, replacing `typing.Self`, `typing.override`, and PEP 695 type aliases with `typing_extensions` equivalents
- **Removed `browser-use` dependency** - Dropped from both root and vendored openhands-tools dependencies
- **Removed `openhands-agent-server` dependency** - Dropped unused dependency from vendored openhands-workspace
- **Fixed lint/type errors** - Resolved ruff F401 unused import and mypy type error in mini_swe_agent_v2

## [0.0.4] - 2026-02-14

### Added

- **Token usage tracking** - `AgentResult` now reports `input_tokens`, `output_tokens`, `cache_read_tokens`, and `cache_write_tokens` throughout the pipeline
- **Fallback cost calculator** - New `pricing.py` module computes cost from token counts when litellm doesn't report it, with manual pricing table for custom endpoints
- **Log directory passthrough** - Runners now pass `log_dir` path to agents for downstream logging (PR #32)
- **Eval stats in summary** - Run summary JSON now includes pass rate and per-task eval results when auto-eval is enabled
- **Gold conflict checker** - New `scripts/check_gold_conflicts.py` to detect merge conflicts between gold patches across all tasks using parallel Modal sandboxes
- **Benchmark runner script** - New `scripts/run_benchmark.sh` for quick experiment launches
- **Model smoke test** - New `scripts/test_model.py` to verify models work via LiteLLM

### Changed

- **Improved cooperation prompt** - Replaced situational-awareness prompt with explicit numbered workflow (plan → coordinate → summarize) after A/B testing showed better coordination and lower cost
- **OpenHands SDK dependencies promoted to core** - Moved from `[openhands]` optional extra into base dependencies for simpler installation
- **HTTP retry logic** - Remote conversation requests now retry on 5xx errors with exponential backoff (via tenacity)
- **Patch extraction timing** - Patches are now extracted while the sandbox is still alive, before stats collection

### Fixed

- **Pricing calculation** - Fixed cost reporting for models where litellm returns zero cost
- **MaxIterationsReached handling** - Now caught inside the conversation loop instead of as an outer exception, preventing lost patches
- **Custom API base URLs** - `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` now forwarded to sandboxes

## [0.0.3] - 2026-02-04

### Added

- **Agent SDK support** - New agent SDK framework with Modal support for sandboxed execution
- **Inter-agent messaging** - Added messaging capability between agents in cooperative settings
- **GCP Batch evaluator** - New GCP-based evaluator using Google Cloud Batch for scalable evaluation
- **GCP execution environment** - Added GCP VM support for agent execution
- **Docker-based Git server** - Local Git server running on Docker for coop mode collaboration
- **External agents support** - Support for external agents via environment variables and registry
- **Agent configuration** - CLI and runner now accept optional agent config path
- **Auto-eval feature** - Automatic evaluation after task completion
- **Interactive GCP configuration wizard** - Streamlined GCP setup with comprehensive documentation

### Changed

- Increased default max steps to 100
- Improved messaging prompts and fixed messaging bugs
- Consolidated GCP documentation into single comprehensive guide
- Updated dataset lite split
- Re-run tasks with Error status instead of skipping

### Fixed

- Git server configuration now properly passed to runners
- Fixed resource leaks on GCP and formatting of cwd path
- Docker timeout fixes
- Fixed skip errored tasks behavior
- Various linter fixes and test improvements

## [0.0.2] - 2026-01-31

### Changed

- **Complete architecture rewrite** - Replaced OpenHands-based execution with Modal sandboxes
- New agent framework: `mini_swe_agent` with tool-based interface
- Simplified CLI: `cooperbench run` and `cooperbench eval` commands
- Redis-based inter-agent messaging for cooperative settings
- Optional git collaboration for shared code changes

### Removed

- OpenHands Docker integration
- Planning phase (agents now plan and execute in single flow)
- `[llm]`, `[execution]`, `[serve]` optional dependencies
- Old Python API (`BenchSetting`, `FileInterface`, `create_plan`, `create_execution`)

### Added

- Modal sandbox execution environment
- `mini_swe_agent` framework with bash, file editing, and messaging tools
- Git connector for multi-agent code collaboration
- Comprehensive test suite

## [0.1.0] - 2026-01-15

### Added

- Initial release with OpenHands-based execution
- Planning and execution phases
- Support for single, solo, coop, and coop_ablation settings
- HuggingFace dataset integration
