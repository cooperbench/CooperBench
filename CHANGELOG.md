# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
