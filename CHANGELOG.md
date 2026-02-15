# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
