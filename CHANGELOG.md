# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
