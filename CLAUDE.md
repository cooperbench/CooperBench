# CooperBench — Claude Code instructions

## Verify CI passes before pushing

Before any `git push`, verify the same checks GitHub Actions will run:

1. `uv run ruff check src/cooperbench/`
2. `uv run ruff format --check src/cooperbench/`
3. `uv run python -m mypy src/cooperbench/`
4. `uv run python -m pytest tests/ -v --tb=short` (Python 3.12 and 3.13 in CI)

If any fail, fix them before pushing. After pushing, prefer `gh run watch` to confirm CI is actually green — do not move on or open a PR until it is.

The workflows that gate this are `.github/workflows/lint.yml` and `.github/workflows/test.yml`. If those files change, update the checklist above.
