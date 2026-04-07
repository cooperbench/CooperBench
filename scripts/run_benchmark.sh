#!/usr/bin/env bash
# run_benchmark.sh — Run a CooperBench benchmark experiment.
#
# Usage:
#   ./scripts/run_benchmark.sh MODEL
#
# Examples:
#   ./scripts/run_benchmark.sh "anthropic/MiniMax-M2.5"
#   ./scripts/run_benchmark.sh gpt-4o
#
# Extra cooperbench options can be appended after the model name:
#   ./scripts/run_benchmark.sh "anthropic/MiniMax-M2.5" -s lite --setting solo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

MODEL="${1:?Usage: $0 MODEL [extra cooperbench run options...]}"
shift

cooperbench run -m "$MODEL" -a openhands_sdk "$@"
