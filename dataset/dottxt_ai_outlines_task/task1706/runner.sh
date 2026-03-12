#!/bin/bash

set -e

# Cleanup function
cleanup() {
    echo "Cleaning up repository..."
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        git reset --hard HEAD 2>/dev/null || true
        git clean -fdx 2>/dev/null || true
        echo "Repository cleaned."
    fi
}

trap cleanup EXIT INT TERM

# Get input params
TEST_PATCH="$1"
FEATURE_PATCH="$2"

if [[ -z "$TEST_PATCH" ]]; then
    echo "Usage: docker run -v \$(pwd):/patches <image> <test_patch> [feature_patch]"
    exit 1
fi

cd /workspace/repo

# Ensure we start with a clean state
echo "Ensuring clean repository state..."
git reset --hard HEAD
git clean -fdx

# Apply feature patch if provided
if [[ -n "$FEATURE_PATCH" ]]; then
    echo "Applying feature patch: $FEATURE_PATCH"
    if [[ -f "/patches/$FEATURE_PATCH" ]]; then
        git apply --ignore-whitespace --ignore-space-change "/patches/$FEATURE_PATCH" || git apply --3way "/patches/$FEATURE_PATCH"
        echo "Feature patch applied successfully."
    else
        echo "Error: Feature patch not found at /patches/$FEATURE_PATCH"
        exit 1
    fi
fi

# Apply test patch
echo "Applying test patch: $TEST_PATCH"
if [[ -f "/patches/$TEST_PATCH" ]]; then
    git apply --ignore-whitespace --ignore-space-change "/patches/$TEST_PATCH" || git apply --3way "/patches/$TEST_PATCH"
    echo "Test patch applied successfully."
else
    echo "Error: Test patch not found at /patches/$TEST_PATCH"
    exit 1
fi

# Set up Python environment using uv (SYSTEM PYTHON)
echo "Setting up Python environment with uv..."
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export PANDAS_NO_USE_PYARROW=1
export LLAMA_CPP_FORCE_CPU=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Install dependencies
echo "Installing dependencies..."
# uv sync doesn't support --system flag, it manages project dependencies
# Skip it if it fails (dependencies are already installed in Dockerfile)
uv sync 2>/dev/null || true
uv pip install --system -e .
uv pip install --system pytest pytest-xdist pytest_mock pytest-asyncio pytest-benchmark pytest-cov
uv pip install --system torch "transformers<5" "tokenizers<0.21" sentencepiece xgrammar llama-cpp-python==0.3.16 psutil

# Run tests with timeout
echo "Running tests..."
# Run pytest - segfaults from llama-cpp-python during cleanup are non-fatal if tests pass
set +e  # Don't exit on error
timeout 300 python -m pytest tests/backends/test_xgrammar.py -v 2>&1 | tee /tmp/pytest_output.log
PYTEST_EXIT=${PIPESTATUS[0]}  # Get exit code of timeout/pytest, not tee
set -e  # Re-enable exit on error

# Check if tests passed in the output (even if there was a segfault)
if grep -qE "passed.*skipped|passed" /tmp/pytest_output.log && ! grep -qE "FAILED|ERROR" /tmp/pytest_output.log; then
    echo ""
    echo "Test execution completed!"
    # If exit code is 139 (segfault) but tests passed, exit with 0
    if [[ $PYTEST_EXIT -eq 139 ]]; then
        echo "Note: Segfault during cleanup (likely from llama-cpp-python) - tests passed successfully"
        exit 0
    elif [[ $PYTEST_EXIT -eq 0 ]]; then
        exit 0
    else
        echo "Tests may have issues, exit code: $PYTEST_EXIT"
        exit $PYTEST_EXIT
    fi
else
    echo "Tests failed - check output above"
    exit ${PYTEST_EXIT:-1}
fi

