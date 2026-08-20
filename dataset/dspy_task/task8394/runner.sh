#!/bin/bash

set -e

# Cleanup function
cleanup() {
    echo "Cleaning up repository..."
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        CURRENT_BRANCH_OR_COMMIT=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null)
        echo "Resetting to HEAD ($CURRENT_BRANCH_OR_COMMIT) and cleaning..."
        git reset --hard HEAD || true
        git clean -fdx || true
        echo "Repository cleaned."
    else
        echo "Not inside a git repository, skipping git cleanup."
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

# Apply test patch with better error handling
echo "Applying test patch..."
if [[ -f "/patches/$TEST_PATCH" ]]; then
    if ! git apply --check "/patches/$TEST_PATCH" 2>/dev/null; then
        echo "Warning: Patch check failed. Attempting to apply anyway..."
    fi
    
    if ! git apply "/patches/$TEST_PATCH"; then
        echo "Error: Failed to apply test patch. Repository state may not match expected base commit."
        echo "Patch file: /patches/$TEST_PATCH"
        exit 1
    fi
    echo "Test patch applied successfully."
else
    echo "Error: Test patch not found at /patches/$TEST_PATCH"
    exit 1
fi

# Optionally apply feature patch
if [[ -n "$FEATURE_PATCH" ]]; then
    echo "Applying feature patch..."
    if [[ -f "/patches/$FEATURE_PATCH" ]]; then
        if ! git apply --check "/patches/$FEATURE_PATCH" 2>/dev/null; then
            echo "Warning: Feature patch check failed. Attempting to apply anyway..."
        fi
        if ! git apply "/patches/$FEATURE_PATCH"; then
            echo "Error: Failed to apply feature patch."
            echo "Patch file: /patches/$FEATURE_PATCH"
            exit 1
        fi
        echo "Feature patch applied successfully."
    else
        echo "Error: Feature patch not found at /patches/$FEATURE_PATCH"
        exit 1
    fi
fi

# Set up Python environment (SYSTEM PYTHON)
echo "Installing package in editable mode with dev dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Run tests with timeout and better error handling
echo "Running tests..."
# No --maxfail: stopping at the first failure left 2 of 14 tests unexecuted in observed runs, so
# the reported tests_failed count was a floor rather than a count. `timeout 300` already bounds it.
timeout 300 python -m pytest "tests/clients/test_cache.py" -v --tb=short

# Run whatever test files this feature's patch actually creates or modifies, in addition to the
# primary suite. Two failure modes this avoids, both observed:
#   * gating secondary tests on "$FEATURE_PATCH" means the base run never executes the feature's
#     own tests, so a feature scores as passing on an untouched tree;
#   * hardcoding a filename misses per-feature files (test_grounded_proposer1.py .. 5.py), so the
#     graded tests never run at all in either pass.
PATCH_TARGETS=$(grep -oE '^\+\+\+ b/.*' "/patches/$TEST_PATCH" 2>/dev/null | sed 's|^+++ b/||' | sort -u)
for file in $PATCH_TARGETS; do
    if [[ -e "$file" ]]; then
        echo "Running tests from the test patch: $file"
        timeout 300 python -m pytest "$file" -v || exit $?
    else
        echo "Error: test target named by the patch is missing: $file"
        exit 1
    fi
done

echo "Test execution completed!"
