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

# Apply feature patch FIRST (before test patch) so test code can reference feature functions
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

# Analyze test patch for test functions
echo "Analyzing test patch for test functions..."
TEST_FUNCS=()
NEW_FUNCS=$(grep -o "func Test[a-zA-Z0-9_]*" "/patches/$TEST_PATCH" 2>/dev/null | sed 's/func //' || true)
for func in $NEW_FUNCS; do
    TEST_FUNCS+=("$func")
done

TEST_PATTERN=""
if [ ${#TEST_FUNCS[@]} -gt 0 ]; then
    # Anchored: `-run` takes an unanchored regex, so a bare `TestFoo` is also satisfied by an
    # agent-authored `TestFooExtra`. Anchoring makes the filter mean the functions the test
    # patch actually declares.
    TEST_PATTERN="^($(IFS="|"; echo "${TEST_FUNCS[*]}" | sort -u))$"
    echo "Found test functions to run: $TEST_PATTERN"
fi

# Run Go tests with timeout
echo "Running Go tests..."
if [ -n "$TEST_PATTERN" ]; then
    TEST_OUTPUT=$(timeout 300 go test -v -run "$TEST_PATTERN" ./... 2>&1)
    TEST_EXIT_CODE=$?
else
    TEST_OUTPUT=$(timeout 300 go test -v ./... 2>&1)
    TEST_EXIT_CODE=$?
fi
echo "$TEST_OUTPUT"

# A filter that selects nothing is a failed run, not a passing one. `go test` exits 0 and prints
# `ok <pkg> <time> [no tests to run]` when its -run pattern matches no test, which would let a
# feature score without any implementation at all.
if [ "$(echo "$TEST_OUTPUT" | grep -c '^=== RUN')" -eq 0 ]; then
    echo "Error: no test was executed. The filter '$TEST_PATTERN' matched nothing, which means the"
    echo "       graded tests are absent, excluded by a build constraint, or failed to compile."
    exit 1
fi

if [ "$TEST_EXIT_CODE" -ne 0 ]; then
    echo "Error: Go tests failed with exit code $TEST_EXIT_CODE"
    exit "$TEST_EXIT_CODE"
fi

echo "Test execution completed!"
