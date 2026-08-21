#!/bin/bash

set -e

# Cleanup function (preserve target/ for cached builds)
cleanup() {
    echo "Cleaning up repository..."
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        git reset --hard HEAD 2>/dev/null || true
        git clean -fd 2>/dev/null || true  # No -x to preserve target/
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

# Ensure we start with a clean state (preserve target/ for cached builds)
echo "Ensuring clean repository state..."
git reset --hard HEAD
git clean -fd  # No -x to preserve target/

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

# Build typst
echo "Building Typst..."
cargo build --package typst --package typst-cli

# Run the tests this feature's patch actually declares.
#
# These filters used to be hardcoded to `string-first` and `string-last`, which are the block names
# feature 7 happens to use. Every other feature of this task declares `str-*` blocks — feature 8's
# are `str-case-parameter` / `str-case-invalid` — so for features 2,3,4,5,6,8,9,10 the graded tests
# were never executed at all, and the feature scored on whatever the base suite did.
echo "Deriving test filters from the test patch..."
BLOCKS=$(grep -oE '^\+--- [A-Za-z0-9_-]+ ---' "/patches/$TEST_PATCH" 2>/dev/null \
         | sed -E 's/^\+--- (.*) ---$/\1/' | sort -u || true)

FAILED=0
RAN_ANY=0

run_and_check() {
    # $1 = filter, or empty for the whole suite
    local filter="$1" out rc
    if [ -n "$filter" ]; then
        echo "Running tests..."
        out=$(cargo test -p typst-tests -- "$filter" 2>&1) && rc=0 || rc=$?
    else
        echo "Running the whole suite..."
        out=$(cargo test -p typst-tests 2>&1) && rc=0 || rc=$?
    fi
    echo "$out"
    # typst-tests prints "N passed, M failed, K skipped". A filter that selects nothing still exits
    # 0 with "0 passed, 0 failed" — that is a broken filter, not a pass.
    if echo "$out" | grep -qE '^[0-9]+ passed, [0-9]+ failed'; then
        local executed
        executed=$(echo "$out" | grep -oE '^[0-9]+ passed, [0-9]+ failed' \
                   | awk '{s+=$1+$3} END {print s+0}')
        [ "$executed" -gt 0 ] && RAN_ANY=1
    fi
    [ "$rc" -ne 0 ] && FAILED=1
    return 0
}

if [ -z "$BLOCKS" ]; then
    # Some feature patches add cases inside an existing block instead of declaring a new one
    # (feature 1 does). There is no filter that isolates those, so grade against the whole suite.
    echo "Test patch declares no new test blocks; grading against the full suite."
    run_and_check ""
else
    echo "Test blocks declared by this patch: $(echo "$BLOCKS" | tr '\n' ' ')"
    while IFS= read -r block; do
        [ -z "$block" ] && continue
        run_and_check "$block"
    done <<< "$BLOCKS"
fi

if [ "$RAN_ANY" -eq 0 ]; then
    echo "Error: no test was executed. The derived filters matched nothing, which means the graded"
    echo "       tests are absent or the block names in the test patch have changed."
    exit 1
fi

if [ "$FAILED" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi

echo "Test execution completed!"
