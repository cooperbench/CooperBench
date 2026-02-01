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

# Run tests (string-first and string-last tests)
echo "Running tests..."
cargo test -p typst-tests -- string-first
cargo test -p typst-tests -- string-last

echo "Test execution completed!"
