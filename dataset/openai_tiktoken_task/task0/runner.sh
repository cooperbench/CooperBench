#!/bin/bash

set -e

# Cleanup function
cleanup() {
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        git reset --hard HEAD 2>/dev/null || true
        # No -x: keeps the image's prebuilt native extension.
        git clean -fd 2>/dev/null || true
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

# Apply feature patch if provided
if [[ -n "$FEATURE_PATCH" ]]; then
    if [[ -f "/patches/$FEATURE_PATCH" ]]; then
        git apply --ignore-whitespace --ignore-space-change "/patches/$FEATURE_PATCH" || git apply --3way "/patches/$FEATURE_PATCH"
    else
        echo "Error: Feature patch not found at /patches/$FEATURE_PATCH"
        exit 1
    fi
fi

# Apply test patch
if [[ -f "/patches/$TEST_PATCH" ]]; then
    git apply --ignore-whitespace --ignore-space-change "/patches/$TEST_PATCH" || git apply --3way "/patches/$TEST_PATCH"
else
    echo "Error: Test patch not found at /patches/$TEST_PATCH"
    exit 1
fi

# Install dependencies (SYSTEM PYTHON)
pip install -e .
pip install pytest pytest-xdist regex hypothesis

# Discover test files matching test_feature*.py, fallback to all tests if none found
if compgen -G "tests/test_feature*.py" > /dev/null; then
    UNIT_TEST_SPEC="tests/test_feature*.py"
else
    UNIT_TEST_SPEC="tests/"
fi

# Run test
python -m pytest $UNIT_TEST_SPEC -v
