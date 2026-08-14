#!/bin/bash

set -e

# Cleanup function
cleanup() {
    echo "CLEANING_UP: Restoring repository to original state..."
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        git reset --hard HEAD 2>/dev/null || true
        # No -x: keeps the image's prebuilt native extension.
        git clean -fd 2>/dev/null || true
        echo "CLEANUP: Repository restored to clean state"
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
    echo "APPLYING_FEATURE_PATCH: $FEATURE_PATCH"
    if [[ -f "/patches/$FEATURE_PATCH" ]]; then
        git apply --ignore-whitespace --ignore-space-change "/patches/$FEATURE_PATCH" || git apply --3way "/patches/$FEATURE_PATCH"
    else
        echo "Error: Feature patch not found at /patches/$FEATURE_PATCH"
        exit 1
    fi
fi

# Apply test patch
echo "APPLYING_TEST_PATCH: $TEST_PATCH"
if [[ -f "/patches/$TEST_PATCH" ]]; then
    git apply --ignore-whitespace --ignore-space-change "/patches/$TEST_PATCH" || git apply --3way "/patches/$TEST_PATCH"
else
    echo "Error: Test patch not found at /patches/$TEST_PATCH"
    exit 1
fi

echo "INSTALLING_DEPENDENCIES..."

# Install dependencies (SYSTEM PYTHON)
pip install -e .
pip install pytest pytest-xdist numpy

# Run test
echo "RUNNING_TESTS..."
python -m pytest "Tests/test_image_quantize.py" -v

echo "TEST_EXECUTION_COMPLETED"
