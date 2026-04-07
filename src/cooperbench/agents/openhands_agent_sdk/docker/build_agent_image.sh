#!/bin/bash
# Build OpenHands agent-server images for CooperBench tasks
#
# This script layers the OpenHands agent-server ON TOP of existing
# CooperBench task images. The task's original environment is preserved.
#
# Usage:
#   ./build_agent_image.sh <repo> <task>
#   ./build_agent_image.sh llama-index task17244
#   ./build_agent_image.sh dspy task8394
#   ./build_agent_image.sh typst task6554
#
# This will:
#   1. Use base image: akhatua/cooperbench-<repo>:<task>
#   2. Layer Python 3.12 + agent-server on top
#   3. Push to: akhatua/cooperbench-<repo>:<task>-oh

set -e

REPO="${1:?Usage: $0 <repo> <task>}"
TASK="${2:?Usage: $0 <repo> <task>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_SDK_DIR="$(dirname "${SCRIPT_DIR}")"  # Parent dir containing openhands-*
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.agent-server"
BUILD_CONTEXT="${SCRIPT_DIR}/.build-context"

# Docker Hub org
ORG="akhatua"

# Image names
BASE_IMAGE="${ORG}/cooperbench-${REPO}:${TASK}"
TARGET_IMAGE="${ORG}/cooperbench-${REPO}:${TASK}-oh"

echo "================================================"
echo "Building OpenHands agent-server image"
echo "================================================"
echo "Base image:   ${BASE_IMAGE}"
echo "Target image: ${TARGET_IMAGE}"
echo "Platform:     linux/amd64"
echo "SDK dir:      ${AGENT_SDK_DIR}"
echo "================================================"

# Create temporary build context with local OpenHands packages
echo "Preparing build context with local OpenHands packages..."
rm -rf "${BUILD_CONTEXT}"
mkdir -p "${BUILD_CONTEXT}"

# Copy Dockerfile
cp "${DOCKERFILE}" "${BUILD_CONTEXT}/Dockerfile"

# Copy local OpenHands packages
cp -r "${AGENT_SDK_DIR}/openhands-sdk" "${BUILD_CONTEXT}/openhands-sdk"
cp -r "${AGENT_SDK_DIR}/openhands-tools" "${BUILD_CONTEXT}/openhands-tools"
cp -r "${AGENT_SDK_DIR}/openhands-workspace" "${BUILD_CONTEXT}/openhands-workspace"

# Remove __pycache__ directories to keep image clean
find "${BUILD_CONTEXT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Ensure buildx is available and create builder if needed
if ! docker buildx inspect cooperbench-builder &>/dev/null; then
    echo "Creating buildx builder..."
    docker buildx create --name cooperbench-builder --use
fi
docker buildx use cooperbench-builder

# Build and push image (--no-cache ensures we get latest local packages)
# Note: Modal only supports linux/amd64, so we build for that platform only
echo "Building amd64 image..."
docker buildx build \
    --no-cache \
    --platform linux/amd64 \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${TARGET_IMAGE}" \
    -f "${BUILD_CONTEXT}/Dockerfile" \
    --push \
    "${BUILD_CONTEXT}"

# Cleanup build context
rm -rf "${BUILD_CONTEXT}"

# Clean up Docker to save disk space
echo "Cleaning up Docker to save disk space..."
docker buildx prune -f --filter "until=1h" 2>/dev/null || true
docker image prune -f 2>/dev/null || true

echo "================================================"
echo "Successfully built and pushed: ${TARGET_IMAGE}"
echo "================================================"
