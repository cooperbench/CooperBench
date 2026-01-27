#!/bin/bash
# push_images.sh
#
# Builds and pushes all CooperBench Docker images to Docker Hub.
# Skips images that already exist in the registry.
# Runs builds in parallel (default: 5).
#
# Usage:
#   ./scripts/push_images.sh [--multi-arch] [--parallel N] [--force]
#
# Options:
#   --multi-arch    Build for both amd64 and arm64 (slower, requires buildx)
#   --parallel N    Number of parallel builds (default: 5)
#   --force         Rebuild even if image already exists in registry
#
# Prerequisites:
#   - Docker logged in: docker login
#   - Buildx enabled: docker buildx create --use (only for multi-arch)

set -e

REGISTRY="akhatua"
PREFIX="cooperbench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")/dataset"
LOG_DIR="/tmp/cooperbench_build_logs"

# Parse arguments
MULTI_ARCH=false
PARALLEL=5
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-arch)
            MULTI_ARCH=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Architecture: $(if $MULTI_ARCH; then echo 'multi (amd64+arm64)'; else echo 'current only'; fi)"
echo "  Parallel builds: $PARALLEL"
echo "  Force rebuild: $FORCE"
echo ""

# Verify Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Setup buildx for multi-arch if needed
if $MULTI_ARCH; then
    echo "Setting up Docker Buildx..."
    docker buildx create --name cooperbench-builder --use 2>/dev/null || docker buildx use cooperbench-builder
    docker buildx inspect --bootstrap
fi

# Create log directory
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# Check if image exists in registry (with correct architectures for multi-arch)
image_exists() {
    local image="$1"
    
    # Get the manifest
    local manifest
    manifest=$(docker manifest inspect "$image" 2>/dev/null) || return 1
    
    if $MULTI_ARCH; then
        # For multi-arch, verify BOTH amd64 and arm64 are present
        local has_amd64 has_arm64
        has_amd64=$(echo "$manifest" | grep -c '"architecture": "amd64"' || true)
        has_arm64=$(echo "$manifest" | grep -c '"architecture": "arm64"' || true)
        
        if [[ $has_amd64 -gt 0 && $has_arm64 -gt 0 ]]; then
            return 0  # Both architectures present
        else
            return 1  # Missing one or both architectures
        fi
    else
        # For single-arch, just check if manifest exists
        return 0
    fi
}

# Build and push a single image
build_and_push() {
    local dir="$1"
    local image_name="$2"
    local log_file="$3"
    
    {
        echo "=== Building: $image_name ==="
        echo "Directory: $dir"
        echo "Started: $(date)"
        echo ""
        
        if $MULTI_ARCH; then
            docker buildx build \
                --platform linux/amd64,linux/arm64 \
                -t "$image_name" \
                --push \
                "$dir" 2>&1
        else
            docker build -t "$image_name" "$dir" 2>&1
            docker push "$image_name" 2>&1
            # Clean up local image after successful push
            echo "Cleaning up local image..."
            docker rmi "$image_name" 2>&1 || true
        fi
        
        # Prune dangling images and build cache to save space
        echo "Pruning dangling images..."
        docker image prune -f 2>&1 || true
        
        echo ""
        echo "Finished: $(date)"
    } >> "$log_file" 2>&1
    
    return $?
}

cd "$DATASET_DIR"

# Collect all images
declare -a IMAGES
declare -a DIRS

for dockerfile in $(find . -name "Dockerfile" -type f | sort); do
    dir=$(dirname "$dockerfile")
    repo_task=$(echo "$dir" | sed 's|^\./||')
    repo=$(echo "$repo_task" | cut -d'/' -f1 | sed 's|_task$||')
    task=$(echo "$repo_task" | cut -d'/' -f2)
    repo_clean=$(echo "$repo" | tr '_' '-')
    image_name="${REGISTRY}/${PREFIX}-${repo_clean}:${task}"
    
    IMAGES+=("$image_name")
    DIRS+=("$dir")
done

total=${#IMAGES[@]}
echo "Found $total images"
echo ""

# Check which images need building
declare -a TO_BUILD_IMAGES
declare -a TO_BUILD_DIRS
skipped=0

echo "Checking registry for existing images..."
for i in "${!IMAGES[@]}"; do
    image="${IMAGES[$i]}"
    dir="${DIRS[$i]}"
    
    if ! $FORCE && image_exists "$image"; then
        echo "  [SKIP] $image (already exists)"
        ((skipped++))
    else
        TO_BUILD_IMAGES+=("$image")
        TO_BUILD_DIRS+=("$dir")
        echo "  [QUEUE] $image"
    fi
done

to_build=${#TO_BUILD_IMAGES[@]}
echo ""
echo "Skipped: $skipped (already in registry)"
echo "To build: $to_build"
echo ""

if [[ $to_build -eq 0 ]]; then
    echo "Nothing to build. All images already exist!"
    exit 0
fi

echo "Starting builds (logs in $LOG_DIR)..."
echo ""

# Track results
declare -a SUCCESS
declare -a FAILED
declare -a PIDS
declare -a PID_IMAGES

running=0
builds_since_cleanup=0
CLEANUP_INTERVAL=10  # Prune build cache every N builds

for i in "${!TO_BUILD_IMAGES[@]}"; do
    image="${TO_BUILD_IMAGES[$i]}"
    dir="${TO_BUILD_DIRS[$i]}"
    log_file="$LOG_DIR/$(echo "$image" | tr '/:' '_').log"
    
    # Wait if at parallel limit
    while [[ $running -ge $PARALLEL ]]; do
        # Check for any finished job
        for j in "${!PIDS[@]}"; do
            pid="${PIDS[$j]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished, check result
                wait "$pid" 2>/dev/null
                exit_code=$?
                img="${PID_IMAGES[$j]}"
                
                if [[ $exit_code -eq 0 ]]; then
                    SUCCESS+=("$img")
                    echo "[DONE] $img"
                    ((builds_since_cleanup++))
                else
                    FAILED+=("$img")
                    echo "[FAIL] $img (see log)"
                fi
                
                # Periodic build cache cleanup to prevent disk filling up
                if [[ $builds_since_cleanup -ge $CLEANUP_INTERVAL ]]; then
                    echo "[CLEANUP] Pruning build cache..."
                    docker builder prune -f --keep-storage=5GB >/dev/null 2>&1 || true
                    builds_since_cleanup=0
                fi
                
                unset 'PIDS[j]'
                unset 'PID_IMAGES[j]'
                ((running--))
                break
            fi
        done
        sleep 1
    done
    
    # Start new build
    echo "[START] $image"
    build_and_push "$dir" "$image" "$log_file" &
    pid=$!
    PIDS+=($pid)
    PID_IMAGES+=("$image")
    ((running++))
done

# Wait for remaining jobs
echo ""
echo "Waiting for remaining builds..."
for j in "${!PIDS[@]}"; do
    pid="${PIDS[$j]}"
    img="${PID_IMAGES[$j]}"
    wait "$pid" 2>/dev/null
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        SUCCESS+=("$img")
        echo "[DONE] $img"
    else
        FAILED+=("$img")
        echo "[FAIL] $img (see log)"
    fi
done

# Summary
echo ""
echo "=============================================="
echo "BUILD SUMMARY"
echo "=============================================="
echo "Total images: $total"
echo "Skipped (already exist): $skipped"
echo "Built successfully: ${#SUCCESS[@]}"
echo "Failed: ${#FAILED[@]}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "Failed images:"
    for img in "${FAILED[@]}"; do
        echo "  - $img"
        log_file="$LOG_DIR/$(echo "$img" | tr '/:' '_').log"
        echo "    Log: $log_file"
    done
fi

# Final cleanup
echo ""
echo "Final cleanup..."
docker image prune -f >/dev/null 2>&1 || true
docker builder prune -f --keep-storage=2GB >/dev/null 2>&1 || true
echo "Cleanup complete."

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi

echo ""
echo "All builds completed successfully!"
