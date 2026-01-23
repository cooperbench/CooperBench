#!/bin/bash
# list_images.sh
#
# Lists all CooperBench Docker images that would be built/pushed.
# Useful for verifying image names before pushing.
#
# Usage:
#   ./scripts/list_images.sh

REGISTRY="akhatua"
PREFIX="cooperbench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")/dataset"

echo "CooperBench Docker Images"
echo "========================="
echo ""
echo "Registry: $REGISTRY"
echo "Prefix: $PREFIX"
echo ""
echo "Images:"
echo "-------"

cd "$DATASET_DIR"

count=0
for dockerfile in $(find . -name "Dockerfile" -type f | sort); do
    dir=$(dirname "$dockerfile")
    
    # Extract repo and task from path
    repo_task=$(echo "$dir" | sed 's|^\./||')
    repo=$(echo "$repo_task" | cut -d'/' -f1 | sed 's|_task$||')
    task=$(echo "$repo_task" | cut -d'/' -f2)
    repo_clean=$(echo "$repo" | tr '_' '-')
    
    image_name="${REGISTRY}/${PREFIX}-${repo_clean}:${task}"
    
    printf "  %-50s  <- %s\n" "$image_name" "$dir"
    ((count++))
done

echo ""
echo "Total: $count images"
