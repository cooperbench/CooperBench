#!/usr/bin/env bash
# Build ALL OpenHands agent-server images for CooperBench tasks
#
# Usage:
#   ./build_all_images.sh           # Build all tasks
#   ./build_all_images.sh --dry-run # Show what would be built
#   ./build_all_images.sh --check   # Check which base images exist

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
CHECK_ONLY=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - showing what would be built"
    echo ""
elif [[ "$1" == "--check" ]]; then
    CHECK_ONLY=true
    echo "CHECK MODE - verifying base images exist"
    echo ""
fi

# All 30 CooperBench tasks
# Updated: 2026-02-03
BUILDS=(
    # dottxt_ai_outlines_task (3 images)
    "dottxt-ai-outlines:task1371"
    "dottxt-ai-outlines:task1655"
    "dottxt-ai-outlines:task1706"
    # dspy_task (4 images)
    "dspy:task8394"
    "dspy:task8563"
    "dspy:task8587"
    "dspy:task8635"
    # go_chi_task (3 images)
    "go-chi:task26"
    "go-chi:task27"
    "go-chi:task56"
    # huggingface_datasets_task (3 images)
    "huggingface-datasets:task3997"
    "huggingface-datasets:task6252"
    "huggingface-datasets:task7309"
    # llama_index_task (3 images)
    "llama-index:task17070"
    "llama-index:task17244"
    "llama-index:task18813"
    # openai_tiktoken_task (1 image)
    "openai-tiktoken:task0"
    # pallets_click_task (3 images)
    "pallets-click:task2068"
    "pallets-click:task2800"
    "pallets-click:task2956"
    # pallets_jinja_task (3 images)
    "pallets-jinja:task1465"
    "pallets-jinja:task1559"
    "pallets-jinja:task1621"
    # pillow_task (3 images)
    "pillow:task25"
    "pillow:task68"
    "pillow:task290"
    # react_hook_form_task (2 images)
    "react-hook-form:task85"
    "react-hook-form:task153"
    # samuelcolvin_dirty_equals_task (1 image)
    "samuelcolvin-dirty-equals:task43"
    # typst_task (1 image)
    "typst:task6554"
)

TOTAL=${#BUILDS[@]}
SUCCESS=0
FAILED=0
FAILED_LIST=()

echo "================================================"
if $CHECK_ONLY; then
    echo "Checking $TOTAL base images"
else
    echo "Building $TOTAL OpenHands agent images"
fi
echo "================================================"
echo ""

# Build each image
COUNT=0
for build in "${BUILDS[@]}"; do
    COUNT=$((COUNT + 1))
    repo="${build%%:*}"
    task="${build##*:}"
    base_image="akhatua/cooperbench-${repo}:${task}"
    
    if $CHECK_ONLY; then
        echo -n "[$COUNT/$TOTAL] $base_image ... "
        if docker manifest inspect "$base_image" > /dev/null 2>&1; then
            echo "✅ exists"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "❌ NOT FOUND"
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$repo:$task")
        fi
    elif $DRY_RUN; then
        echo "[$COUNT/$TOTAL] Building $repo:$task"
        echo "  Would run: ./build_agent_image.sh $repo $task"
        SUCCESS=$((SUCCESS + 1))
        echo ""
    else
        echo "[$COUNT/$TOTAL] Building $repo:$task"
        if "${SCRIPT_DIR}/build_agent_image.sh" "$repo" "$task"; then
            SUCCESS=$((SUCCESS + 1))
            echo "  ✅ Success"
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$repo:$task")
            echo "  ❌ Failed"
        fi
        echo ""
    fi
done

echo ""
echo "================================================"
echo "Summary"
echo "================================================"
echo "Total:   $TOTAL"
echo "Success: $SUCCESS"
echo "Failed:  $FAILED"

if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    echo ""
    echo "Failed:"
    for item in "${FAILED_LIST[@]}"; do
        echo "  - $item"
    done
fi

echo "================================================"
