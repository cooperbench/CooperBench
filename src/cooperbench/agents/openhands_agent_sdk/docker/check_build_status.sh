#!/usr/bin/env bash
# Quick check of -oh image build status

echo "=== Build Progress ==="
if [ -f /tmp/build_remaining.log ]; then
    grep -E '^\[|✅|❌|Summary' /tmp/build_remaining.log 2>/dev/null || echo "No progress yet"
fi

echo ""
echo "=== Process Status ==="
if pgrep -f "build_remaining" > /dev/null; then
    echo "🔄 Build still running"
    tail -5 /tmp/build_remaining.log 2>/dev/null
else
    echo "✅ Build process finished"
fi

echo ""
echo "=== Docker Hub Verification ==="
built=0
missing=0
for build in \
    "dottxt-ai-outlines:task1371" "dottxt-ai-outlines:task1655" "dottxt-ai-outlines:task1706" \
    "dspy:task8394" "dspy:task8563" "dspy:task8587" "dspy:task8635" \
    "go-chi:task26" "go-chi:task27" "go-chi:task56" \
    "huggingface-datasets:task3997" "huggingface-datasets:task6252" "huggingface-datasets:task7309" \
    "llama-index:task17070" "llama-index:task17244" "llama-index:task18813" \
    "openai-tiktoken:task0" \
    "pallets-click:task2068" "pallets-click:task2800" "pallets-click:task2956" \
    "pallets-jinja:task1465" "pallets-jinja:task1559" "pallets-jinja:task1621" \
    "pillow:task25" "pillow:task68" "pillow:task290" \
    "react-hook-form:task85" "react-hook-form:task153" \
    "samuelcolvin-dirty-equals:task43" \
    "typst:task6554"; do
    repo="${build%%:*}"
    task="${build##*:}"
    image="akhatua/cooperbench-${repo}:${task}-oh"
    if docker manifest inspect "$image" > /dev/null 2>&1; then
        ((built++))
    else
        ((missing++))
        echo "  ❌ Missing: $image"
    fi
done
echo ""
echo "Total: $built/30 images on Docker Hub"
if [ $missing -eq 0 ]; then
    echo "🎉 All 30 images built!"
fi
