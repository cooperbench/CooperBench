#!/usr/bin/env bash
# Rebuild and push task images after changing a task's Dockerfile or runner.sh.
#
# runner.sh is COPY'd into the image at build time, so edits to it in this repo are invisible to
# every run until the image is rebuilt and pushed. A Dockerfile change (e.g. a base-image version
# bump) is the same. Symptom when you forget: the fix appears to do nothing.
#
# Two things are easy to get wrong here:
#
#   ARCHITECTURE  the published images are multi-arch (linux/amd64 + linux/arm64). Building on
#                 Apple Silicon without --platform produces arm64 only, and pushing that breaks
#                 every amd64 run. Always build both.
#   MODAL CACHE   Modal resolves `Image.from_registry(tag)` once and caches it, so a freshly
#                 pushed image under an unchanged tag is NOT picked up. Run anything that uses the
#                 image once with MODAL_FORCE_BUILD=1 afterwards to refresh it.
#
#   ./scripts/rebuild_task_images.sh                          # every task changed vs origin/main
#   ./scripts/rebuild_task_images.sh go_chi_task/task27 ...    # specific tasks
set -euo pipefail

cd "$(dirname "$0")/.."
PLATFORMS="linux/amd64,linux/arm64"

if [ $# -gt 0 ]; then
    TASKS=("$@")
else
    mapfile -t TASKS < <(
        git diff --name-only origin/main -- 'dataset/*/task*/runner.sh' 'dataset/*/task*/Dockerfile' \
        | sed -E 's|dataset/||; s|/(runner\.sh\|Dockerfile)$||' | sort -u
    )
fi

[ ${#TASKS[@]} -eq 0 ] && { echo "no task images need rebuilding"; exit 0; }
echo "rebuilding ${#TASKS[@]} image(s) for $PLATFORMS"

failed=()
for t in "${TASKS[@]}"; do
    repo="${t%%/*}"; task="${t##*/}"
    image=$(python3 -c "
import sys; sys.path.insert(0, 'src')
from cooperbench.utils import get_image_name
print(get_image_name('$repo', int('${task#task}')))")
    echo ""
    echo "=== $t -> $image"

    # If only runner.sh changed, overlay one COPY layer on the published image instead of
    # rebuilding. A full rebuild re-resolves every dependency (the datasets images pull
    # tensorflow + torch + jax, typst compiles Rust) for two platforms under emulation, which
    # takes tens of minutes for what is a one-file change — and re-resolving deps is exactly how
    # these images drifted into being broken in the first place.
    if git diff --quiet origin/main -- "dataset/$t/Dockerfile"; then
        ctx=$(mktemp -d)
        cp "dataset/$t/runner.sh" "$ctx/runner.sh"
        printf 'FROM %s\nCOPY runner.sh /usr/local/bin/runner.sh\nRUN chmod +x /usr/local/bin/runner.sh\n' \
            "$image" > "$ctx/Dockerfile"
        echo "    runner.sh only -> overlaying on the published image"
        build_ctx="$ctx"
    else
        echo "    Dockerfile changed -> full rebuild"
        build_ctx="dataset/$t"
    fi

    before=$(docker buildx imagetools inspect "$image" 2>/dev/null | grep -m1 '^Digest:' | awk '{print $2}' || true)
    log=$(mktemp)
    if docker buildx build --platform "$PLATFORMS" -t "$image" --push "$build_ctx" >"$log" 2>&1; then
        after=$(docker buildx imagetools inspect "$image" 2>/dev/null | grep -m1 '^Digest:' | awk '{print $2}' || true)
        # A zero exit is not proof the tag moved: a push can no-op silently. Compare digests.
        if [ -n "$before" ] && [ "$before" = "$after" ]; then
            echo "    PUSH DID NOT TAKE (digest unchanged: ${before:0:19})"
            tail -5 "$log" | sed 's/^/      /'
            failed+=("$t")
        else
            echo "    pushed  ${after:0:19}"
        fi
    else
        echo "    BUILD FAILED"
        tail -8 "$log" | sed 's/^/      /'
        failed+=("$t")
    fi
    rm -f "$log"
    [ -n "${ctx:-}" ] && rm -rf "$ctx" && unset ctx
done

echo ""
if [ ${#failed[@]} -gt 0 ]; then
    printf 'failed: %s\n' "${failed[@]}"
    exit 1
fi
echo "all pushed. Modal still holds the OLD images under these tags — refresh with:"
echo "    MODAL_FORCE_BUILD=1 python scripts/check_gradeable.py <repo>/<task>"
