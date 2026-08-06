#!/bin/sh
# Minimal `gh` for CooperBench: pull requests against the shared `origin` remote.
#
# Agents already know `gh pr create` / `gh pr diff` — those commands are everywhere in
# training data. A bespoke command with the same behaviour would not be, so this shim keeps
# the real spelling and implements it with plain git against the team remote.
#
# A PR tracks a BRANCH, the way GitHub does: you open it early so your colleague has context
# for what you are doing, then keep committing, and the new commits are part of it
# automatically. Opening one is recorded as an annotated tag holding the title and body; the
# content is always the current tip of your branch, never a commit frozen at open time.
#
#   refs/tags/pr/<agent_id>   ->  marker, message = "<title>\n\n<body>"
#   refs/heads/<agent_id>     ->  the content, updated by every push
#
# There is no GitHub and no network here. Anything not implemented says so plainly rather
# than failing in a way that reads like a transient error.
set -eu

AGENT="${COOPERBENCH_AGENT_ID:-}"
REMOTE=origin
BASE="${REMOTE}/main"

die() { echo "$*" >&2; exit 1; }

[ -n "$AGENT" ] || die "gh: COOPERBENCH_AGENT_ID is not set; this shim needs to know who you are."

unsupported() {
    cat >&2 <<EOF
gh: '$*' is not available in this environment.

There is no GitHub here — PRs live on the shared 'origin' git remote. Available:
  gh pr create --title T --body B   open your PR (proposes your current commit)
  gh pr list                        PRs opened so far
  gh pr view <agent>                title, body and summary of their PR
  gh pr diff <agent>                the code they are proposing
  gh pr checkout <agent>            check out their PR locally

Review happens over send_message, not PR comments.
EOF
    exit 2
}

pr_ref() { echo "refs/tags/pr/$1"; }

fetch_prs() { git fetch -q --tags "$REMOTE" 2>/dev/null || true; git fetch -q "$REMOTE" 2>/dev/null || true; }

cmd_create() {
    title=""; body=""
    while [ $# -gt 0 ]; do
        case "$1" in
            -t|--title) title="${2:-}"; shift 2 ;;
            -b|--body)  body="${2:-}";  shift 2 ;;
            -T|--body-file) body="$(cat "${2:-}")"; shift 2 ;;
            --draft|-d|--fill) shift ;;
            *) shift ;;
        esac
    done
    [ -n "$title" ] || die "gh pr create: --title is required."

    # Deliberately no "you have no commits yet" check: opening a PR early, before there is
    # much to show, is the point -- it tells your colleague what you are taking.
    #
    # Refuse to publish from someone else's branch. `gh pr checkout <peer>` leaves you on
    # pr-<peer>; opening a PR from there would submit your colleague's code as your own, and
    # nothing downstream could tell the difference.
    current="$(git rev-parse --abbrev-ref HEAD)"
    if [ "$current" != "$AGENT" ]; then
        die "gh pr create: you are on branch '$current', not your own branch '$AGENT'.
Switch back with: git checkout $AGENT"
    fi
    git push -q -f "$REMOTE" "HEAD:$AGENT"
    git tag -f -a "pr/$AGENT" -m "$title

$body" HEAD >/dev/null
    git push -f -q "$REMOTE" "refs/tags/pr/$AGENT"
    echo "opened PR for $AGENT: $title"
    echo "further commits are included automatically — push them with: git push team HEAD:$AGENT"
    git --no-pager diff --stat "$BASE" HEAD
}

cmd_list() {
    fetch_prs
    found=0
    for ref in $(git for-each-ref --format='%(refname)' 'refs/tags/pr/*'); do
        who="${ref##*/}"
        printf '%s\t%s\n' "$who" "$(git tag -l --format='%(contents:subject)' "pr/$who")"
        found=1
    done
    [ "$found" = 1 ] || echo "no PRs opened yet"
}

cmd_view() {
    who="${1:?gh pr view: which agent?}"
    fetch_prs
    git rev-parse -q --verify "$(pr_ref "$who")" >/dev/null \
        || die "gh pr view: $who has not opened a PR yet."
    git tag -l --format='%(contents)' "pr/$who"
    git --no-pager diff --stat "$BASE" "$REMOTE/$who"
}

cmd_diff() {
    who="${1:?gh pr diff: which agent?}"
    fetch_prs
    git rev-parse -q --verify "$(pr_ref "$who")" >/dev/null \
        || die "gh pr diff: $who has not opened a PR yet."
    git --no-pager diff "$BASE" "$REMOTE/$who"
}

cmd_checkout() {
    who="${1:?gh pr checkout: which agent?}"
    fetch_prs
    git checkout -q -B "pr-$who" "$REMOTE/$who"
    echo "checked out $who's PR as branch pr-$who"
}

[ $# -ge 1 ] || unsupported ""
[ "$1" = "pr" ] || unsupported "$@"
shift
[ $# -ge 1 ] || unsupported "pr"
sub="$1"; shift
case "$sub" in
    create)   cmd_create "$@" ;;
    list|ls)  cmd_list ;;
    view)     cmd_view "$@" ;;
    diff)     cmd_diff "$@" ;;
    checkout|co) cmd_checkout "$@" ;;
    *)        unsupported "pr $sub" ;;
esac
