#!/bin/bash
# Installs the @openai/codex CLI into the task container.
# Same shape as the Claude Code setup script — apt + nvm/node + npm.
set -e

# TTY-less containers: force noninteractive apt so debconf doesn't fall
# through Dialog->Readline->Teletype and trip dpkg ("Sub-process
# /usr/bin/dpkg returned an error code (1)").  Rare for a few concurrent
# installs (solo) but dominant under heavier concurrency (coop/team spin
# up 2x the containers), so harden it.
export DEBIAN_FRONTEND=noninteractive

# Retry transient apt/network hiccups (mirror throttling under many
# simultaneous installs from one host) instead of failing the whole run.
_retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        [ "$n" -ge 3 ] && return 1
        sleep $((n * 3))
    done
}

if command -v apt-get >/dev/null 2>&1; then
    _retry apt-get update -qq
    _retry apt-get install -y --no-install-recommends curl ca-certificates gnupg >/dev/null
elif command -v apk >/dev/null 2>&1; then
    _retry apk add --no-cache curl bash nodejs npm >/dev/null
elif command -v yum >/dev/null 2>&1; then
    _retry yum install -y curl >/dev/null
fi

if ! command -v npm >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
    _retry apt-get install -y --no-install-recommends nodejs >/dev/null
fi

VERSION="${CODEX_VERSION:-latest}"
npm install -g --silent "@openai/codex@${VERSION}"
codex --version

# Shared coop install (no-op when /tmp/cb-coop-msg.py is absent, i.e. solo).
if [ -f /tmp/cb-coop-install.sh ]; then
    bash /tmp/cb-coop-install.sh
fi
# Team task-list install (no-op outside team mode).
if [ -f /tmp/cb-team-install.sh ]; then
    bash /tmp/cb-team-install.sh
fi
