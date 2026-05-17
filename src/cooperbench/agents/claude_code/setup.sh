#!/bin/bash
# Installs Claude Code into the task container.
# The shared coop helper install runs inline below (we ship the snippet
# in /tmp/cb-coop-install.sh from the adapter so we don't need network
# access to fetch it).
set -e

if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends curl ca-certificates gnupg >/dev/null
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache curl bash nodejs npm >/dev/null
elif command -v yum >/dev/null 2>&1; then
    yum install -y curl >/dev/null
fi

if ! command -v npm >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
    apt-get install -y --no-install-recommends nodejs >/dev/null
fi

VERSION="${CLAUDE_CODE_VERSION:-latest}"
npm install -g --silent "@anthropic-ai/claude-code@${VERSION}"
claude --version

# Coop helper install (no-op when /tmp/cb-coop-msg.py is absent, i.e. solo).
if [ -f /tmp/cb-coop-install.sh ]; then
    bash /tmp/cb-coop-install.sh
fi
# Team task-list helper install (no-op outside team mode).
if [ -f /tmp/cb-team-install.sh ]; then
    bash /tmp/cb-team-install.sh
fi
