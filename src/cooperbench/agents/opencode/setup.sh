#!/bin/bash
# Installs opencode into the task container via the official installer.
# Binary lands at ~/.opencode/bin/opencode regardless of distro.
set -e

if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends curl ca-certificates git >/dev/null
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache curl bash git >/dev/null
elif command -v yum >/dev/null 2>&1; then
    yum install -y curl git >/dev/null
fi

curl -fsSL https://opencode.ai/install | bash

export PATH="$HOME/.opencode/bin:$PATH"
opencode --version
