#!/bin/bash
# Installs Claude Code into the task container.  Mirrors terminal-bench's
# claude-code-setup.sh.j2 but as a static script (no template rendering).
# Also installs the in-container coop messaging helper used in coop runs.
set -e

# Pick a package manager that exists.  Most CooperBench task images are
# Debian-based python:3.11-slim, but we keep the alpine/yum fallbacks for
# parity with Harbor's adapter so a wider range of base images work.
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends curl ca-certificates gnupg >/dev/null
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache curl bash nodejs npm >/dev/null
elif command -v yum >/dev/null 2>&1; then
    yum install -y curl >/dev/null
fi

# Install Node.js (>=20 required by claude-code).  Use NodeSource so we
# get a recent version on Debian.  Skip when npm is already present (alpine).
if ! command -v npm >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
    apt-get install -y --no-install-recommends nodejs >/dev/null
fi

VERSION="${CLAUDE_CODE_VERSION:-latest}"
npm install -g --silent "@anthropic-ai/claude-code@${VERSION}"

claude --version

# Coop messaging helper: only needed when COOP_REDIS_URL is set, but
# install unconditionally so the container layout is identical.
if command -v pip >/dev/null 2>&1; then
    pip install --quiet --disable-pip-version-check redis >/dev/null || true
elif command -v pip3 >/dev/null 2>&1; then
    pip3 install --quiet --disable-pip-version-check redis >/dev/null || true
fi

# coop_msg.py is dropped at /tmp/claude-coop-msg.py by the adapter.
# Symlink the subcommands so Claude can call them as standalone tools.
if [ -f /tmp/claude-coop-msg.py ]; then
    chmod +x /tmp/claude-coop-msg.py
    cat >/usr/local/bin/coop-send <<'EOF'
#!/bin/bash
exec python3 /tmp/claude-coop-msg.py send "$@"
EOF
    cat >/usr/local/bin/coop-recv <<'EOF'
#!/bin/bash
exec python3 /tmp/claude-coop-msg.py recv "$@"
EOF
    cat >/usr/local/bin/coop-broadcast <<'EOF'
#!/bin/bash
exec python3 /tmp/claude-coop-msg.py broadcast "$@"
EOF
    cat >/usr/local/bin/coop-peek <<'EOF'
#!/bin/bash
exec python3 /tmp/claude-coop-msg.py peek "$@"
EOF
    cat >/usr/local/bin/coop-agents <<'EOF'
#!/bin/bash
exec python3 /tmp/claude-coop-msg.py agents "$@"
EOF
    chmod +x /usr/local/bin/coop-send /usr/local/bin/coop-recv \
        /usr/local/bin/coop-broadcast /usr/local/bin/coop-peek /usr/local/bin/coop-agents
fi
