#!/bin/bash
# Shared coop setup snippet, sourced from each adapter's setup.sh
# (Claude Code, Codex, future agents).  Installs the redis Python lib
# and creates the coop-* shell wrappers at /usr/local/bin/.
# Idempotent.  Assumes /tmp/cb-coop-msg.py was dropped in by the adapter.
set -e

if command -v pip >/dev/null 2>&1; then
    pip install --quiet --disable-pip-version-check redis >/dev/null || true
elif command -v pip3 >/dev/null 2>&1; then
    pip3 install --quiet --disable-pip-version-check redis >/dev/null || true
fi

if [ -f /tmp/cb-coop-msg.py ]; then
    chmod +x /tmp/cb-coop-msg.py
    for sub in send recv broadcast peek agents; do
        cat >"/usr/local/bin/coop-$sub" <<EOF
#!/bin/bash
exec python3 /tmp/cb-coop-msg.py $sub "\$@"
EOF
        chmod +x "/usr/local/bin/coop-$sub"
    done
fi
