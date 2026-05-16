#!/bin/bash
# Team-mode install snippet, sourced from each adapter's setup.sh after
# the coop snippet.  Creates the coop-task-* shell wrappers at
# /usr/local/bin if the helper is present.  Idempotent and a no-op for
# solo / coop runs (the helper file is only dropped in team mode).
set -e

if [ ! -f /tmp/cb-coop-task.py ]; then
    exit 0
fi

chmod +x /tmp/cb-coop-task.py
for sub in create claim update list; do
    cat >"/usr/local/bin/coop-task-$sub" <<EOF
#!/bin/bash
exec python3 /tmp/cb-coop-task.py $sub "\$@"
EOF
    chmod +x "/usr/local/bin/coop-task-$sub"
done
