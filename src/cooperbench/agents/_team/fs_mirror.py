"""Filesystem mirror of the Redis task list.

Mirrors the CC team primitive of "tasks as on-disk files agents can
``ls`` and ``cat``" without giving up the atomicity Redis gives us.
The mirror is the read-only projection; Redis remains the source of
truth and the only thing that accepts writes.

Layout (relative to a caller-supplied target directory):

    <task_id>.json       one file per task, fields mirror
                         ``TaskListClient.get(task_id)``
    _index.json          {"updated_at": <float>, "ids": [...]} —
                         cheap directory listing
    _log.jsonl           audit log copy (one JSON event per line)

Snapshots are eventually consistent and triggered explicitly by the
in-container ``coop-task-list`` CLI (and by the team runner at the
start of each run).  Each file is written via ``tempfile + replace``
so a concurrent reader either sees the old or new content, never a
half-written one.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from cooperbench.agents._team.task_list import TaskListClient


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` such that readers see either the old
    or new file but never a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile gives us a unique name in the same directory
    # so the final ``os.replace`` is rename(2) on a single filesystem
    # and therefore atomic.
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        delete=False,
        encoding="utf-8",
    )
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, path)


def mirror_to_directory(client: TaskListClient, target_dir: str | Path) -> None:
    """Snapshot the current task list to ``target_dir``.

    Idempotent and safe to call from any process holding a
    ``TaskListClient``.  Removes files for tasks that have been
    deleted from Redis since the last snapshot.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    tasks = client.list()
    live_ids = {t["id"] for t in tasks}

    # 1. Per-task files.
    for task in tasks:
        path = target / f"{task['id']}.json"
        _atomic_write_text(path, json.dumps(task, indent=2, default=str))

    # 2. Remove stale files (tasks that no longer exist in Redis).
    for existing in target.glob("*.json"):
        if existing.name.startswith("_"):
            continue
        if existing.stem not in live_ids:
            try:
                existing.unlink()
            except OSError:
                pass

    # 3. Index file — cheap path for ``ls``-equivalent reads.
    index = {"updated_at": time.time(), "ids": sorted(live_ids)}
    _atomic_write_text(target / "_index.json", json.dumps(index, indent=2))

    # 4. Audit log copy.
    events = client.log_events()
    log_text = "\n".join(json.dumps(e) for e in events)
    if log_text:
        log_text += "\n"
    _atomic_write_text(target / "_log.jsonl", log_text)
