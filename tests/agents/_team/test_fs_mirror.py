"""Unit tests for the filesystem mirror of the task list.

Claude Code's team primitive stores tasks as on-disk JSON files so
agents can ``ls``/``cat``/``grep`` them with their existing tools — no
new tool surface to learn.  We mirror that for CooperBench by
snapshotting the Redis task list to ``/workspace/shared/tasks/`` on
every ``coop-task-list`` invocation.

Layout (all under ``CONTAINER_SCRATCHPAD_DIR/tasks/``):

    <task_id>.json       one file per task, fields mirror TaskListClient.get()
    _index.json          {"updated_at": <ts>, "ids": [...]}; cheap directory listing
                         agents can grep without opening every file
    _log.jsonl           audit log copy (one JSON event per line)

The mirror is *eventually consistent* — there's no inotify-style push
from Redis, just a snapshot every time something runs ``coop-task-list``.
That's fine for our cadence; the bench is built around agents that
poll between turns anyway.
"""

from __future__ import annotations

import json

import fakeredis
import pytest

from cooperbench.agents._team.fs_mirror import mirror_to_directory
from cooperbench.agents._team.task_list import TaskListClient


@pytest.fixture
def client():
    fake = fakeredis.FakeRedis()
    return TaskListClient(redis_client=fake, run_id="test")


@pytest.fixture
def mirror_dir(tmp_path):
    target = tmp_path / "shared" / "tasks"
    return target


class TestMirrorToDirectory:
    def test_creates_one_file_per_task(self, client, mirror_dir):
        a = client.create(title="a", created_by="lead")
        b = client.create(title="b", created_by="lead")
        mirror_to_directory(client, mirror_dir)
        files = sorted(p.name for p in mirror_dir.iterdir() if p.suffix == ".json" and not p.name.startswith("_"))
        assert files == sorted([f"{a}.json", f"{b}.json"])

    def test_file_contents_match_task(self, client, mirror_dir):
        task_id = client.create(title="hello", created_by="lead", owner="agent2")
        mirror_to_directory(client, mirror_dir)
        data = json.loads((mirror_dir / f"{task_id}.json").read_text())
        assert data["id"] == task_id
        assert data["title"] == "hello"
        assert data["owner"] == "agent2"
        assert data["status"] == "open"

    def test_writes_index_file(self, client, mirror_dir):
        ids = [client.create(title=f"t{i}", created_by="lead") for i in range(3)]
        mirror_to_directory(client, mirror_dir)
        index = json.loads((mirror_dir / "_index.json").read_text())
        assert sorted(index["ids"]) == sorted(ids)
        assert "updated_at" in index
        assert isinstance(index["updated_at"], float)

    def test_writes_log_jsonl(self, client, mirror_dir):
        task_id = client.create(title="t", created_by="lead")
        client.claim(task_id, by="agent2")
        mirror_to_directory(client, mirror_dir)
        log_lines = (mirror_dir / "_log.jsonl").read_text().splitlines()
        assert len(log_lines) == 2  # create + claim
        # Each line is valid JSON.
        for line in log_lines:
            json.loads(line)

    def test_removes_stale_files(self, client, mirror_dir):
        a = client.create(title="a", created_by="lead")
        mirror_to_directory(client, mirror_dir)
        # Now delete the underlying task and mirror again — the file
        # for `a` should disappear.
        client._r.delete(client._task_key(a))
        client._r.srem(client._all_key, a)
        mirror_to_directory(client, mirror_dir)
        assert not (mirror_dir / f"{a}.json").exists()

    def test_directory_created_if_missing(self, client, tmp_path):
        target = tmp_path / "does" / "not" / "exist" / "tasks"
        client.create(title="t", created_by="lead")
        mirror_to_directory(client, target)
        assert target.exists()
        assert (target / "_index.json").exists()

    def test_empty_list_still_writes_index(self, client, mirror_dir):
        mirror_to_directory(client, mirror_dir)
        index = json.loads((mirror_dir / "_index.json").read_text())
        assert index["ids"] == []

    def test_subsequent_mirrors_are_atomic_per_file(self, client, mirror_dir):
        """A reader watching ``_index.json`` must always see a consistent view.

        We don't get atomic dirsync, but we do guarantee each file is
        written as a single ``write_text`` (Python's tempfile+replace
        idiom under the hood) — so a reader either sees the old or new
        version of a given file, never a half-written one.
        """
        task_id = client.create(title="initial", created_by="lead")
        mirror_to_directory(client, mirror_dir)
        # Mutate and re-mirror.  We use ``claim`` to change owner +
        # status and ``update`` to add a note — both real writes.
        client.claim(task_id, by="agent1")
        client.update(task_id, by="agent1", status="in_progress", note="started")
        mirror_to_directory(client, mirror_dir)
        contents = (mirror_dir / f"{task_id}.json").read_text()
        assert contents.strip(), "file must never be empty after a successful mirror"
        data = json.loads(contents)
        assert data["owner"] == "agent1"
        assert data["last_note"] == "started"
