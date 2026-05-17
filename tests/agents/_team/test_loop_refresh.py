"""Unit tests for the in-loop task-list refresh helper.

Python-loop adapters (mini_swe_agent_v2, swe_agent, openhands_sdk)
poll their inbox between steps.  When team mode is active they should
*also* refresh the task list and inject a summary as a user message so
the LLM sees the current state of the shared list before its next
response — without needing to remember to call ``coop-task-list``.

``format_task_summary`` and ``poll_team_state`` are pure helpers that
the adapters' existing inbox-check hooks call when ``CB_TEAM_REDIS_URL``
+ ``CB_TEAM_AGENT_ID`` are present in the env.  They return plain
strings the adapter prepends to the conversation; the adapter does
the actual ``add_messages`` call so we don't have to know each
framework's message shape.
"""

from __future__ import annotations

import fakeredis
import pytest

from cooperbench.agents._team.loop_refresh import (
    TeamPoller,
    format_task_summary,
    poll_team_state,
)
from cooperbench.agents._team.task_list import TaskListClient


@pytest.fixture
def shared():
    fake = fakeredis.FakeRedis()
    client = TaskListClient(redis_client=fake, run_id="t")
    return fake, client


class TestFormatTaskSummary:
    def test_summary_groups_by_status(self):
        tasks = [
            {"id": "t1", "title": "alpha", "status": "open", "owner": ""},
            {"id": "t2", "title": "beta", "status": "in_progress", "owner": "agent1"},
            {"id": "t3", "title": "gamma", "status": "done", "owner": "agent2"},
        ]
        text = format_task_summary(tasks, viewer="agent1")
        # Status counts surface up top so the LLM sees the headline first.
        assert "open: 1" in text
        assert "in_progress: 1" in text
        assert "done: 1" in text
        # Each task is mentioned by title.
        assert "alpha" in text
        assert "beta" in text
        assert "gamma" in text

    def test_summary_calls_out_viewer_tasks(self):
        tasks = [
            {"id": "t1", "title": "mine", "status": "in_progress", "owner": "agent1"},
            {"id": "t2", "title": "theirs", "status": "in_progress", "owner": "agent2"},
        ]
        text = format_task_summary(tasks, viewer="agent1")
        # The viewer's own task should be clearly marked.
        assert "your task" in text.lower() or "you own" in text.lower()

    def test_empty_summary_is_concise(self):
        text = format_task_summary([], viewer="agent1")
        assert "no tasks" in text.lower()
        assert len(text) < 200  # don't waste tokens on an empty state


class TestPollTeamState:
    def test_returns_none_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("CB_TEAM_REDIS_URL", raising=False)
        assert poll_team_state() is None

    def test_returns_none_when_redis_unreachable(self, monkeypatch):
        monkeypatch.setenv("CB_TEAM_REDIS_URL", "redis://localhost:1")  # unreachable
        monkeypatch.setenv("CB_TEAM_RUN_ID", "t")
        monkeypatch.setenv("CB_TEAM_AGENT_ID", "agent1")
        # Should swallow connection errors and return None — never
        # crash the agent loop because Redis is down.
        assert poll_team_state() is None

    def test_returns_summary_string_when_reachable(self, monkeypatch, shared):
        fake, client = shared
        client.create(title="t1", created_by="lead", owner="agent1")
        monkeypatch.setenv("CB_TEAM_REDIS_URL", "redis://stub")
        monkeypatch.setenv("CB_TEAM_RUN_ID", "t")
        monkeypatch.setenv("CB_TEAM_AGENT_ID", "agent1")

        # Patch the redis client factory so we use fakeredis.
        from unittest.mock import patch as mp

        with mp("cooperbench.agents._team.loop_refresh._client_from_env", return_value=fake):
            summary = poll_team_state()

        assert summary is not None
        assert "t1" in summary


class TestTeamPoller:
    """Host-side per-agent poller used by Python-loop adapters."""

    def test_poll_returns_summary_when_reachable(self, shared):
        fake, client = shared
        task_id = client.create(title="hello", created_by="lead", owner="agent1")
        poller = TeamPoller(redis_url="redis://stub", run_id="t", agent_id="agent1")
        poller._client = fake  # inject fakeredis directly
        summary = poller.poll()
        assert summary is not None
        assert task_id in summary
        assert "hello" in summary

    def test_poll_returns_none_when_unreachable(self):
        poller = TeamPoller(redis_url="redis://localhost:1", run_id="t", agent_id="agent1")
        # Don't pre-set _client; ensure_client should fail at connect-time
        # and poll should swallow the error.
        assert poller.poll() is None
