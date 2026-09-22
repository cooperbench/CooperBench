"""Shared-git-off runs must not tell an agent it can read origin/<peer>."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from jinja2 import StrictUndefined, Template

from cooperbench.agents.mini_swe_agent_v2.agents.default import DefaultAgent

_COOP_YAML = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "cooperbench"
    / "agents"
    / "mini_swe_agent_v2"
    / "config"
    / "coop.yaml"
)


def _render(git_enabled: bool, messaging_enabled: bool = True) -> str:
    raw = yaml.safe_load(_COOP_YAML.read_text())
    templates = raw["agent"]["system_template"] + raw["agent"]["instance_template"]
    return Template(templates, undefined=StrictUndefined).render(
        task="Implement the feature",
        agent_id="agent1",
        agents=["agent1", "agent2"],
        git_enabled=git_enabled,
        messaging_enabled=messaging_enabled,
        system="Linux",
        release="test",
        version="0",
        machine="x86_64",
    )


class _Comm:
    def __init__(self, published: bool):
        self._published = published

    def has_published(self, peer: str) -> bool:
        return self._published


def _agent(git_enabled: bool, published: bool) -> DefaultAgent:
    agent = DefaultAgent(
        model=MagicMock(),
        env=MagicMock(),
        comm=_Comm(published),
        system_template="sys",
        instance_template="task",
    )
    agent.extra_template_vars["git_enabled"] = git_enabled
    return agent


def test_prompt_hides_peer_branch_when_shared_git_is_off():
    text = _render(False)
    assert "git fetch origin" not in text
    assert "origin/agent2" not in text
    assert "published branch" not in text
    assert "cannot fetch" in text
    assert "gh pr create" in text


def test_prompt_shows_peer_branch_when_shared_git_is_on():
    text = _render(True)
    assert "git fetch origin && git diff origin/main origin/agent2" in text
    assert "published branch" in text
    assert "colleague can read it" in text


@pytest.mark.parametrize("git_enabled", [False, True])
@pytest.mark.parametrize("messaging_enabled", [False, True])
def test_prompt_only_recommends_available_channels(git_enabled, messaging_enabled):
    text = _render(git_enabled, messaging_enabled)
    assert ("send_message" in text) is messaging_enabled
    assert ("## Messaging" in text) is messaging_enabled
    assert ("git fetch origin" in text) is git_enabled
    assert ("colleague can read it" in text) is git_enabled
    assert "gh pr create" in text
    if not messaging_enabled:
        assert "message them" not in text
        assert "settle it over messaging" not in text
        assert "You communicate naturally" not in text
    if not git_enabled and not messaging_enabled:
        assert "Work independently" in text


@pytest.mark.parametrize("git_enabled", [False, True])
def test_messaging_disabled_does_not_inject_peer_messages(git_enabled):
    model = MagicMock()
    agent = DefaultAgent(model=model, env=MagicMock(), comm=None, system_template="sys", instance_template="task")
    agent.extra_template_vars.update(git_enabled=git_enabled, messaging_enabled=False)
    agent.query = MagicMock(return_value={"role": "assistant"})
    agent.execute_actions = MagicMock(return_value=[])
    agent.step()
    assert agent.messages == []
    model.format_message.assert_not_called()


def test_pointer_does_not_fetch_when_shared_git_is_off_even_if_local_pr_exists():
    text = _agent(False, True)._peer_work_pointer("agent2")
    assert "git fetch" not in text
    assert "not visible" in text
    assert "graded submission" in text


def test_pointer_fetches_only_when_shared_git_published():
    visible = _agent(True, True)._peer_work_pointer("agent2")
    assert "git fetch origin && git diff HEAD...origin/agent2" in visible
    hidden = _agent(True, False)._peer_work_pointer("agent2")
    assert "could NOT be published" in hidden
    assert "git fetch" not in hidden
