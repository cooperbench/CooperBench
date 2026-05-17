"""Unit tests for the team-mode prompt assembly.

Team mode adds two role-specific blocks on top of the shared coop prompt:

  - ``lead`` block — names the agent ``team-lead``, explains the role
    (organize via the task list, verify integration; you're also a
    developer), and documents the coop-task-* CLI.
  - ``member`` block — names the lead by id, documents the same CLI from
    a worker's perspective, encourages claiming open tasks.

Both branches must be agent-agnostic (no mention of Claude / Codex /
specific tools beyond the coop-* CLI).
"""

from cooperbench.agents._team.prompt import build_team_instruction


def _both(text: str) -> None:
    """Sanity checks every team-mode prompt should satisfy."""
    assert "coop-task-create" in text
    assert "coop-task-claim" in text
    assert "coop-task-list" in text
    assert "coop-task-update" in text
    # Shared scratchpad path is exposed to both roles.
    assert "/workspace/shared" in text


class TestLeadPrompt:
    def test_lead_named_team_lead(self):
        text = build_team_instruction(
            task="Implement feature X",
            agents=["agent1", "agent2", "agent3"],
            agent_id="agent1",
            team_role="lead",
            git_enabled=False,
        )
        assert "Implement feature X" in text
        assert "team-lead" in text
        _both(text)

    def test_lead_lists_member_ids(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2", "agent3"],
            agent_id="agent1",
            team_role="lead",
            git_enabled=False,
        )
        assert "agent2" in text
        assert "agent3" in text

    def test_lead_block_explains_assign_workflow(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2"],
            agent_id="agent1",
            team_role="lead",
            git_enabled=False,
        )
        # The lead block must teach the assign-then-verify workflow.
        # We check for the verbs rather than exact phrasing so the
        # prompt can evolve without breaking the test.
        lower = text.lower()
        assert "claim" in lower or "assign" in lower
        assert "review" in lower or "verify" in lower or "integrate" in lower


class TestMemberPrompt:
    def test_member_named_with_lead_reference(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2", "agent3"],
            agent_id="agent2",
            team_role="member",
            git_enabled=False,
        )
        # Member should know who the lead is by id.
        assert "agent1" in text
        # And know its own id.
        assert "agent2" in text
        _both(text)

    def test_member_block_directs_to_open_tasks(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2"],
            agent_id="agent2",
            team_role="member",
            git_enabled=False,
        )
        lower = text.lower()
        # Members should know to look for open tasks first.
        assert "coop-task-list" in text
        assert "open" in lower

    def test_member_does_not_get_lead_block(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2"],
            agent_id="agent2",
            team_role="member",
            git_enabled=False,
        )
        # The phrase "you are the team-lead" must not appear in a
        # member prompt — that would confuse the agent about its role.
        assert "you are the team-lead" not in text.lower()


class TestSoloFallback:
    def test_no_team_role_no_team_block(self):
        text = build_team_instruction(
            task="t",
            agents=None,
            agent_id=None,
            team_role=None,
            git_enabled=False,
        )
        # Solo-shape prompt: must NOT mention team primitives.
        assert "coop-task" not in text
        assert "team-lead" not in text.lower()
        assert "/workspace/shared" not in text

    def test_single_agent_treated_as_solo(self):
        """A team of one is degenerate — fall back to solo prompt."""
        text = build_team_instruction(
            task="t",
            agents=["agent1"],
            agent_id="agent1",
            team_role="lead",
            git_enabled=False,
        )
        assert "coop-task" not in text


class TestFinalSubmissionExplicit:
    """Regression test for the e2e finding where members wrote diffs
    only to the scratchpad and never to ``patch.txt``."""

    def test_lead_prompt_demands_patch_txt(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2"],
            agent_id="agent1",
            team_role="lead",
            git_enabled=False,
        )
        assert "patch.txt" in text
        assert "REQUIRED" in text or "MUST" in text
        assert "/workspace/repo/patch.txt" in text

    def test_member_prompt_demands_patch_txt(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2"],
            agent_id="agent2",
            team_role="member",
            git_enabled=False,
        )
        assert "/workspace/repo/patch.txt" in text
        assert "git diff > patch.txt" in text

    def test_both_prompts_call_out_shared_not_evaluated(self):
        for role in ("lead", "member"):
            text = build_team_instruction(
                task="t",
                agents=["agent1", "agent2"],
                agent_id="agent1" if role == "lead" else "agent2",
                team_role=role,
                git_enabled=False,
            )
            # The whole point of the fix: tell the agent shared/ is
            # coordination, not submission.
            assert "/workspace/shared" in text
            assert "NOT" in text or "not evaluated" in text.lower()


class TestGitInteraction:
    def test_git_block_appended_when_enabled(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2"],
            agent_id="agent1",
            team_role="lead",
            git_enabled=True,
        )
        # Existing coop+git section composes with team block.
        assert "## Git collaboration" in text
        assert "team/" in text  # remote-branch shape

    def test_no_git_block_when_disabled(self):
        text = build_team_instruction(
            task="t",
            agents=["agent1", "agent2"],
            agent_id="agent1",
            team_role="lead",
            git_enabled=False,
        )
        assert "## Git collaboration" not in text
