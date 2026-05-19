"""Prompt assembly for team mode.

Builds on the shared coop prompt (``cooperbench.agents._coop.prompt``)
by appending a role-specific team block:

  - ``lead`` block: names the agent ``team-lead``, lists members,
    documents the ``coop-task-*`` CLI from an organizer's perspective,
    points to the shared scratchpad.
  - ``member`` block: names the lead, encourages claiming open tasks,
    documents the same CLI from a worker's perspective.

A "team of one" is degenerate and falls back to a plain solo prompt
(no team block emitted).
"""

from __future__ import annotations

from cooperbench.agents._coop.prompt import build_instruction as _build_coop_instruction

_TEAM_LIST_USAGE = """Available shell commands (Redis-backed, atomic):

```bash
coop-task-create "<title>"                 # creates an open, unassigned task; prints task_id
coop-task-create --assign <agent> "<title>" # creates and pre-assigns; agent must still claim
coop-task-claim <task_id>                   # exit 0 if you got it, 2 if someone else owns it
coop-task-update <task_id> <status> [-n "<note>"]
                                            # statuses: open | in_progress | blocked | done
coop-task-list                              # JSON list of every task in the team
coop-task-list --mine                       # tasks you own
coop-task-list --open                       # open, unassigned tasks
```

Messaging (`coop-send` / `coop-recv` / `coop-broadcast` / `coop-peek`)
also works and is the right tool for short questions that don't
warrant a new task."""


def _lead_block(agent_id: str, members: list[str]) -> str:
    member_list = ", ".join(f"`{m}`" for m in members)
    return f"""## You are the team-lead

You are **{agent_id}** — the team-lead.  Members reporting to you:
{member_list}.

### The integration step is the WHOLE point of your job — DO NOT SKIP IT

The benchmark scores the **merged** team output, not your individual
contribution.  If you submit `patch.txt` containing only your own
feature, the team **fails** — even if your feature works perfectly.
The single most common failure mode here is "lead ran `git diff > patch.txt`
before pulling in the member's work."  Avoid this.

Workflow:

1. **Plan to avoid conflicts.** Read the feature spec.  If features
   touch the same file/function, divide the work so changes occupy
   disjoint regions (e.g., one feature owns the function signature,
   the other owns helper additions).  Drop a one-paragraph plan in
   `/workspace/shared/PLAN.md` so members see your decomposition.
2. Break the spec into 2-4 concrete tasks and assign one per member
   with `coop-task-create --assign <agent> "..."`.
3. You **may also implement a task yourself** in parallel.
4. Poll progress: `coop-task-list` periodically.  If a member's task
   is `blocked`, read the `last_note` and either reassign or
   unblock it with `coop-send`.

### Before submitting — MANDATORY integration checklist

You may submit ONLY after **all five** of these are true:

- [ ] `coop-task-list` shows every team task as `done`.
- [ ] Every member has dropped their patch at
      `/workspace/shared/<agent_id>.patch`.  (Run
      `ls /workspace/shared/*.patch` to verify.)
- [ ] You have read each member's patch (`cat /workspace/shared/<agent>.patch`)
      and applied it to your working tree (`git apply
      /workspace/shared/<agent>.patch`, fixing any conflicts
      manually with Edit until the tree builds).
- [ ] The merged tree builds / compiles.  Run the project's build
      or import test before continuing.
- [ ] `git diff` now shows BOTH features — your own and every
      member's — present in the working tree.

Only when ALL five boxes are checked, write `/workspace/repo/patch.txt`
via `git diff` and exit — this is REQUIRED:

```bash
cd /workspace/repo && git diff > patch.txt && wc -l patch.txt
```

If `wc -l patch.txt` looks too small to contain everyone's work, you
skipped the integration step — go back to step 3 of the checklist.

{_TEAM_LIST_USAGE}

A shared scratchpad volume is mounted at `/workspace/shared/` for
coordination — files there are NOT evaluated.  Drop design notes,
interface contracts, or member-produced patches there so the whole
team can see them with `ls /workspace/shared/`."""


def _member_block(agent_id: str, lead: str) -> str:
    return f"""## You are a team member

You are **{agent_id}**.  The team-lead is **{lead}**, who will
organize work into tasks for you to claim.

### Stay in your lane

The lead may have left a plan at `/workspace/shared/PLAN.md`.  Read it
first — it tells you which file regions your feature owns, vs. those
reserved for your peers.  Edit ONLY the regions your task owns.

Workflow:

1. Run `coop-task-list --open` to see what needs doing.  If your
   `agent_id` appears as a pre-assigned `owner` on a task, that's the
   one the lead expects you to take.  You can also `ls
   /workspace/shared/tasks/` to browse without the CLI.
2. `coop-task-claim <task_id>`.  If you lose the race (exit code 2),
   pick another.
3. Read `/workspace/shared/PLAN.md` if it exists, then implement.
   Stay within your region.  When you hit a blocker, run
   `coop-task-update <task_id> blocked -n "<what you need>"` and
   `coop-send {lead} "blocked on task <id>: ..."`.
4. When done, ALWAYS export your diff for the lead to consume:

   ```bash
   cd /workspace/repo && git diff > /workspace/shared/{agent_id}.patch
   ```

   Then `coop-task-update <task_id> done -n "patch at
   /workspace/shared/{agent_id}.patch"`.  The lead will integrate.

### Final submission — REQUIRED

The bench scores per-agent.  Before exiting you MUST write your own
`/workspace/repo/patch.txt`:

```bash
cd /workspace/repo && git diff > patch.txt && wc -l patch.txt
```

Your `/workspace/repo/patch.txt` should reflect your final working
tree.  If the lead asked you to merge in their plan, do so first.

{_TEAM_LIST_USAGE}

A shared scratchpad volume is mounted at `/workspace/shared/` for
coordination — files there are NOT evaluated.  Use it for anything
your peers might need to see — partial diffs, interface sketches,
error logs from your reproduction script."""


def team_task_section(
    *,
    agents: list[str] | None,
    agent_id: str | None,
    team_role: str | None,
) -> str:
    """Return JUST the team-task-list section for an adapter to append.

    Used by Python-loop adapters that already have their own coop
    prompts covering messaging / git / submission, but need to teach
    the LLM about the new ``coop-task-*`` CLI + role split without
    re-explaining everything else.  CLI adapters use the bigger
    ``build_team_instruction`` instead.

    Empty string when team mode isn't active (no role, <2 agents).
    """
    if not team_role or not agents or not agent_id or len(agents) < 2:
        return ""
    members = [a for a in agents if a != agent_id]
    if team_role == "lead":
        return _lead_block(agent_id, members)
    lead = members[0] if members else "team-lead"
    return _member_block(agent_id, lead)


def build_team_instruction(
    task: str,
    *,
    agents: list[str] | None,
    agent_id: str | None,
    team_role: str | None,
    git_enabled: bool = False,
) -> str:
    """Compose the full instruction for a team-mode agent run.

    Args:
        task: Raw feature spec.
        agents: All agent ids in the team.  A team of one falls back to
            solo (no team block).
        agent_id: This agent's id.  Required when ``team_role`` is set.
        team_role: ``"lead"`` or ``"member"``.  ``None`` means we're not
            in team mode — falls back to the coop / solo prompt.
        git_enabled: When True, the shared coop+git block is appended
            (same as coop mode).

    Returns the assembled prompt.  Team mode injects its own block
    INSTEAD of the regular coop messaging block — the coop-task CLI
    is the primary coordination primitive in team mode; `coop-send`
    is documented inside the team block as the secondary channel.
    """
    # Base prompt = task + submission protocol (no coop block — we
    # provide our own).
    base = _build_coop_instruction(task)

    if not team_role or not agents or not agent_id or len(agents) < 2:
        return base

    members = [a for a in agents if a != agent_id]
    if team_role == "lead":
        team_section = _lead_block(agent_id, members)
    else:  # member
        # Pick the first non-self as the implied lead.  The runner always
        # sets agents=[lead, member1, member2, ...] so this is correct.
        lead = members[0] if members else "team-lead"
        # If the caller passed a specific lead via members[0], honour it.
        team_section = _member_block(agent_id, lead)

    sections = [base, team_section]
    if git_enabled:
        from cooperbench.agents._coop.prompt import _git_block

        sections.append(_git_block(agent_id, members))
    return "\n\n---\n\n".join(sections)
