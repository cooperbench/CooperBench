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

Your job is to **organize the work via the shared task list before
writing code**.  Concretely, at the start of the session:

1. Break the feature spec into 2-4 concrete tasks (titles like
   "add filters parameter to prompt()", "wire create_jinja_env helper").
2. Create them with `coop-task-create` and assign one per member with
   `coop-task-create --assign <agent> "..."`.
3. While members work, run `coop-task-list` periodically to watch
   progress.  If a task is `blocked`, read the `last_note` and either
   reassign it or unblock it by sending a `coop-send` message.
4. When all tasks are `done`, integrate them: read each member's
   patch under `/workspace/shared/` (members are encouraged to write
   their diff and notes there), verify the merged tree compiles, and
   write your final `patch.txt`.

You **may also pick up tasks yourself** if it would unblock the team
faster than waiting.  But your default mode is to assign, not
implement.

{_TEAM_LIST_USAGE}

A shared scratchpad volume is mounted at `/workspace/shared/`.  Drop
design notes, interface contracts, or member-produced patches there
so the whole team can see them with `ls /workspace/shared/`."""


def _member_block(agent_id: str, lead: str) -> str:
    return f"""## You are a team member

You are **{agent_id}**.  The team-lead is **{lead}**, who will
organize work into tasks for you to claim.

Recommended workflow:

1. Run `coop-task-list --open` to see what needs doing.  If your
   `agent_id` appears as a pre-assigned `owner` on a task, that's the
   one the lead expects you to take.
2. `coop-task-claim <task_id>` it.  If you lose the race (exit code 2),
   the task is taken — pick another.
3. Implement.  When you hit a blocker, run
   `coop-task-update <task_id> blocked -n "<what you need>"` and
   `coop-send {lead} "blocked on task <id>: ..."`.
4. When done, write your contribution to `/workspace/shared/<your-id>.patch`
   AND your local `patch.txt`, then
   `coop-task-update <task_id> done -n "patch at /workspace/shared/<your-id>.patch"`.

{_TEAM_LIST_USAGE}

A shared scratchpad volume is mounted at `/workspace/shared/`.  Use
it for anything your peers might need to see — partial diffs,
interface sketches, error logs from your reproduction script."""


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
