"""Instruction templates for solo and coop modes.

Claude Code doesn't speak the ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
sentinel that mini-swe-agent v2 uses — it just exits when it considers
the work done.  All we need is for the agent to write its unified diff
to ``patch.txt`` in the repo before exiting; the adapter reads the file
post-run.  In coop mode we additionally document the coop-* messaging
helpers so the agent can coordinate with its peers.
"""

from __future__ import annotations

_SUBMISSION_BLOCK = """## Submission protocol

When you are done editing the codebase, write your final unified diff to
`/workspace/repo/patch.txt` BEFORE exiting.  The bench evaluator reads
that file; nothing else is inspected.  A typical pattern:

```bash
cd /workspace/repo
git diff > patch.txt
cat patch.txt   # sanity-check that the diff is what you intend to submit
```

Constraints on `patch.txt`:

- Must be a unified diff (`git diff` output is fine).
- Should contain ONLY source files you intentionally modified to implement
  the feature.  Exclude reproduction scripts, scratch tests, or
  helper files you wrote during development.
- Do not include changes to test files unless the task explicitly asks
  you to modify tests.

You are free to read files, run shell commands, and run tests as needed."""


def _git_block(agent_id: str, partners: list[str]) -> str:
    partner_branches = ", ".join(f"`team/{p}`" for p in partners)
    first_partner = partners[0]
    partner_merge_lines = "\n".join(
        f"  git fetch team && git merge --no-edit team/{p} || true   # pull in {p}'s work"
        for p in partners
    )
    return f"""## Git collaboration — MERGE IS REQUIRED BEFORE SUBMITTING

A shared git remote named `team` is already configured in this repo.
- Your branch: `{agent_id}` (already created and pushed)
- Partner branches: {partner_branches}
- Base reference: `team/main` (pristine starting state)

### The submission rule the bench actually enforces

The bench evaluates each agent's `patch.txt` against EVERY feature's
test suite.  If your `patch.txt` only contains your own work, the
peer feature's tests will fail with `ImportError` because the
symbols they introduced aren't in your tree.  **You MUST pull in
your peers' branches before generating your final `patch.txt`** — or
your submission will be incomplete by construction.

### Required final sequence — run this verbatim before exiting

```bash
# 1. Commit your own work so it's on your branch tip.
cd /workspace/repo
git add -A
git commit -m 'wip: my work' || true   # ok if nothing to commit

# 2. Push so peers can fetch you (optional but recommended).
git push team {agent_id} --force || true

# 3. Pull in every peer's branch.  Use --no-edit to take the default
#    merge commit message.  || true so a clean-no-op doesn't abort.
{partner_merge_lines}

# 4. Rebuild patch.txt from the MERGED tree.  This is the artifact the
#    bench evaluates — it must contain both your work and your peers'.
git diff team/main..HEAD > patch.txt

# 5. Sanity-check: the diff should mention symbols you didn't write
#    yourself (your peers' contributions).
wc -l patch.txt
head -30 patch.txt
```

### During the run

```bash
git fetch team                                   # see what peers published
git branch -r                                    # list every team branch
git log team/{first_partner} --oneline -10       # inspect a peer's history
git show team/{first_partner} -- path/to/file    # inspect a peer's change
```

If you skip the merge step, you will lose points the bench would
otherwise have awarded.  The team-mode metric `tasks_done` only
measures coordination, not correctness — correctness comes from
`patch.txt` containing the union of the team's work."""


def _coop_block(agent_id: str, partners: list[str]) -> str:
    partner_str = ", ".join(partners)
    return f"""## Cooperation protocol

You are **{agent_id}**, working alongside: **{partner_str}**.
Each agent has been assigned a separate feature from the same codebase;
your features may overlap (touch the same files), so coordinate to avoid
clobbering each other's changes.

Available shell commands for cross-agent messaging (Redis-backed inbox,
one inbox per agent):

```bash
coop-send <recipient> "message text here"   # send to a specific peer
coop-broadcast "message text here"          # send to every other peer
coop-recv                                    # drain your inbox (prints JSON list)
coop-peek                                    # number of unread messages
coop-agents                                  # list every agent id
```

Recommended workflow:

1. At the start, `coop-broadcast` a short summary of your feature and
   which files you intend to touch.
2. Periodically `coop-recv` to read what your peers have sent — at
   minimum after major edits and before submitting.
3. If two agents need to modify the same file, coordinate explicitly
   (split the file, agree on one owner, or merge changes).
4. Keep messages short and focused: file names, function names, and
   one-sentence intents are usually enough.

Messages are not magic — your peers only know what you tell them.
"""


def build_instruction(
    task: str,
    *,
    agents: list[str] | None = None,
    agent_id: str | None = None,
    git_enabled: bool = False,
) -> str:
    """Compose the full instruction for a single agent run.

    Args:
        task: The raw feature spec (the user-facing task description).
        agents: All agent ids in the run.  When this has 2+ entries we
            emit the coop messaging block.
        agent_id: This agent's id.  Required when ``agents`` is multi.
        git_enabled: Whether the shared git remote is configured.  When
            true (and we're in coop mode), append a git collaboration
            section to the prompt.
    """
    partners: list[str] = []
    if agents and agent_id:
        partners = [a for a in agents if a != agent_id]
    sections = [task, _SUBMISSION_BLOCK]
    if partners and agent_id:
        sections.append(_coop_block(agent_id, partners))
        if git_enabled:
            sections.append(_git_block(agent_id, partners))
    return "\n\n---\n\n".join(sections)
