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
    return f"""## Git collaboration

A shared git remote named `team` is already configured in this repo.
Use it to share code with your peers — messages alone aren't enough
when you both touch the same files.

- Your branch: `{agent_id}` (already created and pushed)
- Partner branches: {partner_branches}
- Base reference: `team/main` (pristine starting state)

Recommended workflow:

```bash
# See what your peers have published:
git fetch team
git branch -r
git log team/{first_partner} --oneline -10

# Inspect a peer's change without merging:
git show team/{first_partner} -- path/to/file

# Pull a peer's work into your tree.  Resolve any merge conflicts in
# the working tree, then commit.  If the merge is clean, --no-edit
# takes the default commit message and you're done.
git fetch team && git merge --no-edit team/{first_partner}

# Publish your own progress (after committing locally):
git add -A
git commit -m 'wip: <one-line summary>'
git push team {agent_id}
```

Concretely: before submitting, run `git fetch team` and look at every
peer branch.  If one of them edited a file you also edited, merge
their branch into yours and rebuild `patch.txt` from the merged tree
(`git diff team/main..HEAD > patch.txt`) so your submission contains
both your work and theirs."""


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
