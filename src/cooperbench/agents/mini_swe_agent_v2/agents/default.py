"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import re
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from cooperbench.agents.mini_swe_agent_v2 import Environment, Model, __version__
from cooperbench.agents.mini_swe_agent_v2.connectors.messaging import MessagingConnector
from cooperbench.agents.mini_swe_agent_v2.exceptions import InterruptAgentFlow, LimitsExceeded
from cooperbench.agents.mini_swe_agent_v2.utils.serialize import recursive_merge

# Name of the shared git remote created by GitConnector.  Referenced in the messages the
# agent sees, so it must match GitConnector.REMOTE_NAME.
GIT_REMOTE = "origin"


class AgentConfig(BaseModel):
    """Check the config files in config/ for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    compaction_enabled: bool = True
    """Enable context compaction (summarization of old messages)."""
    compaction_token_trigger: int = 28000
    """Compact when prompt token count exceeds this threshold."""
    compaction_keep_recent_turns: int = 2
    """Number of recent assistant turns to keep verbatim after compaction."""
    compaction_summary_prompt: str = (
        "You are summarizing the transcript below so the agent can continue in a "
        "fresh context without re-running commands. You are an outside observer "
        "of the conversation, not its next participant — output the summary as a "
        "single text response.\n"
        "\n"
        "The system prompt and the original task are preserved separately — DO NOT "
        "restate them. Focus on what would otherwise be LOST when the prior turns "
        "are discarded.\n"
        "\n"
        "Cite only what actually appears in the transcript. Do NOT invent line "
        "counts, file sizes, line numbers, or file contents you did not see. "
        "If you don't know a value, omit the field rather than guessing.\n"
        "\n"
        "Output ONLY the summary, using these exact section headings. Quote "
        "verbatim where stated; paraphrasing forces the agent to re-run commands.\n"
        "\n"
        "## FILE MAP\n"
        "One line per file the agent has touched (read, edited, or referenced):\n"
        "    `<path>`: <one-phrase description of what it contains>. Read so far: "
        '<line ranges or "all" or "first N lines" — only if known from `wc`, '
        "`head`, `tail`, or `sed -n` output above>. Modified: <yes/no>.\n"
        "Include `<line_count> lines` ONLY if a `wc -l` was actually run on it.\n"
        "\n"
        "## RELEVANT CODE READ\n"
        "For each file the agent read, quote the lines that matter for the task, "
        "verbatim, with citations:\n"
        "\n"
        "    `<path>` lines `<a>-<b>`: <one-line note on why this region matters>\n"
        "    ```\n"
        "    <verbatim snippet — exact indentation>\n"
        "    ```\n"
        "\n"
        "Include the parts likely needed again (definitions, key call sites, "
        "similar patterns nearby, struct shapes). Skip true boilerplate (license "
        "headers, unused imports). If a file was read and nothing in it matters, "
        "write one line: `<path>: read, nothing relevant`.\n"
        "\n"
        "## KEY SYMBOLS / IDENTIFIERS\n"
        "Flat list of important names the agent has discovered, with locations:\n"
        "`<name>` -> `<path>:<line>` — <one-phrase role>.\n"
        "Only include locations actually seen in the transcript.\n"
        "\n"
        "## SEARCH RESULTS WORTH KEEPING\n"
        "For each grep/find/ls/git-log: command + only the matches that matter:\n"
        "    `<cmd>` -> `<file>:<line>: <verbatim match>`.\n"
        "Also note negative results: `<cmd>` -> no matches.\n"
        "\n"
        "## EDITS ALREADY APPLIED\n"
        "For each edit (sed/echo>/cat-heredoc/python-write):\n"
        "    `<path>` <one-line description>:\n"
        "    ```diff\n"
        "    - <before>\n"
        "    + <after>\n"
        "    ```\n"
        "\n"
        "## BUILD / TEST OUTPUT\n"
        "For each test/build run: command, exit code, verbatim error lines "
        "(with `<file>:<line>:` refs). Skip pass noise.\n"
        "\n"
        "## COLLEAGUE MESSAGES\n"
        "Every send_message sent and `[Message from …]` received, verbatim, "
        "chronological.\n"
        "\n"
        "## OPEN QUESTIONS / UNREAD REGIONS\n"
        "What still needs investigation. Be specific — list file paths + line "
        "ranges + what to look for, so the next read is targeted not exploratory.\n"
        "\n"
        "## CURRENT PLAN\n"
        "The most recent stated plan and the immediate next intended step.\n"
        "\n"
        "Sections with no content: write `(none)`. Begin with `## FILE MAP` — "
        "no preamble."
    )
    """Prompt appended to conversation history when requesting a summary."""


class DefaultAgent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        comm: MessagingConnector | None = None,
        agent_id: str = "agent",
        config_class: type = AgentConfig,
        **kwargs,
    ):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.comm = comm
        self.agent_id = agent_id
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.sent_messages: list[dict] = []
        # Compaction state
        self._last_prompt_tokens: int = 0
        self._compaction_count: int = 0
        self._segments: list[dict] = []
        self._current_segment_messages: list[dict] = []

    def log(self, msg: str):
        """Log message with agent prefix."""
        self.logger.debug(f"[{self.agent_id}] {msg}")

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        try:
            while True:
                try:
                    self.step()
                except InterruptAgentFlow as e:
                    self.add_messages(*e.messages)
                except Exception as e:
                    self.handle_uncaught_exception(e)
                    raise
                finally:
                    self.save(self.config.output_path)
                if self.messages[-1].get("role") == "exit":
                    if self._nudge_unsubmitted():
                        continue
                    break
        finally:
            # Announce departure on EVERY exit path, not just a clean submit. A crash or a
            # step-limit exit used to leave this unset, so the peer's has_exited() stayed False
            # and it waited on someone who was never coming back -- one agent burned 91 minutes
            # of sleep that way and outlived its own sandbox.
            if self.comm:
                # `published` means "the peer can see my work on the remote", i.e. a PR is open.
                try:
                    published = self._opened_pr()
                except Exception:
                    published = False
                self.comm.mark_exited(published=published)
        return self.messages[-1].get("extra", {})

    MAX_SUBMIT_NUDGES = 2

    def _nudge_unsubmitted(self) -> bool:
        """Tell an agent that is ending with unshared work, instead of letting it end.

        `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and actually submitting are separate
        acts, so an agent can finish the task, verify its tests, announce "all changes are
        complete", fire the sentinel -- and have published nothing. Grading sees an empty
        patch, indistinguishable from an agent that never wrote any code. Measured: one agent
        did this after 943 steps of real work.

        Returns True if the episode should continue rather than exit.
        """
        if not self.comm:  # solo has no peer branch to compare against; left as-is
            return False
        if getattr(self, "_submit_nudges", 0) >= self.MAX_SUBMIT_NUDGES:
            return False  # it was told twice; ending with nothing is now its own choice
        if self._opened_pr():
            return False

        agent = self.comm.agent_id
        dirty = (self.env.execute({"command": "git status --porcelain"}).get("output") or "").strip()
        ahead = (
            self.env.execute({"command": f"git rev-list --count {GIT_REMOTE}/{agent}..HEAD 2>/dev/null"})
            .get("output", "")
            .strip()
        )
        unpushed = ahead not in ("", "0")
        if not dirty and not unpushed:
            return False  # nothing to lose: it genuinely has no work to submit

        self._submit_nudges = getattr(self, "_submit_nudges", 0) + 1
        held = "uncommitted changes" if dirty else "commits that are not pushed"
        self.log(f"SUBMIT NUDGE {self._submit_nudges} for {agent}: {held}, no PR")
        self.add_messages(
            self.model.format_message(
                role="user",
                content=(
                    f"You have not submitted anything. Your work is still local ({held}) and "
                    f"no pull request exists, so it would be graded as an empty submission.\n\n"
                    f"    git add <the files you want to submit>\n"
                    f'    git commit -m "..."\n'
                    f"    git push {GIT_REMOTE} HEAD:{agent}\n"
                    f'    gh pr create --title "..." --body "..."\n\n'
                    f"Then echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT again."
                ),
            )
        )
        return True

    def _opened_pr(self) -> bool:
        """Whether this agent's PR exists on the shared remote.

        Replaces `_publish_final_work`, which pushed `patch.txt` to the agent's branch at
        exit. Submission is now a PR the agent opens itself, so there is nothing left to
        publish -- that method only logged `no patch.txt to publish` on every run.
        """
        if not self.comm:
            return False
        r = self.env.execute({"command": f"git ls-remote --tags {GIT_REMOTE} refs/tags/pr/{self.comm.agent_id}"})
        return bool((r.get("output") or "").strip())

    def step(self) -> list[dict]:
        """Query the LM, execute actions. Polls for inter-agent messages
        and (in team mode) the shared task list before querying."""
        # Check for inter-agent messages before querying LLM
        if self.comm:
            if hasattr(self.comm, "heartbeat"):
                self.comm.heartbeat()
            messages = self.comm.receive()
            for msg in messages:
                ts = msg.get("timestamp", "")[:19].replace("T", " ")
                self.log(f"INBOX: [{msg['from']} @ {ts}] {msg['content']}")
                # Say so when the sender is stalled on us. Without it the message reads as
                # ordinary context and gets deprioritised behind the agent's own edit loop.
                blocked = ""
                if getattr(self.comm, "is_awaited_by", None) and self.comm.is_awaited_by(msg["from"]):
                    blocked = (
                        f" — {msg['from']} is BLOCKED waiting for your reply and cannot continue"
                        f" until you answer. Reply with send_message before your next command."
                    )
                self.add_messages(
                    self.model.format_message(
                        role="user",
                        content=f"[Message from {msg['from']}{blocked}]: {msg['content']}",
                    )
                )
            self._announce_departed_peers()
        # In team mode, also refresh the shared task list so the LLM
        # sees the live state of who's working on what before its next
        # response.  ``team_poller`` is set by the adapter when team
        # kwargs are present; absent for solo/coop.
        poller = getattr(self, "team_poller", None)
        if poller is not None:
            summary = poller.poll()
            if summary:
                self.add_messages(self.model.format_message(role="user", content=summary))
            notice = self._team_conflict_notice()
            if notice:
                self.add_messages(self.model.format_message(role="user", content=notice))
        return self.execute_actions(self.query())

    def _team_conflict_notice(self) -> str | None:
        """Just-in-time merge-conflict notice via an in-container 3-way
        trial merge of teammates' published diffs against this agent's
        working tree.  Returns None on any failure so the agent loop
        never breaks because the probe couldn't run."""
        try:
            from cooperbench.team_harness.jit_merge import (
                build_probe_command,
                format_conflict_notice,
                parse_conflicts,
            )

            out = self.env.execute({"command": build_probe_command(self.agent_id)})
            conflicts = parse_conflicts(out.get("output") or "")
            return format_conflict_notice(conflicts) or None
        except Exception:
            return None

    def _team_read_tasks(self) -> list[dict] | None:
        """Return the live task list via the team poller's redis client,
        or ``None`` if team mode isn't wired or Redis is unreachable.

        Centralised so the gate / prefix / blocking helpers all read the
        same state and fail uniformly (returning None never breaks the
        loop)."""
        poller = getattr(self, "team_poller", None)
        if poller is None:
            return None
        try:
            from cooperbench.team_harness.loop_refresh import _read_tasks

            client = poller._ensure_client()  # type: ignore[attr-defined]
            if client is None:
                return None
            return _read_tasks(client, poller._run_id)  # type: ignore[attr-defined]
        except Exception:
            return None

    def _team_required_actions(self, cmd: str) -> list[dict]:
        """Coordination actions to auto-apply *before* the agent's command.

        Two cases (each fires once until its condition flips):

        - Unclaimed-own-task: if any of my tasks is still ``status=open``
          and the command isn't itself a claim, prepend a claim.
        - Submit-with-in-progress: if the command is a final submit and any
          of my tasks is still ``status=in_progress``, prepend an update to
          mark it done first.

        Returns a list of action dicts of the form
        ``{"kind": "claim"|"update", "task_id": ..., "title": ..., ...}``.
        Pure decision function — does not mutate state.
        """
        if not cmd:
            return []
        tasks = self._team_read_tasks()
        if not tasks:
            return []

        mine = [t for t in tasks if t.get("owner") == self.agent_id]
        actions: list[dict] = []

        if "coop-task-claim" not in cmd:
            for t in mine:
                if t.get("status") == "open":
                    actions.append({"kind": "claim", "task_id": t["id"], "title": t.get("title", "")})

        is_submit = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd
        if is_submit and "coop-task-update" not in cmd:
            for t in mine:
                if t.get("status") == "in_progress":
                    title = (t.get("title") or "")[:60].replace("'", "").replace("\n", " ")
                    actions.append({"kind": "update", "task_id": t["id"], "status": "done", "note": f"auto: {title}"})

        return actions

    def _team_required_prefix(self, cmd: str) -> list[str]:
        """Back-compat shim: render required actions as CLI strings.

        Some tests assert the CLI-string shape directly.  Production uses
        ``_team_apply_prefix`` which mutates Redis server-side.
        """
        out: list[str] = []
        for a in self._team_required_actions(cmd):
            if a["kind"] == "claim":
                out.append(f"coop-task-claim {a['task_id']}")
            elif a["kind"] == "update":
                out.append(f"coop-task-update {a['task_id']} {a['status']} -n '{a['note']}'")
        return out

    def _team_apply_prefix(self, cmd: str) -> str | None:
        """Apply required coordination actions server-side via the host
        TaskListClient.  Returns a synthesized tool-output string to
        prepend to the agent's observation, or ``None`` if nothing was
        applied.

        Bypasses the in-container ``coop-task-*`` shell CLI (which needs
        the ``redis`` python module — not always available in task base
        images, the install snippet silently no-ops if pip can't reach it).
        The audit-log events still land in Redis so ``task_log.json`` and
        downstream ``conversation.json`` see the claim/update events.
        """
        actions = self._team_required_actions(cmd)
        if not actions:
            return None
        poller = getattr(self, "team_poller", None)
        if poller is None:
            return None
        try:
            from cooperbench.team_harness.task_list import TaskListClient

            client = poller._ensure_client()  # type: ignore[attr-defined]
            if client is None:
                return None
            tlc = TaskListClient(redis_client=client, run_id=poller._run_id)  # type: ignore[attr-defined]
        except Exception:
            return None

        chunks: list[str] = []
        for a in actions:
            try:
                if a["kind"] == "claim":
                    ok = tlc.claim(a["task_id"], by=self.agent_id)
                    if ok:
                        chunks.append(
                            f"$ coop-task-claim {a['task_id']}\n[auto] claimed: {a.get('title', '')}".rstrip()
                        )
                elif a["kind"] == "update":
                    tlc.update(a["task_id"], by=self.agent_id, status=a["status"], note=a["note"])
                    chunks.append(f'$ coop-task-update {a["task_id"]} {a["status"]} -n "{a["note"]}"\n[auto] updated')
            except Exception:
                continue
        return "\n".join(chunks) if chunks else None

    def _team_blocking_reason(self, cmd: str) -> str | None:
        """Return a refusal message if the command must be blocked outright.

        Only one case is auto-fix-impossible: a lead trying to submit while
        a peer's task is not yet ``status=done``.  The lead has nothing to
        auto-execute — it has to wait for the peer to update.
        """
        if not cmd:
            return None
        is_submit = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd
        if not is_submit:
            return None
        tasks = self._team_read_tasks()
        if not tasks:
            return None
        mine = [t for t in tasks if t.get("owner") == self.agent_id]
        if not mine:
            return None
        am_lead = any("Lead-only" in (t.get("title") or "") for t in mine)
        if not am_lead:
            return None
        peer_open = [
            t for t in tasks if t.get("owner") and t.get("owner") != self.agent_id and t.get("status") != "done"
        ]
        if not peer_open:
            return None
        lines = [
            f"{t.get('id', '?')} [{t.get('status', '?')}] owner={t.get('owner', '?')}: {t.get('title', '')}"
            for t in peer_open
        ]
        return "[coord-gate] Cannot submit yet: peer task(s) not yet done.\n  " + "\n  ".join(lines)

    def _team_coord_gate(self, cmd: str) -> dict | None:
        """Legacy combined gate, retained for backward-compat tests.

        Returns a refusal observation if any rule applies, or ``None``.
        Equivalent to the old behavior before split into auto-prefix +
        blocking-reason.  Production execution uses the split helpers
        directly so it can auto-execute the prefix rather than refuse.
        """
        poller = getattr(self, "team_poller", None)
        if poller is None or not cmd:
            return None
        try:
            from cooperbench.team_harness.loop_refresh import _read_tasks

            client = poller._ensure_client()  # type: ignore[attr-defined]
            if client is None:
                return None
            tasks = _read_tasks(client, poller._run_id)  # type: ignore[attr-defined]
        except Exception:
            return None
        if not tasks:
            return None

        mine = [t for t in tasks if t.get("owner") == self.agent_id]
        others = [t for t in tasks if t.get("owner") and t.get("owner") != self.agent_id]

        # Rule 1: unclaimed-own-task gate
        unclaimed = [t for t in mine if t.get("status") == "open"]
        if unclaimed and "coop-task-claim" not in cmd:
            ids = ", ".join(t.get("id", "?") for t in unclaimed)
            msg = (
                f"[coord-gate] You have unclaimed task(s) assigned to you: {ids}. "
                f"Claim before running other commands:  coop-task-claim <task_id>"
            )
            return {"output": msg, "returncode": 1, "exception_info": ""}

        is_submit = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd

        # Rule 2: own-task-not-done gate (on submit)
        if is_submit:
            in_prog = [t for t in mine if t.get("status") == "in_progress"]
            if in_prog:
                ids = ", ".join(t.get("id", "?") for t in in_prog)
                msg = (
                    f"[coord-gate] Cannot submit: your task(s) {ids} still in_progress. "
                    f'Mark done first:  coop-task-update <task_id> done -n "<summary>"'
                )
                return {"output": msg, "returncode": 1, "exception_info": ""}

            # Rule 3: peer-not-done gate (on submit) — applies to lead only
            # in practice; member submissions are independent of peers.
            peer_open = [t for t in others if t.get("status") != "done"]
            if peer_open and mine:  # only gate if I'm a coordinator (have my own task)
                # Heuristic: a "lead" task has "Lead-only" in its title (per the
                # current task-creation prompts).  If none of my tasks look like
                # a lead task, don't gate on peer state — members can submit
                # independently.
                am_lead = any("Lead-only" in (t.get("title") or "") for t in mine)
                if am_lead:
                    open_lines = [
                        f"{t.get('id', '?')} [{t.get('status', '?')}] owner={t.get('owner', '?')}: {t.get('title', '')}"
                        for t in peer_open
                    ]
                    msg = (
                        "[coord-gate] Cannot submit: peer task(s) not yet done. "
                        "Wait for them to update, then integrate:\n  " + "\n  ".join(open_lines)
                    )
                    return {"output": msg, "returncode": 1, "exception_info": ""}

        return None

    def _get_prompt_tokens(self, message: dict) -> int:
        return message.get("extra", {}).get("response", {}).get("usage", {}).get("prompt_tokens", 0)

    def _should_compact(self) -> bool:
        return self.config.compaction_enabled and self._last_prompt_tokens >= self.config.compaction_token_trigger

    @staticmethod
    def _find_turn_boundary(messages: list[dict], n_turns: int) -> int:
        """Return the index where the last n_turns complete assistant turns start."""
        assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if not assistant_indices or n_turns <= 0:
            return len(messages)
        start = max(0, len(assistant_indices) - n_turns)
        return assistant_indices[start]

    def _close_current_segment(self, kind: str = "solver") -> None:
        """Append accumulated messages as a named segment and reset the buffer."""
        msgs = self._current_segment_messages or self.messages
        if msgs:
            self._segments.append({"kind": kind, "messages": list(msgs)})
            self._current_segment_messages = []

    def _compact_messages(self) -> None:
        """Summarize old messages and replace history, keeping recent turns verbatim."""
        summarize_fn = getattr(self.model, "summarize_context", None)
        if not callable(summarize_fn):
            self.log("Model does not support summarize_context, skipping compaction")
            return

        prefix = self.messages[:2]  # system + task
        conversation = self.messages[2:]
        boundary = self._find_turn_boundary(conversation, self.config.compaction_keep_recent_turns)
        old_turns = conversation[:boundary]
        recent_turns = conversation[boundary:]

        if not old_turns:
            return

        self._close_current_segment("solver")

        summarizer_input = prefix + old_turns
        summary_msg = summarize_fn(
            summarizer_input,
            summary_prompt=self.config.compaction_summary_prompt,
        )
        self._segments.append(
            {
                "kind": "summarizer",
                "messages": [
                    *[{k: v for k, v in m.items() if k != "extra"} for m in summarizer_input],
                    {"role": "user", "content": self.config.compaction_summary_prompt},
                    summary_msg,
                ],
            }
        )

        self.messages = prefix + [summary_msg] + recent_turns
        self._compaction_count += 1
        self.log(
            f"Compaction #{self._compaction_count}: {self._last_prompt_tokens} prompt tokens -> compacted "
            f"({len(old_turns)} messages summarized, {len(recent_turns)} kept)"
        )

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if self._should_compact():
            self._compact_messages()
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self._last_prompt_tokens = self._get_prompt_tokens(message)
        self.add_messages(message)
        self._current_segment_messages = list(self.messages)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them.

        Only the ``bash`` tool is registered with the model (see adapter.py) —
        ``send_message`` is invoked by the agent embedding a shell command
        like ``send_message <recipient> <<'MSG' ... MSG`` inside the bash
        command string.  We parse any such calls out of the command, run
        them through the messaging connector, and execute the remainder (if
        any) against the docker env.  Single-tool registration is much
        more reliable for smaller models than exposing two tools.
        """
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for action in actions:
            tool_name = action.get("tool_name", "bash")
            if tool_name == "send_message" and self.comm:
                # Defensive: supported for legacy callers that still
                # register send_message as a tool.
                outputs.append(self._handle_send_message(action))
                continue

            cmd = action.get("command", "")

            # Team-mode coordination gate: block what can't be auto-fixed,
            # auto-execute required prefix commands (claim, update) for the
            # rest.  See _team_blocking_reason / _team_required_prefix.
            blocked = self._team_blocking_reason(cmd)
            if blocked is not None:
                outputs.append({"output": blocked, "returncode": 1, "exception_info": ""})
                continue

            prefix_text = self._team_apply_prefix(cmd) or ""

            if self.comm:
                sm_matches = _parse_send_messages(cmd)
                if not sm_matches and re.search(r"\bsend_message\b", cmd):
                    # send_message is parsed out of the command string, not a real binary, so an
                    # unparsed one reaches bash and dies as `command not found` -- and the agent
                    # reads that as "messaging is broken" rather than "my quoting was wrong".
                    # Measured: 7 lost sends across two runs (bare --wait with no body,
                    # unterminated heredoc, `< file` redirect, `-t 60`, `timeout N bash -c ...`).
                    outputs.append(
                        {
                            "output": (
                                "send_message: could not parse that call, so nothing was sent.\n"
                                "Use one of:\n"
                                "  send_message [--wait] <agent> <<'MSG'\n  your text\n  MSG\n"
                                '  send_message [--wait] <agent> "your text"\n'
                                "The message body is required, and --wait takes no argument."
                            ),
                            "returncode": 1,
                            "exception_info": "",
                        }
                    )
                    continue
                if sm_matches:
                    sm_outputs = []
                    for recipient, content, wait in sm_matches:
                        r = self._handle_send_message({"recipient": recipient, "content": content, "wait": wait})
                        sm_outputs.append(r["output"])
                    remaining = _strip_send_message(cmd)
                    combined = "\n".join(sm_outputs)
                    if not remaining.strip():
                        output = combined
                        if prefix_text:
                            output = prefix_text + "\n" + output
                        outputs.append({"output": output, "returncode": 0, "exception_info": ""})
                        continue
                    env_out = self.env.execute({**action, "command": remaining})
                    output = combined + "\n" + env_out.get("output", "")
                    if prefix_text:
                        output = prefix_text + "\n" + output
                    env_out["output"] = output
                    outputs.append(env_out)
                    continue

            env_out = self.env.execute(action)
            if prefix_text:
                env_out["output"] = prefix_text + "\n" + (env_out.get("output") or "")
            outputs.append(env_out)
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _announce_departed_peers(self) -> None:
        """Tell the agent, once, when a peer has finished and left.

        Otherwise an agent can spend the rest of its run waiting for an answer from a
        colleague who is no longer running, with nothing in its context to indicate that.
        Announced once per peer; the recovery path is named because the peer's work is
        still reachable through git even though the conversation is over.
        """
        if not self.comm:
            return
        announced = getattr(self, "_departed_announced", None)
        if announced is None:
            announced = self._departed_announced = set()
        for peer in getattr(self.comm, "agents", []) or []:
            if peer == self.comm.agent_id or peer in announced:
                continue
            if not self.comm.has_exited(peer):
                continue
            announced.add(peer)
            gone = getattr(self.comm, "is_unreachable", None) and self.comm.is_unreachable(peer)
            self.log(f"PEER {'UNREACHABLE' if gone else 'EXITED'}: {peer}")
            headline = (
                f"[{peer} is no longer running] They stopped without submitting, so nothing "
                f"further will arrive from them."
                if gone
                else f"[{peer} has completed their work and exited] They will not read or answer further messages."
            )
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content=f"{headline}\n\n{self._peer_work_pointer(peer)}",
                )
            )

    def _peer_work_pointer(self, peer: str) -> str:
        """Where to find a departed peer's work — only claimed when it is really there.

        Publication to the shared remote is best-effort, so asserting the branch holds
        their submission when it does not would repeat exactly the failure this whole
        change exists to remove: telling the agent something untrue and letting it act
        on it.
        """
        if self.comm and self.comm.has_published(peer):
            return (
                f"Their submitted patch is on branch {GIT_REMOTE}/{peer}. If your changes "
                f"overlap theirs, inspect and reconcile before you submit:\n"
                f"    git fetch {GIT_REMOTE} && git diff HEAD...{GIT_REMOTE}/{peer}"
            )
        return (
            f"Their work could NOT be published to {GIT_REMOTE}/{peer}, so that branch does "
            f"not reflect what they submitted — do not rely on it. Proceed on your own "
            f"judgement and keep your changes as self-contained as you can."
        )

    def _handle_send_message(self, action: dict) -> dict:
        """Handle a send_message call via the messaging connector.

        ``wait=True`` (when the agent wrote ``send_message --wait ...`` in bash) blocks
        until the peer replies, the peer exits, or the timeout elapses.

        A send to a peer that has already finished is reported as a failure, not a
        success.  Reporting success there is actively misleading: the agent believes it
        has coordinated, keeps waiting for an answer that cannot arrive, and ships work
        that was never reconciled.
        """
        recipient = action.get("recipient", "")
        content = action.get("content", "")
        wait = action.get("wait", False)

        if wait and hasattr(self.comm, "send_and_wait"):
            delivered, replies = self.comm.send_and_wait(recipient, content, timeout=60)
            if not delivered:
                return self._peer_gone_result(recipient)
            self.log(f"SENT (blocking) to {recipient}: {content[:80]}...")
            self.sent_messages.append({"to": recipient, "content": content})
            output = f"Message sent to {recipient}"
            if replies:
                for r in replies:
                    output += f"\n\n[Reply from {r['from']}]: {r['content']}"
            elif self.comm.has_exited(recipient):
                output += (
                    f"\n\nNo reply: {recipient} has since completed their work and exited.\n"
                    f"{self._peer_work_pointer(recipient)}"
                )
            return {"output": output, "returncode": 0, "exception_info": ""}

        if not self.comm.send(recipient, content):
            return self._peer_gone_result(recipient)
        self.log(f"SENT to {recipient}: {content[:80]}...")
        self.sent_messages.append({"to": recipient, "content": content})
        return {"output": f"Message sent to {recipient}", "returncode": 0, "exception_info": ""}

    def _peer_gone_result(self, recipient: str) -> dict:
        """Tell the agent its peer is gone, and what to do about it.

        Phrased as a terminal state rather than a delivery error so the agent does not
        retry, and names the recovery path so the peer's work is still reachable.
        """
        self.log(f"NOT DELIVERED to {recipient}: peer already exited")
        return {
            "output": (
                f"{recipient} has already completed their work and exited. Your message was "
                f"NOT delivered and no reply will come — do not send further messages to "
                f"them.\n\n{self._peer_work_pointer(recipient)}"
            ),
            "returncode": 1,
            "exception_info": "",
        }

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        if self._compaction_count > 0:
            segments = list(self._segments)
            current = self._current_segment_messages or self.messages
            if current:
                segments.append({"kind": "solver", "messages": list(current)})
            agent_data["segments"] = segments
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data


def _parse_send_messages(cmd: str) -> list[tuple[str, str, bool]]:
    """Extract (recipient, content, wait) tuples from send_message calls.

    ``--wait`` may appear before or after the recipient.  Supports three
    formats: heredoc (``<<'MSG'``), double-quoted, single-quoted.
    """
    matches: list[tuple[str, str, bool]] = []
    for m in re.finditer(
        r"send_message\s+(--wait\s+)?(\w+)(\s+--wait)?\s+<<'?(\w+)'?\s*\n(.*?)\n\4",
        cmd,
        re.DOTALL,
    ):
        wait = bool(m.group(1) or m.group(3))
        matches.append((m.group(2), m.group(5), wait))
    if not matches:
        for m in re.finditer(r'send_message\s+(--wait\s+)?(\w+)(\s+--wait)?\s+"([^"]*)"', cmd):
            wait = bool(m.group(1) or m.group(3))
            matches.append((m.group(2), m.group(4), wait))
        for m in re.finditer(r"send_message\s+(--wait\s+)?(\w+)(\s+--wait)?\s+'([^']*)'", cmd):
            wait = bool(m.group(1) or m.group(3))
            matches.append((m.group(2), m.group(4), wait))
    return matches


def _strip_send_message(cmd: str) -> str:
    """Remove send_message calls from a compound bash command."""
    cmd = re.sub(
        r"send_message\s+(--wait\s+)?\w+(\s+--wait)?\s+<<'?(\w+)'?\s*\n.*?\n\3",
        "",
        cmd,
        flags=re.DOTALL,
    )
    cmd = re.sub(r'send_message\s+(--wait\s+)?\w+(\s+--wait)?\s+"[^"]*"', "", cmd)
    cmd = re.sub(r"send_message\s+(--wait\s+)?\w+(\s+--wait)?\s+'[^']*'", "", cmd)
    cmd = re.sub(r"^\s*&&\s*", "", cmd)
    cmd = re.sub(r"\s*&&\s*$", "", cmd)
    cmd = re.sub(r"&&\s*&&", "&&", cmd)
    cmd = re.sub(r"\|\|\s*\|\|", "||", cmd)
    return cmd.strip()
