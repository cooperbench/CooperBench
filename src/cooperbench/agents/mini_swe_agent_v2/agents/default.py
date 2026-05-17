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
        "Please summarize the conversation above. Include: the original task, "
        "key findings, files examined or modified, commands run and their results, "
        "decisions made, and current status. Be thorough — the agent will continue "
        "working from your summary without access to the original history."
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
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions. Polls for inter-agent messages
        and (in team mode) the shared task list before querying."""
        # Check for inter-agent messages before querying LLM
        if self.comm:
            messages = self.comm.receive()
            for msg in messages:
                ts = msg.get("timestamp", "")[:19].replace("T", " ")
                self.log(f"INBOX: [{msg['from']} @ {ts}] {msg['content']}")
                self.add_messages(
                    self.model.format_message(
                        role="user",
                        content=f"[Message from {msg['from']}]: {msg['content']}",
                    )
                )
        # In team mode, also refresh the shared task list so the LLM
        # sees the live state of who's working on what before its next
        # response.  ``team_poller`` is set by the adapter when team
        # kwargs are present; absent for solo/coop.
        poller = getattr(self, "team_poller", None)
        if poller is not None:
            summary = poller.poll()
            if summary:
                self.add_messages(self.model.format_message(role="user", content=summary))
        return self.execute_actions(self.query())

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
            if self.comm:
                sm_matches = _parse_send_messages(cmd)
                if sm_matches:
                    sm_outputs = []
                    for recipient, content, wait in sm_matches:
                        r = self._handle_send_message({"recipient": recipient, "content": content, "wait": wait})
                        sm_outputs.append(r["output"])
                    remaining = _strip_send_message(cmd)
                    combined = "\n".join(sm_outputs)
                    if not remaining.strip():
                        outputs.append({"output": combined, "returncode": 0, "exception_info": ""})
                        continue
                    env_out = self.env.execute({**action, "command": remaining})
                    env_out["output"] = combined + "\n" + env_out.get("output", "")
                    outputs.append(env_out)
                    continue

            outputs.append(self.env.execute(action))
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _handle_send_message(self, action: dict) -> dict:
        """Handle a send_message call via the messaging connector.

        ``wait=True`` (when the agent wrote ``send_message --wait ...`` in
        bash) uses ``send_and_wait`` so the peer's reply comes back in the
        same tool output.
        """
        recipient = action.get("recipient", "")
        content = action.get("content", "")
        wait = action.get("wait", False)

        if wait and hasattr(self.comm, "send_and_wait"):
            replies = self.comm.send_and_wait(recipient, content, timeout=60)
            self.log(f"SENT (blocking) to {recipient}: {content[:80]}...")
            self.sent_messages.append({"to": recipient, "content": content})
            output = f"Message sent to {recipient}"
            for r in replies or []:
                output += f"\n\n[Reply from {r['from']}]: {r['content']}"
            return {"output": output, "returncode": 0, "exception_info": ""}

        self.comm.send(recipient, content)
        self.log(f"SENT to {recipient}: {content[:80]}...")
        self.sent_messages.append({"to": recipient, "content": content})
        return {"output": f"Message sent to {recipient}", "returncode": 0, "exception_info": ""}

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
