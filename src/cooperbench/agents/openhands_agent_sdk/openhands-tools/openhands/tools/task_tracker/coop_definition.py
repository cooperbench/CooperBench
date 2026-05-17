"""Redis-backed CoopTaskTracker tool for team-mode runs.

Same action / observation shape as ``TaskTrackerTool`` so the LLM
doesn't have to learn new semantics — same ``plan`` and ``view``
commands, same ``TaskItem`` list — but persistence is the shared
``cb:<run_id>:`` Redis namespace instead of a per-agent
``TASKS.json``.  This makes peer-visible task management appear as a
**typed tool** (which gpt-5.5 strongly prefers) rather than a shell
command.

Importing this module re-registers the existing ``TaskTrackerTool``
name to point at the Coop variant, so callers that already do
``Tool(name=TaskTrackerTool.name)`` (i.e. all of openhands) pick the
Redis-backed executor transparently.  The openhands adapter
(``ModalSandboxContext.__enter__``) injects this file into the Modal
sandbox's openhands install and appends a side-effect import to
``openhands.tools.task_tracker.__init__`` so the registration runs at
package-import time.

Wire shape on Redis matches what the host-side ``TaskListClient``
writes — see ``cooperbench/agents/_team/task_list.py`` for the
authoritative description.

KNOWN LIMITATION — Modal sandbox network isolation
---------------------------------------------------

In the current openhands_sdk deployment, the agent-server runs inside
a Modal sandbox that's network-isolated from the host where Redis
runs.  Connecting to ``host.docker.internal:6379`` (the rewritten URL
the host adapter passes via ``CB_TEAM_REDIS_URL``) fails with
``socket.getaddrinfo`` — Modal sandboxes aren't docker containers on
the host and don't have that hostname mapped.

So this tool is correctly registered and the LLM can call it, but
every operation will return an "Shared task list unavailable" error
until Redis is reachable from inside the Modal sandbox.  The fix is
in the deployment layer (Modal tunnels, a Modal-hosted Redis, or
running openhands directly via docker like the other adapters), not
here.

For non-Modal openhands deployments (e.g. local docker-backed
openhands runs, future remote-conversation transports that share the
host network), this tool works as designed and produces the expected
Redis-backed shared task list.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import (
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

from openhands.tools.task_tracker.definition import (
    TaskItem,
    TaskTrackerAction,
    TaskTrackerObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


logger = get_logger(__name__)


def _redis_client_from_env():
    """Open a redis connection from CB_TEAM_REDIS_URL.

    Imported lazily so ``redis`` is only a runtime dependency when team
    mode is active.
    """
    import redis  # type: ignore[import-not-found]

    url = os.environ["CB_TEAM_REDIS_URL"]
    if "#" in url:
        url, _ = url.split("#", 1)
    return redis.from_url(url, socket_timeout=5)


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


# Mapping between the openhands TaskTracker statuses (todo / in_progress /
# done) and the cooperbench shared list statuses (open / in_progress /
# blocked / done).  We collapse "blocked" → "in_progress" on read since
# the tool's enum doesn't have a corresponding value; that's a small
# loss of fidelity but keeps the tool surface stable.
_OH_TO_CB_STATUS = {"todo": "open", "in_progress": "in_progress", "done": "done"}
_CB_TO_OH_STATUS = {
    "open": "todo",
    "in_progress": "in_progress",
    "blocked": "in_progress",
    "done": "done",
}


class CoopTaskTrackerExecutor(ToolExecutor[TaskTrackerAction, TaskTrackerObservation]):
    """Executor that persists tasks to the shared Redis task list."""

    def __init__(self) -> None:
        self._run_id = os.environ.get("CB_TEAM_RUN_ID", "")
        self._agent_id = os.environ.get("CB_TEAM_AGENT_ID", "agent")
        self._ns = f"cb:{self._run_id}"
        logger.info(
            "CoopTaskTrackerExecutor: run_id=%s agent_id=%s",
            self._run_id,
            self._agent_id,
        )

    # --- Redis helpers (mirroring TaskListClient's wire shape) -----

    def _log(self, client, **event) -> None:
        event["ts"] = time.time()
        client.rpush(f"{self._ns}:task-log", json.dumps(event))

    def _read_all(self, client) -> list[TaskItem]:
        ids = sorted(_decode(m) for m in client.smembers(f"{self._ns}:tasks:all"))
        items: list[TaskItem] = []
        for tid in ids:
            raw = client.hgetall(f"{self._ns}:task:{tid}")
            if not raw:
                continue
            decoded = {_decode(k): _decode(v) for k, v in raw.items()}
            title = decoded.get("title", "")
            note = decoded.get("last_note", "")
            cb_status = decoded.get("status", "open")
            oh_status = _CB_TO_OH_STATUS.get(cb_status, "todo")
            # Prefix the title with the owner so peers' tasks are visible
            # in the LLM's view of the list — that's the whole point of
            # routing through Redis.
            owner = decoded.get("owner", "")
            display_title = f"[{owner or 'unassigned'}] {title}" if owner else title
            items.append(TaskItem(title=display_title, notes=note, status=oh_status))
        return items

    def _write_plan(self, client, requested: list[TaskItem]) -> None:
        """Reconcile the agent's requested task list with the shared one.

        Strategy: every ``plan`` call is treated as the agent's view of
        ITS OWN tasks.  We delete the agent's existing owned tasks and
        re-create them, leaving peers' tasks untouched.  This keeps the
        plan/view contract that openhands' TaskTracker offers while
        preventing one agent from clobbering another's work.
        """
        # 1. Delete tasks this agent currently owns.
        ids = sorted(_decode(m) for m in client.smembers(f"{self._ns}:tasks:all"))
        for tid in ids:
            owner = _decode(client.hget(f"{self._ns}:task:{tid}", "owner")) or ""
            if owner == self._agent_id:
                client.delete(f"{self._ns}:task:{tid}")
                client.srem(f"{self._ns}:tasks:all", tid)

        # 2. Re-add the agent's plan as owned tasks.
        for item in requested:
            tid = uuid.uuid4().hex[:10]
            # Strip any "[owner] " prefix the agent may have written back
            # from a prior view().
            title = item.title
            if title.startswith("[") and "] " in title:
                title = title.split("] ", 1)[1]
            cb_status = _OH_TO_CB_STATUS.get(item.status, "open")
            fields = {
                "id": tid,
                "title": title,
                "owner": self._agent_id,
                "status": cb_status,
                "created_by": self._agent_id,
                "created_at": str(time.time()),
                "last_note": item.notes or "",
                "metadata": "{}",
            }
            client.hset(f"{self._ns}:task:{tid}", mapping=fields)
            client.sadd(f"{self._ns}:tasks:all", tid)
            self._log(
                client,
                kind="create",
                task_id=tid,
                by=self._agent_id,
                title=title,
            )
            # Auto-claim — the agent is implicitly the owner of tasks it
            # plans for itself.
            self._log(client, kind="claim", task_id=tid, by=self._agent_id)
            if cb_status != "open":
                self._log(
                    client,
                    kind="update",
                    task_id=tid,
                    by=self._agent_id,
                    status=cb_status,
                )

    def __call__(
        self,
        action: TaskTrackerAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> TaskTrackerObservation:
        try:
            client = _redis_client_from_env()
        except Exception as e:  # pragma: no cover - infra failure
            logger.warning("CoopTaskTracker: redis unavailable (%s); returning empty", e)
            return TaskTrackerObservation.from_text(
                text=(f"Shared task list unavailable: {e}"),
                command=action.command,
                task_list=[],
                is_error=True,
            )

        if action.command == "plan":
            self._write_plan(client, action.task_list)
            items = self._read_all(client)
            return TaskTrackerObservation.from_text(
                text=(
                    f"Updated your tasks in the shared list. "
                    f"Total tasks across the team: {len(items)}."
                ),
                command=action.command,
                task_list=items,
            )
        if action.command == "view":
            items = self._read_all(client)
            if not items:
                return TaskTrackerObservation.from_text(
                    text=('Shared task list is empty. Use "plan" to add tasks.'),
                    command=action.command,
                    task_list=[],
                )
            return TaskTrackerObservation.from_text(
                text=self._format_team_view(items),
                command=action.command,
                task_list=items,
            )
        return TaskTrackerObservation.from_text(
            text=(
                f"Unknown command: {action.command}. "
                'Supported commands are "view" and "plan".'
            ),
            is_error=True,
            command=action.command,
            task_list=[],
        )

    def _format_team_view(self, items: list[TaskItem]) -> str:
        out = ["# Shared team task list", ""]
        for i, item in enumerate(items, 1):
            icon = {"todo": "⏳", "in_progress": "🔄", "done": "✅"}.get(item.status, "⏳")
            out.append(f"{i}. {icon} {item.title}")
            if item.notes:
                out.append(f"   {item.notes}")
        return "\n".join(out)


COOP_TASK_TRACKER_DESCRIPTION = """Shared, team-visible task tracker (replaces the local TaskTracker in team mode).

Same shape as the local TaskTracker — ``plan`` to write your tasks,
``view`` to read the current state — but persistence is a Redis-backed
list that EVERY TEAMMATE reads from and writes to.  Use this to
coordinate work allocation with peers:

- ``view`` shows every team member's tasks (yours and your peers').
  Peer tasks are prefixed with ``[<their_agent_id>]``.
- ``plan`` replaces YOUR OWN tasks in the shared list (other agents'
  tasks are untouched).  Each task you list is automatically owned
  by you.

Workflow:

1. Start each session with ``view`` to see what your peers have
   already planned.  If they're touching files you also need, message
   them via SendMessage to coordinate.
2. ``plan`` your own work as a list of TaskItems with status="todo".
3. When you start a task, re-run ``plan`` with that task at
   status="in_progress".  When done, status="done".
4. Re-run ``view`` periodically to see your peers' progress.

This is the ONLY task-list mechanism in team mode.  Do NOT also call
the shell commands ``coop-task-create`` / ``coop-task-claim`` —
they're the same Redis backend, calling both creates duplicates.
"""


class CoopTaskTrackerTool(ToolDefinition[TaskTrackerAction, TaskTrackerObservation]):
    """Drop-in replacement for ``TaskTrackerTool`` that uses shared Redis."""

    @classmethod
    def create(
        cls, conv_state: "ConversationState"  # noqa: ARG003
    ) -> Sequence["CoopTaskTrackerTool"]:
        return [
            cls(
                description=COOP_TASK_TRACKER_DESCRIPTION,
                action_type=TaskTrackerAction,
                observation_type=TaskTrackerObservation,
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=CoopTaskTrackerExecutor(),
            )
        ]


# Register under BOTH names:
#   - "CoopTaskTrackerTool" so callers can request it explicitly
#   - "TaskTrackerTool" (the same name as the local one) so importing
#     this module overrides the existing registration.  The override
#     is what makes the Modal-side swap work — host sends
#     ``Tool(name="TaskTrackerTool")`` and the agent-server's registry
#     resolves to our Redis-backed variant after this module is
#     imported via the .pth file the adapter installs.
from openhands.tools.task_tracker.definition import TaskTrackerTool as _LocalTaskTrackerTool

register_tool("CoopTaskTrackerTool", CoopTaskTrackerTool)
register_tool(_LocalTaskTrackerTool.name, CoopTaskTrackerTool)
