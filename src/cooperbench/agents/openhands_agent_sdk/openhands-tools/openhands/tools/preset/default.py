"""Default preset configuration for OpenHands agents."""

import os

from openhands.sdk import Agent
from openhands.sdk.context.condenser import (
    LLMSummarizingCondenser,
)
from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.llm.llm import LLM
from openhands.sdk.logger import get_logger
from openhands.sdk.tool import Tool

logger = get_logger(__name__)


def register_default_tools(enable_browser: bool = True) -> None:
    """Register the default set of tools."""
    # Tools are now automatically registered when imported
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    logger.debug(f"Tool: {TerminalTool.name} registered.")
    logger.debug(f"Tool: {FileEditorTool.name} registered.")
    logger.debug(f"Tool: {TaskTrackerTool.name} registered.")

    if enable_browser:
        from openhands.tools.browser_use import BrowserToolSet

        logger.debug(f"Tool: {BrowserToolSet.name} registered.")
    
    # Register collaboration tools (only active when REDIS_URL is set)
    from openhands.tools.collaboration import ReceiveMessageTool, SendMessageTool
    logger.debug(f"Tool: {SendMessageTool.name} registered.")
    logger.debug(f"Tool: {ReceiveMessageTool.name} registered.")


def get_default_tools(
    enable_browser: bool = True,
    team_mode: bool = False,  # noqa: ARG001 — kept for API stability + tests
) -> list[Tool]:
    """Get the default set of tool specifications for the standard experience.

    Args:
        enable_browser: Whether to include browser tools.
        team_mode: Accepted but currently unused.  Team-mode swap of the
            TaskTracker tool happens *server-side* via the .pth-driven
            import of ``coop_definition.py`` in the Modal sandbox, which
            re-registers ``TaskTrackerTool`` to point at the Redis-backed
            executor.  Keeping this kwarg for tests + future use.
    """
    register_default_tools(enable_browser=enable_browser)

    # Import tools to access their name attributes
    from openhands.tools.collaboration import ReceiveMessageTool, SendMessageTool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(name=SendMessageTool.name),  # Only active when REDIS_URL is set
        Tool(name=ReceiveMessageTool.name),  # Only active when REDIS_URL is set
    ]
    if enable_browser:
        from openhands.tools.browser_use import BrowserToolSet

        tools.append(Tool(name=BrowserToolSet.name))
    
    return tools


def get_default_condenser(llm: LLM) -> CondenserBase:
    # Create a condenser to manage the context. The condenser will automatically
    # truncate conversation history when it exceeds max_size, and replaces the dropped
    # events with an LLM-generated summary.
    condenser = LLMSummarizingCondenser(llm=llm, max_size=80, keep_first=4)

    return condenser


def get_coop_system_prompt(agent_id: str, teammates: list[str], messaging_enabled: bool, git_enabled: bool) -> str:
    """Generate the collaboration section for the system prompt."""
    all_agents = [agent_id] + teammates
    agents_str = ", ".join(all_agents)
    teammate_name = teammates[0] if teammates else "teammate"

    collab_section = f"""You are {agent_id} working as a team with: {agents_str}.

<collaboration>

## Scenario
Your teammate ({teammate_name}) is implementing the other feature in this same codebase right now. You are both editing files in parallel but you cannot see each other's changes. When you're done, your work will be merged — if any of your edits touch the same lines, both patches are thrown away.

## Required workflow

1. Before you write any code, explore the codebase to understand what files you need to change. Then send a message to {teammate_name} listing every file and function you plan to modify:
   send_message(recipient="{teammate_name}", content="I plan to modify: <list files and functions>")

2. Wait for a reply from {teammate_name}. Their messages will appear automatically in your conversation as [Message from {teammate_name}]: ... — check for overlaps and coordinate if needed.

3. While you are working, if you discover you need to change files you didn't mention, message {teammate_name} before editing them.

4. When you are done, send a final summary of every file you changed:
   send_message(recipient="{teammate_name}", content="Done. I changed: <list files and changes>")

Do not skip these steps. If your edits conflict with your teammate's, both of your patches will be discarded.
"""

    if git_enabled:
        collab_section += f"""
## Git
A shared remote called `team` is configured. Your branch is `{agent_id}`.
Teammates' branches are at `team/<name>` (e.g., `team/{teammate_name}`).
* Push: `git push team {agent_id}`
* Fetch: `git fetch team`
"""
    # Experiment knob: when CB_COOP_TRUST_EMPHASIS is set, add a strong
    # instruction to trust the teammate's stated promises (rather than
    # re-verifying or defensively duplicating their work).  Off by default
    # so the baseline coop prompt is unchanged.
    if os.environ.get("CB_COOP_TRUST_EMPHASIS"):
        collab_section += (
            "\n## Trust\n"
            "YOU SHOULD TRY TO TRUST YOUR TEAMMATE REGARDING WHAT THEY PROMISED. "
            f"When {teammate_name} tells you which files/functions they will handle or how an "
            "interface will behave, take them at their word: build against what they promised, "
            "do not re-implement or second-guess their part, and do not duplicate their changes. "
            "Coordinate through messages and rely on their commitments.\n"
        )

    collab_section += """
</collaboration>
"""
    return collab_section


def get_default_agent(
    llm: LLM,
    cli_mode: bool = False,
    coop_info: dict | None = None,
) -> Agent:
    """Get a configured default agent.
    
    Args:
        llm: The LLM to use
        cli_mode: Whether running in CLI mode (disables browser tools)
        coop_info: Optional collaboration info dict with keys:
            - agent_id: This agent's ID
            - agents: List of all agent IDs in the team
            - messaging_enabled: Whether messaging is available
            - git_enabled: Whether git sharing is available
    """
    # Team mode swaps the local TaskTracker for the Redis-backed one
    # (CoopTaskTrackerTool).  The tool definition is baked into the
    # Modal sandbox via add_local_file + a .pth file at image-build
    # time — see ``ModalSandboxContext.__enter__`` for the injection.
    team_mode = bool(coop_info and coop_info.get("team_env"))
    tools = get_default_tools(
        # Disable browser tools in CLI mode
        enable_browser=not cli_mode,
        team_mode=team_mode,
    )
    
    # Build system prompt kwargs
    system_prompt_kwargs = {"cli_mode": cli_mode}
    
    # Add collaboration instructions to system prompt if in coop mode
    # The default system_prompt.j2 now has {{ collaboration }} slot
    if coop_info and coop_info.get("agents") and len(coop_info["agents"]) > 1:
        agent_id = coop_info.get("agent_id", "agent")
        agents = coop_info["agents"]
        teammates = [a for a in agents if a != agent_id]
        messaging_enabled = coop_info.get("messaging_enabled", True)
        git_enabled = coop_info.get("git_enabled", False)

        collab_section = get_coop_system_prompt(agent_id, teammates, messaging_enabled, git_enabled)

        # Team mode: the host adapter pre-renders a ``team_section`` string
        # describing the coop-task-* CLI + lead/member role.  Append it to
        # the collaboration block so it ends up in the SYSTEM prompt
        # (where it competes on equal footing with the coop_section above)
        # rather than getting buried in the user message — that was the
        # ``oh_team_v2`` failure mode.  In team mode we ALSO swap the
        # local TaskTracker tool for a Redis-backed CoopTaskTracker
        # (see ``get_default_tools(team_mode=True)``), so when the LLM
        # reaches for its TaskTracker tool it's writing to the shared
        # list automatically — no shell calls required.  The section
        # documents the equivalent CLI for agents that prefer shell.
        team_section = coop_info.get("team_section")
        if team_section:
            collab_section += (
                "\n\n## Team-mode shared task list\n"
                "Your TaskTracker tool in this run is wired to a SHARED\n"
                "Redis-backed list that every teammate reads from and writes\n"
                "to.  Use TaskTracker normally — your `plan` calls update\n"
                "the shared list with you as the owner of those items,\n"
                "and `view` shows every team member's tasks (peers'\n"
                "items are prefixed with `[<their_agent_id>]`).\n\n"
                "The shell commands `coop-task-create` / `coop-task-claim`\n"
                "/ `coop-task-update` / `coop-task-list` are also\n"
                "available and hit the same backend; use whichever fits\n"
                "your current action better, but pick one per task to\n"
                "avoid duplicates.\n\n"
                + team_section
                + "\n"
            )

        system_prompt_kwargs["collaboration"] = collab_section
    
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_filename="system_prompt.j2",
        system_prompt_kwargs=system_prompt_kwargs,
        condenser=get_default_condenser(
            llm=llm.model_copy(update={"usage_id": "condenser"})
        ),
    )
    return agent
