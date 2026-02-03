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
) -> list[Tool]:
    """Get the default set of tool specifications for the standard experience.

    Args:
        enable_browser: Whether to include browser tools.
    """
    register_default_tools(enable_browser=enable_browser)

    # Import tools to access their name attributes
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool
    from openhands.tools.collaboration import ReceiveMessageTool, SendMessageTool

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
    """Generate the collaboration section for the system prompt.
    
    Mirrors the mini-swe-agent prompt format for consistency.
    """
    all_agents = [agent_id] + teammates
    agents_str = ", ".join(all_agents)
    
    # Opening section (mirrors mini-swe-agent format)
    collab_section = f"""You are {agent_id} working as a team with: {agents_str}.
You are all working on related features in the same codebase. Each agent has their own workspace.
"""
    if git_enabled:
        collab_section += """A shared git remote called "team" is available for code sharing between agents.
"""
    if messaging_enabled:
        collab_section += """Use send_message to coordinate.
"""

    # Collaboration block (mirrors mini-swe-agent format)
    collab_section += """
<collaboration>
Each agent has their own workspace. At the end, all agents' changes will be merged together.
**Important**: Coordinate to avoid merge conflicts - your patches must cleanly combine!
"""
    if git_enabled:
        teammate_name = teammates[0] if teammates else "agent_0"
        collab_section += f"""
## Git
A shared remote called "team" is configured. Your branch is `{agent_id}`.
Teammates' branches are at `team/<name>` (e.g., `team/{teammate_name}`).

Example:
```
git push team {agent_id}
git fetch team
```
"""
    if messaging_enabled:
        collab_section += """
## Messaging (coordinate with teammates)
Send messages to teammates. Messages appear in their next turn.
```
send_message(recipient="<agent_name>", content="your message")
```
Messages from teammates appear as: [Message from <agent_name>]: ...
"""
    collab_section += "</collaboration>\n"
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
    tools = get_default_tools(
        # Disable browser tools in CLI mode
        enable_browser=not cli_mode,
    )
    
    # Build system prompt kwargs
    system_prompt_kwargs = {"cli_mode": cli_mode}
    
    # Add collaboration instructions to system prompt if in coop mode
    if coop_info and coop_info.get("agents") and len(coop_info["agents"]) > 1:
        agent_id = coop_info.get("agent_id", "agent")
        agents = coop_info["agents"]
        teammates = [a for a in agents if a != agent_id]
        messaging_enabled = coop_info.get("messaging_enabled", True)
        git_enabled = coop_info.get("git_enabled", False)
        
        collab_section = get_coop_system_prompt(agent_id, teammates, messaging_enabled, git_enabled)
        system_prompt_kwargs["collaboration"] = collab_section
    
    # Use custom template if in coop mode (template has {{ collaboration }} variable)
    system_prompt_filename = None
    if coop_info and coop_info.get("agents") and len(coop_info["agents"]) > 1:
        # Use the custom template written to /tmp in the sandbox
        system_prompt_filename = "/tmp/system_prompt_coop.j2"
    
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_filename=system_prompt_filename or "system_prompt.j2",
        system_prompt_kwargs=system_prompt_kwargs,
        condenser=get_default_condenser(
            llm=llm.model_copy(update={"usage_id": "condenser"})
        ),
    )
    return agent
