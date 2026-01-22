"""
Base agent implementation for planning phase.

This module provides the abstract base class for planning agents, handling
LLM interactions, tool execution, and iteration management.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import jinja2

from cooperbench.llm import call_llm
from cooperbench.planning.tools import BaseTool, ToolResult
from cooperbench.planning.trajectory import TrajectoryLogger


def _print_llm_output(agent_id: str, content: str) -> None:
    """Print LLM response content."""
    if content:
        print(f"\n[{agent_id}] LLM Response:\n{content[:500]}{'...' if len(content) > 500 else ''}")


def _print_tool_call(agent_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
    """Print tool call details."""
    args_preview = {k: v[:100] if isinstance(v, str) and len(v) > 100 else v for k, v in arguments.items()}
    args_preview.pop("agent_workspace_path", None)
    print(f"\n[{agent_id}] Tool Call: {tool_name}")
    print(f"  Args: {args_preview}")


def _print_tool_result(agent_id: str, success: bool, content: str, error: str) -> None:
    """Print tool execution result."""
    status = "SUCCESS" if success else "FAILED"
    output = content if success else error
    print(f"[{agent_id}] Tool Result ({status}): {output[:200]}{'...' if len(output) > 200 else ''}")


class BaseAgent(ABC):
    """Abstract base class for planning agents."""

    def __init__(
        self,
        agent_workspace_path: Path,
        model: str,
        max_iterations: int,
        system_prompt_file: str,
        trajectory_logger: TrajectoryLogger | None = None,
        agent_id: str = "agent",
    ) -> None:
        self.agent_workspace_path = agent_workspace_path.resolve()
        self.model = model
        self.max_iterations = max_iterations
        self.system_prompt_file = system_prompt_file
        self.current_iteration = 0
        self.trajectory_logger = trajectory_logger
        self.agent_id = agent_id

        self.tools = self.get_tools()
        self.messages: list[dict[str, Any]] = []

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Return the list of tools available to this agent."""
        pass

    def get_system_prompt(self) -> str:
        """Load and render the system prompt template."""
        template_dir = Path(self.system_prompt_file).parent
        template_name = Path(self.system_prompt_file).name

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)

        template = env.get_template(template_name)
        agent_workspace_name = self.agent_workspace_path.name
        return template.render(agent_workspace_name=agent_workspace_name)

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            result = ToolResult(success=False, error=f"Tool {tool_name} not found")
        else:
            arguments["agent_workspace_path"] = self.agent_workspace_path
            try:
                result = await tool.execute(**arguments)
            except TypeError as e:
                error_msg = str(e)
                if "missing" in error_msg and "required positional argument" in error_msg:
                    missing_param = error_msg.split("'")[-2] if "'" in error_msg else "unknown parameter"
                    result = ToolResult(
                        success=False,
                        error=f"Tool call error: Missing required parameter '{missing_param}'.",
                    )
                elif "unexpected keyword argument" in error_msg:
                    unexpected_param = error_msg.split("'")[-2] if "'" in error_msg else "unknown parameter"
                    result = ToolResult(
                        success=False,
                        error=f"Tool call error: Unexpected parameter '{unexpected_param}'.",
                    )
                else:
                    result = ToolResult(
                        success=False,
                        error=f"Tool call error: {error_msg}.",
                    )
            except Exception as e:
                result = ToolResult(success=False, error=f"Tool execution failed: {str(e)}")

        _print_tool_call(self.agent_id, tool_name, arguments)
        _print_tool_result(self.agent_id, result.success, result.content, result.error)

        if self.trajectory_logger:
            self.trajectory_logger.log_tool_execution(
                agent_id=self.agent_id,
                tool_name=tool_name,
                arguments=arguments,
                result_success=result.success,
                result_content=result.content,
                result_error=result.error,
            )

        return result

    async def run_iteration(self) -> tuple[bool, str]:
        """Run a single planning iteration.
        
        Returns:
            Tuple of (should_continue, plan_if_complete)
        """
        self.current_iteration += 1

        response = await call_llm(
            messages=self.messages,
            tools=[tool.get_schema() for tool in self.tools],
            model=self.model,
            return_full_response=True,
        )

        tool_calls, content = response["tool_calls"], response["content"]

        _print_llm_output(self.agent_id, content)

        if self.trajectory_logger:
            self.trajectory_logger.log_llm_call(
                agent_id=self.agent_id,
                messages=self.messages,
                tools=[tool.get_schema() for tool in self.tools],
                response_tool_calls=tool_calls,
                response_content=content,
            )

        self.messages.append(response["message"])

        if not tool_calls:
            return self.current_iteration < self.max_iterations, ""

        plan_result = None
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]

            result = await self.execute_tool(tool_name, arguments)

            if tool_name in ["agreement_reached", "dual_agreement_reached"] and result.success:
                plan_result = result.content

            tool_response = {
                "role": "tool",
                "content": result.content if result.success else f"Error: {result.error}",
                "tool_call_id": tool_call["id"],
            }
            self.messages.append(tool_response)

        if plan_result:
            return False, plan_result

        return self.current_iteration < self.max_iterations, ""

    async def run(self, task_description: str) -> str:
        """Run the planning process.
        
        Args:
            task_description: The initial task description
            
        Returns:
            The final plan string
        """
        self.messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": task_description},
        ]

        try:
            while True:
                should_continue, plan = await self.run_iteration()
                if plan:
                    return plan
                if not should_continue:
                    break

            return ""

        except Exception:
            raise
