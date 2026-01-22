"""
Trajectory logging for planning sessions.

This module captures the complete trajectory of agent planning sessions,
including LLM calls, tool executions, and coordination events for
multi-agent scenarios.
"""

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """Logger for capturing complete trajectory of agent planning sessions."""

    def __init__(self, planning_mode: str, agent_workspace_path: Path) -> None:
        """Initialize trajectory logger.

        Args:
            planning_mode: Type of planning (single, solo, coop, coop_ablation)
            agent_workspace_path: Path to agent workspace directory
        """
        self.planning_mode = planning_mode
        self.agent_workspace_path = agent_workspace_path
        self.agents: list[dict[str, str]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.final_plans: dict[str, str] = {}
        self.start_time = datetime.now(UTC)
        self.end_time: datetime | None = None
        self._lock = threading.Lock()

    def register_agent(self, agent_id: str, model: str, role: str) -> None:
        """Register an agent in the planning session.

        Args:
            agent_id: Unique identifier for the agent
            model: LLM model being used
            role: Role of the agent (single, feature1, feature2)
        """
        with self._lock:
            self.agents.append({"id": agent_id, "model": model, "role": role})

    def log_llm_call(
        self,
        agent_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_tool_calls: list[dict[str, Any]] | None,
        response_content: str,
    ) -> None:
        """Log an LLM call and response.

        Args:
            agent_id: ID of the agent making the call
            messages: Input messages to the LLM
            tools: Available tools for the LLM
            response_tool_calls: Tool calls in the response (if any)
            response_content: Text content of the response
        """
        with self._lock:
            self.trajectory.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_id": agent_id,
                    "type": "llm_call",
                    "data": {
                        "input": {
                            "messages": messages,
                            "tools": [{"name": tool.get("function", {}).get("name", "unknown")} for tool in tools],
                        },
                        "output": {
                            "content": response_content,
                            "tool_calls": response_tool_calls or [],
                        },
                    },
                }
            )

    def log_tool_execution(
        self,
        agent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_success: bool,
        result_content: object,
        result_error: str = "",
    ) -> None:
        """Log a tool execution and its result.

        Args:
            agent_id: ID of the agent executing the tool
            tool_name: Name of the tool being executed
            arguments: Arguments passed to the tool
            result_success: Whether the tool execution succeeded
            result_content: Content returned by the tool
            result_error: Error message if execution failed
        """
        arguments.pop("agent_workspace_path", None)
        with self._lock:
            self.trajectory.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_id": agent_id,
                    "type": "tool_execution",
                    "data": {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": {
                            "success": result_success,
                            "content": result_content,
                            "error": result_error,
                        },
                    },
                }
            )

    def log_coordination(self, action: str, details: dict[str, Any]) -> None:
        """Log coordination events (for multi-agent modes).

        Args:
            action: Type of coordination action
            details: Additional details about the coordination event
        """
        with self._lock:
            self.trajectory.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "agent_id": "coordinator",
                    "type": "coordination",
                    "data": {"action": action, **details},
                }
            )

    def log_final_plans(self, **plans: str) -> None:
        """Log the final implementation plans.

        Args:
            **plans: Plans as keyword arguments
                     - For single/solo: plan=plan_text
                     - For coop: plan1=plan1_text, plan2=plan2_text
        """
        with self._lock:
            self.final_plans = plans
            self.end_time = datetime.now(UTC)

    def output(self) -> dict[str, Any]:
        """Return the complete trajectory as a dictionary.

        Returns:
            dict: The trajectory data including metadata, trajectory events, and final plans
        """
        with self._lock:
            return {
                "metadata": {
                    "planning_mode": self.planning_mode,
                    "agents": self.agents,
                    "agent_workspace_path": str(self.agent_workspace_path),
                    "start_time": self.start_time.isoformat(),
                    "end_time": self.end_time.isoformat() if self.end_time else None,
                },
                "trajectory": self.trajectory,
                "final_plans": self.final_plans,
            }
