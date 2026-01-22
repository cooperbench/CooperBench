"""
Single-agent planning mode for CooperBench.

This module implements the single-agent planning strategy where one agent
creates an implementation plan for a single feature.
"""

from pathlib import Path

import jinja2

from cooperbench.core.interface import FileInterface
from cooperbench.planning.agent import BaseAgent
from cooperbench.planning.tools import (
    AgreementTool,
    BaseTool,
    GrepSearchTool,
    ListFilesTool,
    ReadFileTool,
)
from cooperbench.planning.trajectory import TrajectoryLogger


class SingleAgent(BaseAgent):
    """Agent for single-agent planning with standard exploration and agreement tools."""

    def __init__(
        self,
        agent_workspace_path: Path,
        model: str,
        max_iterations: int,
        system_prompt_file: str,
        feature_description: str,
        trajectory_logger: TrajectoryLogger | None = None,
        agent_id: str = "single",
    ) -> None:
        super().__init__(
            agent_workspace_path,
            model,
            max_iterations,
            system_prompt_file,
            trajectory_logger,
            agent_id,
        )
        self.feature_description = feature_description

    def get_tools(self) -> list[BaseTool]:
        """Return tools needed for single agent planning."""
        return [
            ListFilesTool(),
            ReadFileTool(),
            GrepSearchTool(),
            AgreementTool(),
        ]

    def get_system_prompt(self) -> str:
        """Get system prompt with feature description injected."""
        template_dir = Path(self.system_prompt_file).parent
        template_name = Path(self.system_prompt_file).name

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template(template_name)
        return template.render(
            agent_workspace_name=self.agent_workspace_path.name,
            feature_description=self.feature_description,
        )


async def run_planning(file_interface: FileInterface, max_iterations: int = 50) -> tuple[str, dict]:
    """Run single-agent planning process.

    Args:
        file_interface: FileInterface object for managing file paths
        max_iterations: Maximum number of iterations

    Returns:
        tuple: (plan, trajectory)
    """
    feature_description = file_interface.get_feature_description()
    agent_workspace_path = file_interface.agent_workspace1_path
    model = file_interface.model1

    system_prompt_path = Path(__file__).parent / "templates" / "single.j2"

    trajectory_logger = TrajectoryLogger("single", agent_workspace_path)
    trajectory_logger.register_agent("single", model, "single")

    print(f"\n[PLANNING] Starting single planning")
    print(f"  Workspace: {agent_workspace_path}")
    print(f"  Model: {model}")

    agent = SingleAgent(
        agent_workspace_path=agent_workspace_path,
        model=model,
        max_iterations=max_iterations,
        system_prompt_file=str(system_prompt_path),
        feature_description=feature_description,
        trajectory_logger=trajectory_logger,
        agent_id="single",
    )

    plan = await agent.run("Begin creating your implementation plan.")

    if not plan or plan.strip() == "":
        print("No plan generated. Forcing agreement...")

        original_tools = agent.tools
        agent.tools = [AgreementTool()]

        agent.messages.append(
            {
                "role": "user",
                "content": "You must now submit your implementation plan using agreement_reached.",
            }
        )

        _, plan = await agent.run_iteration()

        agent.tools = original_tools

        if not plan:
            plan = feature_description

    print("\n[PLANNING] Single planning complete")

    trajectory_logger.log_final_plans(plan=plan)
    trajectory = trajectory_logger.output()

    return plan, trajectory
