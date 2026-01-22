"""
Solo planning mode for CooperBench.

This module implements the solo planning strategy where a single agent
creates a unified implementation plan for two features simultaneously.
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


class SoloAgent(BaseAgent):
    """Agent for solo planning - single agent plans both features together."""

    def __init__(
        self,
        agent_workspace_path: Path,
        model: str,
        max_iterations: int,
        system_prompt_file: str,
        feature1_description: str,
        feature2_description: str,
        trajectory_logger: TrajectoryLogger | None = None,
        agent_id: str = "solo",
    ) -> None:
        super().__init__(
            agent_workspace_path,
            model,
            max_iterations,
            system_prompt_file,
            trajectory_logger,
            agent_id,
        )
        self.feature1_description = feature1_description
        self.feature2_description = feature2_description

    def get_tools(self) -> list[BaseTool]:
        """Return tools needed for solo agent planning."""
        return [
            ListFilesTool(),
            ReadFileTool(),
            GrepSearchTool(),
            AgreementTool(),
        ]

    def get_system_prompt(self) -> str:
        """Get system prompt with both feature descriptions injected."""
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
            feature1_description=self.feature1_description,
            feature2_description=self.feature2_description,
        )


async def run_planning(file_interface: FileInterface, max_iterations: int = 50) -> tuple[str, dict]:
    """Run solo planning process (single agent plans both features).

    Args:
        file_interface: FileInterface object for managing file paths
        max_iterations: Maximum number of iterations

    Returns:
        tuple: (unified_plan, trajectory)
    """
    feature1_description = file_interface.get_feature_description(first=True)
    feature2_description = file_interface.get_feature_description(first=False)
    agent_workspace_path = file_interface.agent_workspace1_path
    model = file_interface.model1

    system_prompt_path = Path(__file__).parent / "templates" / "solo.j2"

    trajectory_logger = TrajectoryLogger("solo", agent_workspace_path)
    trajectory_logger.register_agent("solo", model, "both_features")

    print(f"\n[PLANNING] Starting solo planning")
    print(f"  Workspace: {agent_workspace_path}")
    print(f"  Model: {model}")

    agent = SoloAgent(
        agent_workspace_path=agent_workspace_path,
        model=model,
        max_iterations=max_iterations,
        system_prompt_file=str(system_prompt_path),
        feature1_description=feature1_description,
        feature2_description=feature2_description,
        trajectory_logger=trajectory_logger,
        agent_id="solo",
    )

    plan = await agent.run("Begin creating your implementation plan.")

    if not plan or plan.strip() == "":
        print("No plan generated. Forcing agreement...")
        agent.tools = [AgreementTool()]
        agent.messages.append(
            {
                "role": "user",
                "content": "You must now submit your unified implementation plan using agreement_reached.",
            }
        )
        _, plan = await agent.run_iteration()

        if not plan:
            plan = f"{feature1_description}\n\n{feature2_description}"

    print("\n[PLANNING] Solo planning complete")

    trajectory_logger.log_final_plans(plan=plan)
    trajectory = trajectory_logger.output()

    return plan, trajectory
