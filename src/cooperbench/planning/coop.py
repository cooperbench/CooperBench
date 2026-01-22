"""
Cooperative multi-agent planning mode for CooperBench.

This module implements the coop planning strategy where two agents
coordinate to create implementation plans for separate features while
communicating to prevent merge conflicts.
"""

from pathlib import Path

import jinja2

from cooperbench.core.interface import FileInterface
from cooperbench.planning.agent import BaseAgent
from cooperbench.planning.tools import (
    AgreementTool,
    BaseTool,
    CommunicateTool,
    GrepSearchTool,
    ListFilesTool,
    ReadFileTool,
)
from cooperbench.planning.trajectory import TrajectoryLogger


class CoopAgent(BaseAgent):
    """Agent for multi-agent coordination planning with communication capabilities."""

    def __init__(
        self,
        agent_workspace_path: Path,
        model: str,
        max_iterations: int,
        system_prompt_file: str,
        feature_description: str,
        trajectory_logger: TrajectoryLogger | None = None,
        agent_id: str = "agent",
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
        """Return tools needed for coop planning including communication."""
        return [
            ListFilesTool(),
            ReadFileTool(),
            GrepSearchTool(),
            CommunicateTool(),
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


class CoopCoordinator:
    """Coordinates communication between two agents in coop planning."""

    def __init__(
        self,
        agent_workspace_path: Path,
        feature1_description: str,
        feature2_description: str,
        model1: str,
        model2: str,
        max_iterations: int = 50,
        trajectory_logger: TrajectoryLogger | None = None,
    ) -> None:
        self.agent_workspace_path = agent_workspace_path
        self.feature1_description = feature1_description
        self.feature2_description = feature2_description
        self.model1 = model1
        self.model2 = model2
        self.max_iterations = max_iterations
        self.conversation_history: list[dict[str, int | str]] = []
        self.current_agent = 1
        self.trajectory_logger = trajectory_logger

        system_prompt_path = Path(__file__).parent / "templates" / "coop.j2"

        self.agent1 = CoopAgent(
            agent_workspace_path=agent_workspace_path,
            model=model1,
            max_iterations=max_iterations,
            system_prompt_file=str(system_prompt_path),
            feature_description=feature1_description,
            trajectory_logger=trajectory_logger,
            agent_id="agent1",
        )

        self.agent2 = CoopAgent(
            agent_workspace_path=agent_workspace_path,
            model=model2,
            max_iterations=max_iterations,
            system_prompt_file=str(system_prompt_path),
            feature_description=feature2_description,
            trajectory_logger=trajectory_logger,
            agent_id="agent2",
        )

    def get_current_agent(self) -> CoopAgent:
        """Get the agent whose turn it is."""
        return self.agent1 if self.current_agent == 1 else self.agent2

    def switch_agent(self) -> None:
        """Switch to the other agent."""
        old_agent = self.current_agent
        self.current_agent = 2 if self.current_agent == 1 else 1

        switch_details = {
            "from": f"agent{old_agent}",
            "to": f"agent{self.current_agent}",
        }
        print(f"[COORDINATION] Agent switch: {switch_details}")

        if self.trajectory_logger:
            self.trajectory_logger.log_coordination("agent_switch", switch_details)

    def add_to_conversation(self, message: str, agent_id: int) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({"agent": agent_id, "message": message})

        comm_details = {"from": f"agent{agent_id}", "message": message}
        print(f"[COORDINATION] Communication received: {comm_details}")

        if self.trajectory_logger:
            self.trajectory_logger.log_coordination("communication_received", comm_details)

    def get_conversation_context(self) -> str:
        """Get the full conversation history formatted for the agent."""
        if not self.conversation_history:
            return ""

        context = "\n\n## CONVERSATION HISTORY\n"
        for entry in self.conversation_history:
            agent_name = f"Agent {entry['agent']}"
            context += f"\n**{agent_name}**: {entry['message']}\n"

        return context

    async def run_agent_turn(self) -> tuple[bool, str | None, str | None]:
        """Run one turn for the current agent until they communicate or reach agreement.

        Returns:
            Tuple of (should_continue, agreement_plan, communication_message)
        """
        current_agent = self.get_current_agent()
        agent_id = self.current_agent

        conversation_context = self.get_conversation_context()
        task_with_context = f"Begin creating your implementation plan.{conversation_context}"

        if not current_agent.messages:
            current_agent.messages = [
                {
                    "role": "system",
                    "content": current_agent.get_system_prompt(),
                },
                {"role": "user", "content": task_with_context},
            ]
            current_agent.current_iteration = 0
        else:
            current_agent.messages.append({"role": "user", "content": conversation_context})

        while current_agent.current_iteration < current_agent.max_iterations:
            should_continue, plan = await current_agent.run_iteration()

            if current_agent.messages:
                last_message = current_agent.messages[-2]
                if last_message.get("tool_calls"):
                    for tool_call in last_message["tool_calls"]:
                        tool_name = tool_call["function"]["name"]

                        if tool_name == "agreement_reached":
                            return False, plan, None

                        if tool_name == "communicate_with_agent":
                            tool_response = current_agent.messages[-1]
                            communication = tool_response["content"]

                            self.add_to_conversation(communication, agent_id)
                            self.switch_agent()

                            return True, None, communication

            if not should_continue:
                break

        return False, None, None

    async def coordinate_planning(self) -> tuple[str, str]:
        """Coordinate the planning process between two agents.

        Returns:
            Tuple of (plan1, plan2) - implementation plans for each feature
        """
        agent_plans: dict[int, str] = {}

        print(f"\n[PLANNING] Starting coop planning")
        print(f"  Workspace: {self.agent_workspace_path}")
        print(f"  Models: {self.model1}, {self.model2}")

        iteration = 0
        while iteration < self.max_iterations:
            should_continue, plan, _ = await self.run_agent_turn()

            if plan:
                agent_id = self.current_agent
                agent_plans[agent_id] = plan

                if len(agent_plans) == 2:
                    print("\n[PLANNING] Coop planning complete")
                    return agent_plans[1], agent_plans[2]

                self.add_to_conversation(
                    "I agree with this coordination plan. Please use agreement_reached and propose your implementation plan.",
                    agent_id,
                )

                self.switch_agent()

            if not should_continue and len(agent_plans) == 0:
                break

            iteration += 1

        print("Max iterations reached. Forcing agreement...")

        for agent_id in [1, 2]:
            if agent_id not in agent_plans:
                current_agent = self.agent1 if agent_id == 1 else self.agent2
                current_agent.tools = [AgreementTool()]
                current_agent.messages.append(
                    {
                        "role": "user",
                        "content": "You must now submit your implementation plan using agreement_reached.",
                    }
                )
                _, plan = await current_agent.run_iteration()
                agent_plans[agent_id] = plan or f"# FORCED PLAN\nFeature: {current_agent.feature_description}"

        print("\n[PLANNING] Coop planning complete")
        return agent_plans[1], agent_plans[2]


async def run_planning(
    file_interface: FileInterface,
    max_iterations: int = 50,
) -> tuple[str, str, dict]:
    """Run coop (multi-agent coordination) planning process.

    Args:
        file_interface: FileInterface object for managing file paths
        max_iterations: Maximum number of iterations

    Returns:
        Tuple of (plan1, plan2, trajectory): Implementation plans for feature1 and feature2
    """
    feature1_description = file_interface.get_feature_description(first=True)
    feature2_description = file_interface.get_feature_description(first=False)
    agent_workspace_path = file_interface.agent_workspace1_path

    trajectory_logger = TrajectoryLogger("coop", agent_workspace_path)
    trajectory_logger.register_agent("agent1", file_interface.model1, "feature1")
    assert file_interface.model2 is not None, "model2 must be set for coop planning"
    trajectory_logger.register_agent("agent2", file_interface.model2, "feature2")

    coordinator = CoopCoordinator(
        agent_workspace_path=agent_workspace_path,
        feature1_description=feature1_description,
        feature2_description=feature2_description,
        model1=file_interface.model1,
        model2=file_interface.model2,
        max_iterations=max_iterations,
        trajectory_logger=trajectory_logger,
    )

    plan1, plan2 = await coordinator.coordinate_planning()

    trajectory_logger.log_final_plans(plan1=plan1, plan2=plan2)
    trajectory = trajectory_logger.output()

    return plan1, plan2, trajectory
