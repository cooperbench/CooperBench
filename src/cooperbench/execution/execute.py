"""
Unified execution entrypoint for CooperBench.

Handles execution for all settings: single, solo, coop, coop_ablation.
"""

import argparse
import asyncio
from typing import Literal

from dotenv import load_dotenv

from cooperbench import BenchSetting, FileInterface
from cooperbench.core.patch import generate_patch
from cooperbench.execution.openhands import run_execution as run_openhands_execution
from cooperbench.execution.openhands_coop import run_coop_execution
from cooperbench.execution.prompt import render_task_prompt

load_dotenv()


async def create_execution(
    file_interface: FileInterface,
    plan_location: Literal["logs", "cache", "hf"] = "cache",
) -> None:
    """Main execution function.

    Args:
        file_interface: FileInterface object for file management
        plan_location: Where to load the plan from ("logs", "cache", or "hf")
    """
    setting = file_interface.setting

    # Coop mode runs two agents in parallel
    if setting == BenchSetting.COOP:
        await _run_coop_execution(file_interface, plan_location)
        return

    # Coop ablation runs two agents independently (no runtime communication)
    if setting == BenchSetting.COOP_ABLATION:
        await _run_coop_ablation_execution(file_interface, plan_location)
        return

    # Single and solo modes use standard single-agent execution
    file_interface.setup_filesystem(setup_both_agent_workspaces=False)

    # For solo mode, concatenate both feature descriptions
    if setting == BenchSetting.SOLO:
        feature_desc = (
            file_interface.get_feature_description(first=True)
            + "\n\n"
            + file_interface.get_feature_description(first=False)
        )
    else:
        feature_desc = file_interface.get_feature_description()

    prompt = render_task_prompt(feature_desc, file_interface.get_plan(plan_location))

    success = await run_openhands_execution(
        agent_workspace_path=file_interface.agent_workspace1_path,
        prompt=prompt,
        model=file_interface.model1,
        local_trajectory_path=file_interface.get_log_file_path("execution_traj1"),
        container_name=file_interface.get_container_name(),
    )

    if success:
        diff_content = generate_patch(
            agent_workspace_path=file_interface.agent_workspace1_path,
            base_commit=file_interface.base_commit,
        )
        if diff_content and diff_content.strip():
            file_interface.save_patch(diff_content)
        file_interface.save_execution_trajectory()
        print("SUCCESS: Execution completed successfully.")
    else:
        print("FAILURE: Execution failed.")


async def _run_coop_execution(
    file_interface: FileInterface,
    plan_location: Literal["logs", "cache", "hf"] = "cache",
) -> None:
    """Coop execution with two agents running in parallel with MCP communication.

    Args:
        file_interface: FileInterface object for file management
        plan_location: Where to load the plan from
    """
    file_interface.setup_filesystem(setup_both_agent_workspaces=True, check_for_conflicts=False)

    feature_desc_1 = file_interface.get_feature_description(first=True)
    feature_desc_2 = file_interface.get_feature_description(first=False)

    plan_1 = file_interface.get_plan(plan_location, first=True)
    plan_2 = file_interface.get_plan(plan_location, first=False)

    # Run parallel agents with MCP communication
    (
        success,
        traj1_path,
        traj2_path,
        conversation_json_path,
        agent_log_paths,
    ) = await run_coop_execution(
        workspace_1=file_interface.agent_workspace1_path,
        workspace_2=file_interface.agent_workspace2_path,
        feature_desc_1=feature_desc_1,
        feature_desc_2=feature_desc_2,
        plan_1=plan_1,
        plan_2=plan_2,
        model1=file_interface.model1,
        model2=file_interface.model2 or file_interface.model1,
        feature1_id=file_interface.feature1_id,
        feature2_id=file_interface.feature2_id,
        repo_name=file_interface.repo_name,
        task_id=file_interface.task_id,
        k=file_interface.k,
    )

    if success:
        # Generate patch for agent 1
        diff_content_1 = generate_patch(
            agent_workspace_path=file_interface.agent_workspace1_path,
            base_commit=file_interface.base_commit,
        )
        if diff_content_1 and diff_content_1.strip():
            file_interface.save_patch(diff_content_1, first=True)

        # Generate patch for agent 2
        diff_content_2 = generate_patch(
            agent_workspace_path=file_interface.agent_workspace2_path,
            base_commit=file_interface.base_commit,
        )
        if diff_content_2 and diff_content_2.strip():
            file_interface.save_patch(diff_content_2, first=False)

        # Save coop execution files (trajectories, conversation)
        file_interface.save_coop_execution_files(
            execution_traj1_path=traj1_path,
            execution_traj2_path=traj2_path,
            conversation_json_path=conversation_json_path,
            agent_log_paths=agent_log_paths,
        )

        print("SUCCESS: Coop execution completed successfully.")
    else:
        print("FAILURE: Coop execution failed.")


async def _run_coop_ablation_execution(
    file_interface: FileInterface,
    plan_location: Literal["logs", "cache", "hf"] = "cache",
) -> None:
    """Coop ablation execution - same as coop but uses plans from coop setting.

    The planning phase is shared with coop, so we load plans from coop directory.

    Args:
        file_interface: FileInterface object for file management
        plan_location: Where to load the plan from
    """
    file_interface.setup_filesystem(setup_both_agent_workspaces=True, check_for_conflicts=False)

    feature_desc_1 = file_interface.get_feature_description(first=True)
    feature_desc_2 = file_interface.get_feature_description(first=False)

    # Load plans - these come from the coop planning phase (same planning logic)
    plan_1 = file_interface.get_plan(plan_location, first=True)
    plan_2 = file_interface.get_plan(plan_location, first=False)

    prompt_1 = render_task_prompt(feature_desc_1, plan_1)
    prompt_2 = render_task_prompt(feature_desc_2, plan_2)

    # Run both agents in parallel (no communication during execution)
    results = await asyncio.gather(
        run_openhands_execution(
            agent_workspace_path=file_interface.agent_workspace1_path,
            prompt=prompt_1,
            model=file_interface.model1,
            local_trajectory_path=file_interface.get_log_file_path("execution_traj1"),
            container_name=file_interface.get_container_name() + "_agent1",
        ),
        run_openhands_execution(
            agent_workspace_path=file_interface.agent_workspace2_path,
            prompt=prompt_2,
            model=file_interface.model2 or file_interface.model1,
            local_trajectory_path=file_interface.get_log_file_path("execution_traj2"),
            container_name=file_interface.get_container_name() + "_agent2",
        ),
        return_exceptions=True,
    )

    success1 = results[0] if not isinstance(results[0], Exception) else False
    success2 = results[1] if not isinstance(results[1], Exception) else False

    if success1:
        diff_content_1 = generate_patch(
            agent_workspace_path=file_interface.agent_workspace1_path,
            base_commit=file_interface.base_commit,
        )
        if diff_content_1 and diff_content_1.strip():
            file_interface.save_patch(diff_content_1, first=True)

    if success2:
        diff_content_2 = generate_patch(
            agent_workspace_path=file_interface.agent_workspace2_path,
            base_commit=file_interface.base_commit,
        )
        if diff_content_2 and diff_content_2.strip():
            file_interface.save_patch(diff_content_2, first=False)

    if success1 and success2:
        print("SUCCESS: Coop ablation execution completed successfully.")
    else:
        print(f"PARTIAL: Agent1={'OK' if success1 else 'FAIL'}, Agent2={'OK' if success2 else 'FAIL'}")


async def main() -> None:
    """CLI wrapper for create_execution()."""
    parser = argparse.ArgumentParser(description="CooperBench unified execution entrypoint")

    parser.add_argument(
        "--setting",
        "-s",
        required=True,
        choices=[s.value for s in BenchSetting],
        help="Execution setting mode",
    )
    parser.add_argument("--repo-name", type=str, required=True, help="Repository name")
    parser.add_argument("--task-id", type=int, required=True, help="Task number")
    parser.add_argument("--model1", "-m1", required=True, help="Model for first agent")
    parser.add_argument("--model2", "-m2", help="Model for second agent")
    parser.add_argument("--feature1-id", "-i", type=int, required=True, help="First feature ID")
    parser.add_argument("--feature2-id", "-j", type=int, help="Second feature ID")
    parser.add_argument("--save-to-hf", action="store_true", help="Save results to HuggingFace")
    parser.add_argument("--create-pr", action="store_true", help="Create PR when saving to HF")
    parser.add_argument("--k", type=int, default=1, help="Experiment run identifier")
    parser.add_argument(
        "--plan-location",
        choices=["logs", "cache", "hf"],
        default="cache",
        help="Where to load plans from",
    )

    args = parser.parse_args()

    setting = BenchSetting(args.setting)

    file_interface = FileInterface(
        setting=setting,
        repo_name=args.repo_name,
        task_id=args.task_id,
        k=args.k,
        feature1_id=args.feature1_id,
        model1=args.model1,
        feature2_id=args.feature2_id,
        model2=args.model2,
        save_to_hf=args.save_to_hf,
        create_pr=args.create_pr,
    )

    await create_execution(file_interface, args.plan_location)


if __name__ == "__main__":
    asyncio.run(main())
