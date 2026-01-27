"""
Unified planning entrypoint for CooperBench.

Supports multiple planning modes:
- single: Single agent planning for one feature
- solo: Single agent planning for two features simultaneously
- coop: Two agents coordinating with communication
- coop_ablation: Two agents with communication (same as coop, ablation is in execution)
"""

import argparse
import asyncio

from dotenv import load_dotenv

from cooperbench import BenchSetting, FileInterface
from cooperbench.planning.coop import run_planning as run_coop_planning
from cooperbench.planning.single import run_planning as run_single_planning
from cooperbench.planning.solo import run_planning as run_solo_planning

load_dotenv()


async def create_plan(
    file_interface: FileInterface,
    max_iterations: int = 25,
) -> None:
    """Main planning function that can be called from anywhere.

    Args:
        file_interface: FileInterface object for file management
        max_iterations: Maximum number of planning iterations
    """
    file_interface.setup_filesystem(setup_both_agent_workspaces=False)
    try:
        if file_interface.setting == BenchSetting.SINGLE:
            plan1, trajectory = await run_single_planning(
                file_interface=file_interface,
                max_iterations=max_iterations,
            )
            plan2 = None

        elif file_interface.setting == BenchSetting.SOLO:
            assert file_interface.feature2_id is not None
            plan1, trajectory = await run_solo_planning(
                file_interface=file_interface,
                max_iterations=max_iterations,
            )
            plan2 = None

        elif file_interface.setting in (BenchSetting.COOP, BenchSetting.COOP_ABLATION):
            # Both coop and coop_ablation use the same planning with communication
            # The difference is only in the execution phase
            assert file_interface.feature2_id is not None
            plan1, plan2, trajectory = await run_coop_planning(
                file_interface=file_interface,
                max_iterations=max_iterations,
            )

        else:
            raise ValueError(f"Unknown setting: {file_interface.setting}")

        file_interface.save_plan(plan1, first=True)
        file_interface.save_planning_trajectory(trajectory)
        print(f"Planning complete. Plan saved to {file_interface.file_paths['plan1']}")
        if plan2:
            file_interface.save_plan(plan2, first=False)
            print(f"Saved second plan to {file_interface.file_paths['plan2']}")

    except Exception as e:
        raise Exception(f"Planning failed: {str(e)}")


async def main() -> None:
    """CLI wrapper for create_plan()."""
    parser = argparse.ArgumentParser(description="CooperBench unified planning entrypoint")

    parser.add_argument(
        "--setting",
        "-s",
        default="coop",
        choices=["single", "solo", "coop", "coop_ablation"],
        help="Experiment setting mode",
    )
    parser.add_argument("--repo-name", type=str, required=True, help="Repository name")
    parser.add_argument("--task-id", type=int, required=True, help="Task number")
    parser.add_argument("--model1", "-m1", required=True, help="Model for first agent")
    parser.add_argument("--model2", "-m2", help="Model for second agent")
    parser.add_argument("--feature1-id", "-i", required=True, type=int, help="First feature ID")
    parser.add_argument("--feature2-id", "-j", type=int, help="Second feature ID")
    parser.add_argument("--save-to-hf", action="store_true", help="Save results to HuggingFace")
    parser.add_argument("--create-pr", action="store_true", help="Create PR when saving to HF")
    parser.add_argument("--k", type=int, default=1, help="Experiment run identifier")
    parser.add_argument("--max-iterations", type=int, default=25, help="Max planning iterations")

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

    await create_plan(file_interface, args.max_iterations)


if __name__ == "__main__":
    asyncio.run(main())
