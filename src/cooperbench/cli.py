"""
Command-line interface for CooperBench.

Provides the main CLI entry point for running experiments.
"""

import argparse
import asyncio
import sys

from cooperbench import BenchSetting, FileInterface


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cooperbench",
        description="CooperBench: Multi-agent coordination benchmark for code collaboration",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Run planning phase")
    _add_common_args(plan_parser)
    plan_parser.add_argument("--max-iterations", type=int, default=25, help="Max planning iterations")

    # Execute command (placeholder)
    exec_parser = subparsers.add_parser("execute", help="Run execution phase")
    _add_common_args(exec_parser)
    exec_parser.add_argument(
        "--plan-location", default="logs", choices=["logs", "cache", "hf"], help="Where to load plans from"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation phase")
    _add_common_args(eval_parser)
    eval_parser.add_argument(
        "--patch-location", default="logs", choices=["logs", "cache", "hf"], help="Where to load patches from"
    )
    eval_parser.add_argument(
        "--eval-type",
        default="test",
        choices=["test", "merge"],
        help="Evaluation type: test (single/solo) or merge (coop)",
    )

    # Run command (plan + execute + evaluate)
    run_parser = subparsers.add_parser("run", help="Run full pipeline: plan → execute → evaluate")
    _add_common_args(run_parser)
    run_parser.add_argument("--max-iterations", type=int, default=25, help="Max planning iterations")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "plan":
        asyncio.run(_run_plan(args))
    elif args.command == "execute":
        asyncio.run(_run_execute(args))
    elif args.command == "evaluate":
        asyncio.run(_run_evaluate(args))
    elif args.command == "run":
        asyncio.run(_run_full(args))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--setting",
        "-s",
        required=True,
        choices=["single", "solo", "coop", "coop_ablation"],
        help="Experiment setting mode",
    )
    parser.add_argument("--repo-name", required=True, type=str, help="Repository name")
    parser.add_argument("--task-id", required=True, type=int, help="Task number")
    parser.add_argument("--model1", "-m1", required=True, help="Model for first agent")
    parser.add_argument("--model2", "-m2", help="Model for second agent (coop modes)")
    parser.add_argument("--feature1-id", "-i", required=True, type=int, help="First feature ID")
    parser.add_argument("--feature2-id", "-j", type=int, help="Second feature ID (non-single modes)")
    parser.add_argument("--k", type=int, default=1, help="Experiment run identifier")
    parser.add_argument("--save-to-hf", action="store_true", help="Save results to HuggingFace")
    parser.add_argument("--create-pr", action="store_true", help="Create PR when saving to HF")


async def _run_plan(args: argparse.Namespace) -> None:
    """Run the planning phase."""
    from cooperbench.planning import create_plan

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


async def _run_execute(args: argparse.Namespace) -> None:
    """Run the execution phase."""
    from cooperbench.execution import create_execution

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


async def _run_evaluate(args: argparse.Namespace) -> None:
    """Run the evaluation phase."""
    from cooperbench.evaluation import evaluate

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

    await evaluate(file_interface, args.eval_type, args.patch_location)


async def _run_full(args: argparse.Namespace) -> None:
    """Run full pipeline: plan → execute → evaluate."""
    from cooperbench.evaluation import evaluate
    from cooperbench.execution import create_execution
    from cooperbench.planning import create_plan

    setting = BenchSetting(args.setting)
    eval_type = "merge" if setting in (BenchSetting.COOP, BenchSetting.COOP_ABLATION) else "test"

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

    print("\n[1/3] Planning...")
    await create_plan(file_interface, args.max_iterations)

    print("\n[2/3] Executing...")
    await create_execution(file_interface, "logs")

    print("\n[3/3] Evaluating...")
    await evaluate(file_interface, eval_type, "logs")


if __name__ == "__main__":
    main()
