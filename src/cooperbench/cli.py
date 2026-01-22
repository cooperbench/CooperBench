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
    exec_parser.add_argument("--plan-location", default="logs", choices=["logs", "cache", "hf"], help="Where to load plans from")

    # Evaluate command (placeholder)
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation phase")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--patch-location", default="logs", choices=["logs", "cache", "hf"], help="Where to load patches from")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "plan":
        asyncio.run(_run_plan(args))
    elif args.command == "execute":
        print("Execute command not yet implemented in the new codebase")
        sys.exit(1)
    elif args.command == "evaluate":
        print("Evaluate command not yet implemented in the new codebase")
        sys.exit(1)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--setting", "-s",
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
    parser.add_argument("--not-save-to-hf", action="store_true", help="Do not save to HuggingFace")
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
        save_to_hf=not args.not_save_to_hf,
        create_pr=args.create_pr,
    )

    await create_plan(file_interface, args.max_iterations)


if __name__ == "__main__":
    main()
