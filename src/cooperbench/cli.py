"""CooperBench CLI - benchmark runner.

Usage:
    cooperbench run -n my-experiment --setting solo -r llama_index_task
    cooperbench run --setting solo -s lite  # auto-generates name: solo-lite-gemini-3-flash
    cooperbench eval -n my-experiment --force
"""

import argparse
import os
import sys

os.environ["LITELLM_LOG"] = "ERROR"

import litellm

litellm.suppress_debug_info = True  # Suppress "Give Feedback / Get Help" print messages on errors

from cooperbench.agents import get_agent_shorthand  # noqa: E402
from cooperbench.utils import clean_model_name  # noqa: E402


def _generate_run_name(
    setting: str,
    model: str,
    agent: str = "mini_swe_agent",
    subset: str | None = None,
    repo: str | None = None,
    task: int | None = None,
    git_enabled: bool = False,
) -> str:
    """Generate experiment name from parameters.

    Format: {setting}-{agent_short}-{git?}-{model}-{subset?}-{repo?}-{task?}
    Examples:
        solo-msa-gemini-3-flash
        solo-oh-gemini-3-flash-lite
        coop-sw-git-gpt-4o-dspy-8394
    """
    parts = [setting]

    # Add agent shorthand (defined in cooperbench/agents/__init__.py)
    parts.append(get_agent_shorthand(agent))

    if git_enabled:
        parts.append("git")
    parts.append(clean_model_name(model))
    if subset:
        parts.append(subset)
    if repo:
        # Shorten repo name (e.g., llama_index_task -> llama-index)
        repo_short = repo.replace("_task", "").replace("_", "-")
        parts.append(repo_short)
    if task is not None:
        parts.append(str(task))
    return "-".join(parts)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cooperbench",
        description="CooperBench benchmark runner",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === config command ===
    config_parser = subparsers.add_parser(
        "config",
        help="Configure execution backends",
        description="Interactive configuration wizards for backends (GCP, Modal, etc.)",
    )
    config_subparsers = config_parser.add_subparsers(dest="backend", required=True)

    # config gcp
    gcp_config_parser = config_subparsers.add_parser(
        "gcp",
        help="Configure GCP backend",
        description="Set up Google Cloud Platform as execution backend",
    )
    gcp_config_parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip validation tests (faster setup, but no verification)",
    )

    # === run command ===
    run_parser = subparsers.add_parser(
        "run",
        help="Run benchmark tasks",
        description="Run agents on CooperBench tasks",
    )
    run_parser.add_argument(
        "-n",
        "--name",
        help="Experiment name (auto-generated if not provided)",
    )
    run_parser.add_argument(
        "-s",
        "--subset",
        help="Use a predefined subset (e.g., lite). See dataset/subsets/",
    )
    run_parser.add_argument(
        "-r",
        "--repo",
        help="Filter by repository name (e.g., llama_index_task)",
    )
    run_parser.add_argument(
        "-t",
        "--task",
        type=int,
        help="Filter by task ID",
    )
    run_parser.add_argument(
        "-f",
        "--features",
        help="Specific feature pair to run, comma-separated (e.g., 1,2)",
    )
    run_parser.add_argument(
        "-m",
        "--model",
        default="vertex_ai/gemini-3-flash-preview",
        help="LLM model to use (default: vertex_ai/gemini-3-flash-preview)",
    )
    run_parser.add_argument(
        "-a",
        "--agent",
        default="mini_swe_agent",
        help="Agent framework to use (default: mini_swe_agent)",
    )
    run_parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=30,
        help="Number of parallel tasks (default: 20)",
    )
    run_parser.add_argument(
        "--setting",
        choices=["coop", "solo"],
        default="coop",
        help="Benchmark setting: coop (2 agents) or solo (1 agent) (default: coop)",
    )
    run_parser.add_argument(
        "--redis",
        default="redis://localhost:6379",
        help="Redis URL for inter-agent communication (default: redis://localhost:6379)",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if results exist",
    )
    run_parser.add_argument(
        "--git",
        action="store_true",
        help="Enable git collaboration (agents can push/pull/merge via shared remote)",
    )
    run_parser.add_argument(
        "--no-messaging",
        action="store_true",
        help="Disable messaging (send_message command)",
    )
    run_parser.add_argument(
        "--no-auto-eval",
        action="store_true",
        help="Disable automatic evaluation after completion (enabled by default)",
    )
    run_parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=10,
        help="Number of parallel evaluations for auto-eval (default: 10)",
    )
    run_parser.add_argument(
        "--backend",
        choices=["modal", "docker", "gcp"],
        default="modal",
        help="Execution backend: modal (cloud), docker (local), or gcp (GCP VM) (default: modal)",
    )
    run_parser.add_argument(
        "--agent-config",
        help="Path to agent-specific configuration file (format determined by agent)",
    )

    # === eval command ===
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate completed runs",
        description="Evaluate agent runs from logs/ directory",
    )
    eval_parser.add_argument(
        "-n",
        "--name",
        help="Experiment name to evaluate (required for eval)",
    )
    eval_parser.add_argument(
        "-s",
        "--subset",
        help="Use a predefined subset (e.g., lite). See dataset/subsets/",
    )
    eval_parser.add_argument(
        "-r",
        "--repo",
        help="Filter by repository name",
    )
    eval_parser.add_argument(
        "-t",
        "--task",
        type=int,
        help="Filter by task ID",
    )
    eval_parser.add_argument(
        "-f",
        "--features",
        help="Specific feature pair to evaluate, comma-separated (e.g., 1,2)",
    )
    eval_parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=10,
        help="Number of parallel evaluations (default: 10)",
    )
    eval_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if eval.json exists",
    )
    eval_parser.add_argument(
        "--backend",
        choices=["modal", "docker", "gcp"],
        default="modal",
        help="Execution backend: modal (cloud), docker (local), or gcp (GCP Batch) (default: modal)",
    )

    args = parser.parse_args()

    if args.command == "config":
        _config_command(args)
    elif args.command == "run":
        _run_command(args)
    elif args.command == "eval":
        _eval_command(args)


def _config_command(args):
    """Handle the 'config' subcommand."""
    from cooperbench.config import config_gcp_command

    if args.backend == "gcp":
        exit_code = config_gcp_command(skip_tests=args.skip_tests)
        sys.exit(exit_code)


def _run_command(args):
    """Handle the 'run' subcommand."""
    from cooperbench.runner import run

    features = None
    if args.features:
        features = [int(f.strip()) for f in args.features.split(",")]

    # Auto-generate name if not provided
    run_name = args.name
    if not run_name:
        run_name = _generate_run_name(
            setting=args.setting,
            model=args.model,
            agent=args.agent,
            subset=args.subset,
            repo=args.repo,
            task=args.task,
            git_enabled=args.git,
        )

    run(
        run_name=run_name,
        subset=args.subset,
        repo=args.repo,
        task_id=args.task,
        features=features,
        model_name=args.model,
        agent=args.agent,
        concurrency=args.concurrency,
        setting=args.setting,
        redis_url=args.redis,
        force=args.force,
        git_enabled=args.git,
        messaging_enabled=not args.no_messaging,
        auto_eval=not args.no_auto_eval,
        eval_concurrency=args.eval_concurrency,
        backend=args.backend,
        agent_config=args.agent_config if hasattr(args, "agent_config") else None,
    )


def _eval_command(args):
    """Handle the 'eval' subcommand."""
    from cooperbench.eval import evaluate

    if not args.name:
        print("error: -n/--name is required for eval command", file=sys.stderr)
        sys.exit(1)

    features = None
    if args.features:
        features = [int(f.strip()) for f in args.features.split(",")]

    evaluate(
        run_name=args.name,
        subset=args.subset,
        repo=args.repo,
        task_id=args.task,
        features=features,
        concurrency=args.concurrency,
        force=args.force,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
