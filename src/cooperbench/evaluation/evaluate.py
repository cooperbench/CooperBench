"""
Unified evaluation entrypoint for CooperBench.

Supports evaluation modes:
- merge: Test AND merge conflict analysis between features (coop/coop_ablation)
- test: Test execution only (single/solo)
- aggregate: Comprehensive metrics generation across multiple features
"""

import argparse
import asyncio
import json
import logging
import time
from typing import Literal

from dotenv import load_dotenv

from cooperbench import BenchSetting, FileInterface
from cooperbench.core.git import run_git_command
from cooperbench.core.merge import analyze_merge, merge, merge_union
from cooperbench.core.paths import get_branch_name
from cooperbench.evaluation.llm_merge import apply_llm_resolutions
from cooperbench.evaluation.reporter import generate_json_report, generate_single_aggregate_table
from cooperbench.evaluation.test_runner import run_tests, run_tests_with_patch

load_dotenv()

logger = logging.getLogger(__name__)


def _print_merge_summary(results: dict, file_interface: FileInterface) -> None:
    """Print a summary of merge evaluation results and save to file."""
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
    
    def mark(passed: bool | None) -> str:
        if passed is True:
            return f"{GREEN}✓ pass{RESET}"
        elif passed is False:
            return f"{RED}✗ fail{RESET}"
        return f"{YELLOW}? skip{RESET}"
    
    merge_status = results.get("merge_status", "unknown")
    conflict_score = results.get("conflict_score", 0)
    f1 = results.get("feature1", {})
    f2 = results.get("feature2", {})
    
    naive_f1 = f1.get("naive_merge_test_passed")
    naive_f2 = f2.get("naive_merge_test_passed")
    union_f1 = f1.get("union_merge_test_passed")
    union_f2 = f2.get("union_merge_test_passed")
    llm_f1 = f1.get("llm_merge_test_passed")
    llm_f2 = f2.get("llm_merge_test_passed")
    
    # Determine overall result
    any_passed = (
        ((naive_f1 is True) and (naive_f2 is True)) or
        ((union_f1 is True) and (union_f2 is True)) or
        ((llm_f1 is True) and (llm_f2 is True))
    )
    overall = "pass" if any_passed else "fail"
    
    # Save simple result file
    from cooperbench.core.paths import get_log_path
    result_path = get_log_path(file_interface.file_paths["json_merge_report"]).parent / "result.txt"
    result_path.write_text(f"{overall}\n")
    
    status_color = GREEN if merge_status == "clean" else RED
    result_color = GREEN if any_passed else RED
    print(f"\nMerge: {status_color}{merge_status}{RESET} (conflict_score={conflict_score})")
    
    if naive_f1 is not None:
        print(f"  naive:  f1={mark(naive_f1)} f2={mark(naive_f2)}")
    if union_f1 is not None:
        print(f"  union:  f1={mark(union_f1)} f2={mark(union_f2)}")
    if llm_f1 is not None:
        print(f"  llm:    f1={mark(llm_f1)} f2={mark(llm_f2)}")
    
    print(f"Result: {result_color}{overall}{RESET}")


async def evaluate(
    file_interface: FileInterface,
    eval_type: Literal["merge", "test", "aggregate"],
    file_location: Literal["logs", "cache", "hf"],
    filter_patch: bool = True,
    force_reevaluation: bool = False,
) -> dict | None:
    """Unified evaluation function.

    Args:
        file_interface: FileInterface object for file operations
        eval_type: Type of evaluation ("merge", "test", or "aggregate")
        file_location: Location of files ("logs", "cache", "hf")
        filter_patch: Whether to filter non-code changes from patch
        force_reevaluation: Whether to force re-evaluation (for aggregate)

    Returns:
        dict: Evaluation results (None for aggregate)
    """
    try:
        file_interface.setup_filesystem(check_for_conflicts=(eval_type == "merge"))

        if eval_type == "merge":
            results = await run_merge_evaluation(file_interface, file_location, filter_patch)
            _print_merge_summary(results, file_interface)
        elif eval_type == "test":
            if file_interface.setting == BenchSetting.SOLO:
                results = await run_solo_evaluation(file_interface, file_location, filter_patch)
                print(
                    f"\nSolo evaluation completed. Both tests {'passed' if results['both_tests_passed'] else 'failed'}"
                )
            else:
                results = await run_test_evaluation(file_interface, file_location, filter_patch)
                print(f"\nTest evaluation completed. Tests {'passed' if results['test_passed'] else 'failed'}")
        elif eval_type == "aggregate":
            await run_aggregate_evaluation(file_interface, force_reevaluation, file_location, filter_patch)
            print("\nAggregate evaluation completed.")
            return None

        return results

    finally:
        file_interface.flush_pending_uploads()


async def run_test_evaluation(
    file_interface: FileInterface,
    file_location: Literal["logs", "cache", "hf"],
    filter_patch: bool,
) -> dict:
    """Run test evaluation for a single feature.

    Args:
        file_interface: FileInterface object
        file_location: Location of the patch file
        filter_patch: Whether to filter non-code changes

    Returns:
        dict: Test evaluation results
    """
    start_time = time.time()

    repo_name = file_interface.repo_name
    task_id = file_interface.task_id
    feature1_id = file_interface.feature1_id

    logger.info(f"Running test evaluation for {repo_name}/task{task_id}/feature{feature1_id}")

    # Get the feature patch path from logs/cache/hf
    feature_patch_path = file_interface.get_patch(file_location, first=True)

    container_name = f"{repo_name}_{task_id}_f{feature1_id}_k{file_interface.k}"

    test_passed, error_info = run_tests_with_patch(
        repo_name=repo_name,
        task_id=task_id,
        task_dir=file_interface.task_folder_path,
        feature_patch=feature_patch_path,
        test_patch=file_interface.get_tests_patch_path(first=True),
        container_name=container_name,
    )

    results = {
        "task_name": repo_name,
        "task_id": task_id,
        "feature_num": feature1_id,
        "k": file_interface.k,
        "test_passed": test_passed,
        "error_info": error_info,
        "duration": time.time() - start_time,
    }

    file_interface.save_test_report(results)
    return results


async def run_solo_evaluation(
    file_interface: FileInterface,
    file_location: Literal["logs", "cache", "hf"],
    filter_patch: bool,
) -> dict:
    """Run evaluation for solo mode (one patch, two feature tests).

    Args:
        file_interface: FileInterface object
        file_location: Location of the patch file
        filter_patch: Whether to filter non-code changes

    Returns:
        dict: Evaluation results
    """
    start_time = time.time()

    repo_name = file_interface.repo_name
    task_id = file_interface.task_id
    feature1_id = file_interface.feature1_id
    feature2_id = file_interface.feature2_id

    logger.info(f"Running solo evaluation for {repo_name}/task{task_id}")

    # Get the unified patch (solo mode has one patch for both features)
    feature_patch_path = file_interface.get_patch(file_location, first=True)

    # Test feature 1
    container_name1 = f"{repo_name}_{task_id}_f{feature1_id}_f{feature2_id}_test1"
    feature1_tests_passed, output1 = run_tests_with_patch(
        repo_name=repo_name,
        task_id=task_id,
        task_dir=file_interface.task_folder_path,
        feature_patch=feature_patch_path,
        test_patch=file_interface.get_tests_patch_path(first=True),
        container_name=container_name1,
    )

    # Test feature 2
    container_name2 = f"{repo_name}_{task_id}_f{feature1_id}_f{feature2_id}_test2"
    feature2_tests_passed, output2 = run_tests_with_patch(
        repo_name=repo_name,
        task_id=task_id,
        task_dir=file_interface.task_folder_path,
        feature_patch=feature_patch_path,
        test_patch=file_interface.get_tests_patch_path(first=False),
        container_name=container_name2,
    )

    both_tests_passed = feature1_tests_passed and feature2_tests_passed

    results = {
        "task_name": repo_name,
        "task_id": task_id,
        "feature1_id": feature1_id,
        "feature2_id": feature2_id,
        "k": file_interface.k,
        "feature1_test_passed": feature1_tests_passed,
        "feature1_test_output": output1,
        "feature2_test_passed": feature2_tests_passed,
        "feature2_test_output": output2,
        "both_tests_passed": both_tests_passed,
        "duration": time.time() - start_time,
    }

    file_interface.save_solo_test_report(results)
    return results


async def _apply_merged_patch_and_run_tests(
    file_interface: FileInterface,
    file_location: Literal["logs", "cache", "hf"],
    filter_patch: bool,
    merge_type: str,
    merge_results: dict,
) -> dict:
    """Apply the merged patch and run tests for both features.

    Args:
        file_interface: FileInterface object
        file_location: Location of the patch file
        filter_patch: Whether to filter non-code changes
        merge_type: Type of merge to test (naive, union, llm)
        merge_results: Report dictionary to update

    Returns:
        Updated merge results dictionary
    """
    feature1_id = file_interface.feature1_id
    feature2_id = file_interface.feature2_id
    assert feature2_id is not None

    repo_name = file_interface.repo_name
    task_id = file_interface.task_id
    merged_patch_path = file_interface.get_merge_diff_path(merge_type=merge_type, file_location=file_location)

    try:
        # Test feature 1 with merged patch
        container_name1 = f"{repo_name}_{task_id}_{feature1_id}_{feature2_id}_{merge_type}_f1"
        feature1_tests_passed, output1 = run_tests_with_patch(
            repo_name=repo_name,
            task_id=task_id,
            task_dir=file_interface.task_folder_path,
            feature_patch=merged_patch_path,
            test_patch=file_interface.get_tests_patch_path(first=True),
            container_name=container_name1,
        )

        # Test feature 2 with merged patch
        container_name2 = f"{repo_name}_{task_id}_{feature1_id}_{feature2_id}_{merge_type}_f2"
        feature2_tests_passed, output2 = run_tests_with_patch(
            repo_name=repo_name,
            task_id=task_id,
            task_dir=file_interface.task_folder_path,
            feature_patch=merged_patch_path,
            test_patch=file_interface.get_tests_patch_path(first=False),
            container_name=container_name2,
        )

        merge_results["feature1"][f"{merge_type}_merge_test_passed"] = feature1_tests_passed
        merge_results["feature1"][f"{merge_type}_merge_test_output"] = output1
        merge_results["feature2"][f"{merge_type}_merge_test_passed"] = feature2_tests_passed
        merge_results["feature2"][f"{merge_type}_merge_test_output"] = output2

    except Exception as e:
        logger.error(f"Testing of {merge_type} patch failed: {e}")

    return merge_results


async def run_merge_evaluation(
    file_interface: FileInterface,
    file_location: Literal["logs", "cache", "hf"],
    filter_patch: bool,
) -> dict:
    """Run merge evaluation between two features (coop/coop_ablation).

    Supports naive merge, union merge, and LLM merge strategies.

    Args:
        file_interface: FileInterface object
        file_location: Location of patch files
        filter_patch: Whether to filter non-code changes

    Returns:
        dict: Merge evaluation results
    """
    start_time = time.time()

    feature1_id = file_interface.feature1_id
    feature2_id = file_interface.feature2_id
    assert feature2_id, "Feature 2 must be specified for merge evaluation"

    agent_workspace1_path = file_interface.agent_workspace1_path
    agent_workspace2_path = file_interface.agent_workspace2_path

    logger.info(f"Running merge evaluation for features {feature1_id} and {feature2_id}")

    # Reset workspaces
    if file_location in ["logs", "cache", "hf"]:
        run_git_command(
            agent_workspace1_path,
            "reset",
            "--hard",
            file_interface.base_commit,
            check=False,
            capture_output=True,
        )
        run_git_command(
            agent_workspace2_path,
            "reset",
            "--hard",
            file_interface.base_commit,
            check=False,
            capture_output=True,
        )

    # Apply patches
    file_interface.apply_patch(file_location, first=True, filter_patch=filter_patch)
    file_interface.apply_patch(file_location, first=False, filter_patch=filter_patch)

    # Initialize report
    report = file_interface.get_merge_report_metadata()
    report["feature1"]["tests_passed"] = None
    report["feature1"]["test_output"] = None
    report["feature2"]["tests_passed"] = None
    report["feature2"]["test_output"] = None

    # Perform naive merge analysis
    diff_output, conflict = merge(
        agent_workspace2_path,
        get_branch_name(
            file_interface.setting,
            file_interface.k,
            feature1_id,
            feature2_id,
        ),
    )
    report = analyze_merge(diff_output, conflict, report)
    file_interface.save_merge_diff_file(diff_output, merge_type="naive")

    test_script_path = file_interface.get_test_script_path()
    need_union_merge = False

    if report["merge_status"] == "clean":
        # Test naive merge
        report = await _apply_merged_patch_and_run_tests(file_interface, file_location, filter_patch, "naive", report)
    elif report["merge_status"] == "conflicts":
        need_union_merge = True

    if need_union_merge:
        # Union merge
        logger.debug("Running merge with union strategy")
        union_diff_output, _ = merge_union(
            agent_workspace2_path,
            get_branch_name(
                file_interface.setting,
                file_interface.k,
                feature1_id,
                feature2_id,
            ),
        )
        file_interface.save_merge_diff_file(union_diff_output, merge_type="union")

        report = await _apply_merged_patch_and_run_tests(file_interface, file_location, filter_patch, "union", report)

        union_ok = bool(report["feature1"].get("union_merge_test_passed")) and bool(
            report["feature2"].get("union_merge_test_passed")
        )

        # LLM merge only if union tests failed
        if not union_ok:
            logger.debug("Attempting LLM-based conflict resolution")
            run_git_command(agent_workspace2_path, "reset", "--hard", check=False, capture_output=True)

            # Re-do merge to get conflicts
            diff_output, conflict = merge(
                agent_workspace2_path,
                get_branch_name(
                    file_interface.setting,
                    file_interface.k,
                    feature1_id,
                    feature2_id,
                ),
            )

            if conflict:
                total_conflicts, resolved_conflicts = apply_llm_resolutions(agent_workspace2_path)
                logger.info(f"LLM resolved {resolved_conflicts}/{total_conflicts} conflicts")

                llm_diff_output = run_git_command(
                    agent_workspace2_path,
                    "diff",
                    "HEAD",
                    capture_output=True,
                    return_returncode=False,
                )
                assert isinstance(llm_diff_output, str)

                file_interface.save_merge_diff_file(llm_diff_output, merge_type="llm")

                run_git_command(agent_workspace2_path, "add", "-A", check=True, capture_output=True)
                run_git_command(
                    agent_workspace2_path,
                    "commit",
                    "-m",
                    f"LLM merge resolution for features {feature1_id} and {feature2_id}",
                    check=True,
                    capture_output=True,
                )

                # Test LLM merge
                feature1_tests_passed, output1 = run_tests(
                    agent_workspace2_path,
                    feature1_id,
                    test_script_path,
                    file_interface.get_tests_patch_path(first=True),
                    return_errors=True,
                )

                feature2_tests_passed, output2 = run_tests(
                    agent_workspace2_path,
                    feature2_id,
                    test_script_path,
                    file_interface.get_tests_patch_path(first=False),
                    return_errors=True,
                )

                report["feature1"]["llm_merge_test_passed"] = feature1_tests_passed
                report["feature1"]["llm_merge_test_output"] = output1
                report["feature2"]["llm_merge_test_passed"] = feature2_tests_passed
                report["feature2"]["llm_merge_test_output"] = output2
            else:
                logger.warning("No conflicts found for LLM merge (unexpected)")

    report["duration"] = time.time() - start_time
    file_interface.save_json_merge_report(generate_json_report(report))

    return report


async def run_aggregate_evaluation(
    file_interface: FileInterface,
    force_reevaluation: bool,
    file_location: Literal["logs", "cache", "hf"],
    filter_patch: bool,
) -> None:
    """Aggregate evaluation results for a PR across all features.

    Args:
        file_interface: FileInterface object
        force_reevaluation: Whether to force re-evaluation
        file_location: Location of logs
        filter_patch: Whether to filter non-code changes
    """
    feature1_id = file_interface.feature1_id
    feature2_id = file_interface.feature2_id
    assert feature2_id, "Feature 2 must be specified (end of range)"
    setting = file_interface.setting

    results = {
        "setting": setting.value,
        "repo_name": file_interface.repo_name,
        "task_id": file_interface.task_id,
        "k": file_interface.k,
    }

    async def _run_single_aggregation() -> None:
        detailed_results = []
        success = 0
        table_data = {}

        k_target = file_interface.k
        for i in range(feature1_id, feature2_id + 1):
            logger.info(f"Aggregating results for feature {i}, k={k_target}")
            task_file_interface = file_interface.copy_with_params(feature1_id=i, feature2_id=None, k=k_target)
            try:
                data = task_file_interface.get_test_report(file_location)
                if force_reevaluation or not data:
                    raise FileNotFoundError("Forcing re-evaluation or report not found")
            except Exception:
                task_file_interface.setup_filesystem(setup_both_agent_workspaces=True)
                await run_test_evaluation(task_file_interface, file_location, filter_patch)
                data = task_file_interface.get_test_report(file_location)

            test_passed = data["test_passed"]
            success += int(test_passed)
            table_data[(i, 1)] = int(test_passed)
            detailed_results.append({"feature_id": i, "k": k_target, "test_passed": test_passed})

        total_tasks = len(detailed_results)
        success_pct = (success / total_tasks) * 100.0 if total_tasks > 0 else 0.0

        table_output = generate_single_aggregate_table(table_data, 1, feature1_id, feature2_id, success_pct)
        logger.info(table_output)

        results.update(
            {
                "total_tasks": total_tasks,
                "success": success,
                "success_pct": success_pct,
                "detailed_results": detailed_results,
            }
        )
        file_interface.save_aggregate_eval_file(results)
        logger.info(f"Aggregate evaluation completed. Success rate: {success_pct:.1f}% ({success}/{total_tasks})")

    async def _run_multi_aggregation() -> None:
        naive_merge_success, union_merge_success, llm_merge_success, total_tasks = 0, 0, 0, 0
        detailed_results = []

        for i in range(feature1_id, feature2_id):
            for j in range(i + 1, feature2_id + 1):
                total_tasks += 1
                logger.info(f"Aggregating results for feature pair ({i}, {j})")

                task_file_interface = file_interface.copy_with_params(feature1_id=i, feature2_id=j)
                try:
                    data = task_file_interface.get_json_merge_report(file_location)
                    if force_reevaluation or not data:
                        raise FileNotFoundError("Forcing re-evaluation or report not found")
                except Exception:
                    task_file_interface.setup_filesystem(setup_both_agent_workspaces=True)
                    await run_merge_evaluation(task_file_interface, file_location, filter_patch)
                    data = task_file_interface.get_json_merge_report(file_location)

                has_conflict = data["merge_analysis"]["status"] == "conflicts"
                feature1_naive_merge_test_passed = data["features"]["feature1"].get("naive_merge_test_passed")
                feature2_naive_merge_test_passed = data["features"]["feature2"].get("naive_merge_test_passed")
                feature1_union_merge_test_passed = data["features"]["feature1"].get("union_merge_test_passed")
                feature2_union_merge_test_passed = data["features"]["feature2"].get("union_merge_test_passed")
                feature1_llm_merge_test_passed = data["features"]["feature1"].get("llm_merge_test_passed")
                feature2_llm_merge_test_passed = data["features"]["feature2"].get("llm_merge_test_passed")

                naive_ok = (has_conflict is False) and (
                    bool(feature1_naive_merge_test_passed) and bool(feature2_naive_merge_test_passed)
                )
                union_ok_raw = bool(feature1_union_merge_test_passed) and bool(feature2_union_merge_test_passed)
                llm_ok_raw = bool(feature1_llm_merge_test_passed) and bool(feature2_llm_merge_test_passed)

                union_ok = union_ok_raw and not naive_ok
                llm_ok = llm_ok_raw and not naive_ok and not union_ok

                naive_merge_success += 1 if naive_ok else 0
                union_merge_success += 1 if union_ok else 0
                llm_merge_success += 1 if llm_ok else 0

                detailed_results.append(
                    {
                        "feature1_id": i,
                        "feature2_id": j,
                        "has_naive_merge_conflict": has_conflict,
                        "feature1_naive_merge_test_passed": feature1_naive_merge_test_passed,
                        "feature2_naive_merge_test_passed": feature2_naive_merge_test_passed,
                        "feature1_union_merge_test_passed": feature1_union_merge_test_passed,
                        "feature2_union_merge_test_passed": feature2_union_merge_test_passed,
                        "feature1_llm_merge_test_passed": feature1_llm_merge_test_passed,
                        "feature2_llm_merge_test_passed": feature2_llm_merge_test_passed,
                    }
                )

        if total_tasks == 0:
            raise ValueError("No tasks found to evaluate")

        naive_merge_success_pct = (naive_merge_success / total_tasks) * 100.0
        union_merge_success_pct = ((naive_merge_success + union_merge_success) / total_tasks) * 100.0
        llm_merge_success_pct = ((naive_merge_success + union_merge_success + llm_merge_success) / total_tasks) * 100.0

        results.update(
            {
                "total_tasks": total_tasks,
                "naive_merge_success": naive_merge_success,
                "naive_merge_success_pct": naive_merge_success_pct,
                "union_merge_success": naive_merge_success + union_merge_success,
                "union_merge_success_pct": union_merge_success_pct,
                "llm_merge_success": naive_merge_success + union_merge_success + llm_merge_success,
                "llm_merge_success_pct": llm_merge_success_pct,
                "detailed_results": detailed_results,
            }
        )
        file_interface.save_aggregate_eval_file(results)

        logger.info(
            f"\n\nAggregate Results:\n"
            f"  total_tasks: {total_tasks}\n"
            f"  naive_merge_success: {naive_merge_success} ({naive_merge_success_pct:.1f}%)\n"
            f"  union_merge_success: {naive_merge_success + union_merge_success} ({union_merge_success_pct:.1f}%)\n"
            f"  llm_merge_success: {naive_merge_success + union_merge_success + llm_merge_success} ({llm_merge_success_pct:.1f}%)"
        )

    async def _run_solo_aggregation() -> None:
        detailed_results = []
        both_tests_success = 0
        total_tasks = 0

        for i in range(feature1_id, feature2_id):
            for j in range(i + 1, feature2_id + 1):
                total_tasks += 1
                logger.info(f"Aggregating solo results for feature pair ({i}, {j})")

                task_file_interface = file_interface.copy_with_params(feature1_id=i, feature2_id=j)
                try:
                    data = task_file_interface.get_solo_test_report(file_location)
                    if force_reevaluation or not data:
                        raise FileNotFoundError("Forcing re-evaluation or report not found")
                except Exception:
                    task_file_interface.setup_filesystem(setup_both_agent_workspaces=True)
                    await run_solo_evaluation(task_file_interface, file_location, filter_patch)
                    data = task_file_interface.get_solo_test_report(file_location)

                feature1_test_passed = data["feature1_test_passed"]
                feature2_test_passed = data["feature2_test_passed"]
                both_tests_passed = data["both_tests_passed"]

                both_tests_success += 1 if both_tests_passed else 0

                detailed_results.append(
                    {
                        "feature1_id": i,
                        "feature2_id": j,
                        "feature1_test_passed": feature1_test_passed,
                        "feature2_test_passed": feature2_test_passed,
                        "both_tests_passed": both_tests_passed,
                    }
                )

        if total_tasks == 0:
            raise ValueError("No tasks found to evaluate")

        both_tests_success_pct = (both_tests_success / total_tasks) * 100.0

        results.update(
            {
                "total_tasks": total_tasks,
                "both_tests_success": both_tests_success,
                "both_tests_success_pct": both_tests_success_pct,
                "detailed_results": detailed_results,
            }
        )
        file_interface.save_aggregate_eval_file(results)

        logger.info(
            f"\n\nSolo Aggregate Results:\n"
            f"  total_tasks: {total_tasks}\n"
            f"  both_tests_success: {both_tests_success} ({both_tests_success_pct:.1f}%)"
        )

    if setting == BenchSetting.SINGLE:
        await _run_single_aggregation()
    elif setting == BenchSetting.SOLO:
        await _run_solo_aggregation()
    else:
        await _run_multi_aggregation()


async def main() -> None:
    """CLI wrapper for evaluate()."""
    parser = argparse.ArgumentParser(description="CooperBench unified evaluation entrypoint")

    parser.add_argument(
        "evaluation_type",
        choices=["merge", "test", "aggregate"],
        help="Type of evaluation to perform",
    )
    parser.add_argument(
        "--setting",
        "-s",
        required=True,
        choices=[s.value for s in BenchSetting],
        help="Experiment setting mode",
    )
    parser.add_argument("--repo-name", required=True, type=str, help="Repository name")
    parser.add_argument("--task-id", required=True, type=int, help="Task number")
    parser.add_argument("--model1", "-m1", required=True, help="Model for first agent")
    parser.add_argument("--model2", "-m2", help="Model for second agent")
    parser.add_argument("--feature1-id", "-i", required=True, type=int, help="First feature ID")
    parser.add_argument("--feature2-id", "-j", type=int, help="Second feature ID (or end of range for aggregate)")
    parser.add_argument("--k", type=int, default=1, help="Experiment run identifier")
    parser.add_argument("--save-to-hf", action="store_true", help="Save results to HuggingFace")
    parser.add_argument("--create-pr", action="store_true", help="Create PR when saving to HF")
    parser.add_argument(
        "--file-location",
        choices=["logs", "cache", "hf"],
        default="logs",
        help="Where to load patches from",
    )
    parser.add_argument(
        "--not-filter-patch",
        action="store_true",
        help="Do not filter non-code changes from patch",
    )
    parser.add_argument(
        "--force-reevaluation",
        "-fe",
        action="store_true",
        help="Force re-evaluation even if results exist (aggregate only)",
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

    await evaluate(
        file_interface,
        args.evaluation_type,
        args.file_location,
        filter_patch=not args.not_filter_patch,
        force_reevaluation=args.force_reevaluation,
    )


if __name__ == "__main__":
    asyncio.run(main())
