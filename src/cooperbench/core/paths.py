"""
Path conventions and utilities for CooperBench experiments.

This module defines standardized paths for dataset files, agent workspaces,
and experiment outputs. It handles path generation for different experiment
settings (single, solo, coop, coop_ablation).
"""

from pathlib import Path

from cooperbench.core.settings import BenchSetting

LOGS_DIR = Path("logs")
CACHE_DIR = Path.home() / ".cooperbench_cache"


def ensure_dirs() -> None:
    """Create necessary directories if they do not exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_log_path(file_path: Path) -> Path:
    """Get log path for a given file path."""
    return LOGS_DIR / file_path


def get_cache_dir() -> Path:
    """Get cache directory path."""
    return CACHE_DIR


def get_cache_path(file_path: Path) -> Path:
    """Get cache path for a given file."""
    return get_cache_dir() / file_path


def get_task_folder_path(repo_name: str, task_id: int) -> Path:
    """Get the task folder path for a given repository and task ID.
    
    The task folder structure is: dataset/<repo_name>/task<task_id>
    """
    return Path("dataset") / repo_name / f"task{task_id}"


def get_test_script_path(task_folder: Path) -> Path:
    """Get the test script path for a given task folder.
    
    The test script is expected to be run_tests.sh in the task folder.
    """
    test_script = task_folder / "run_tests.sh"
    if not test_script.exists():
        raise FileNotFoundError(f"Test script file {test_script} not found.")
    test_script.chmod(0o755)
    return test_script


def get_feature_folder_path(task_folder: Path, feature_num: int) -> Path:
    """Get the feature folder path for a given task folder and feature number.
    
    The feature folder structure is: task_folder/feature<feature_num>
    """
    return task_folder / f"feature{feature_num}"


def get_feature_file_path(task_folder: Path, feature_num: int) -> Path:
    """Get the path to the feature.md file."""
    feature_file = get_feature_folder_path(task_folder, feature_num) / "feature.md"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature description file {feature_file} not found.")
    return feature_file


def get_tests_patch_path(task_folder: Path, feature_num: int) -> Path:
    """Get the path to the tests.patch file."""
    test_patch = get_feature_folder_path(task_folder, feature_num) / "tests.patch"
    if not test_patch.exists():
        raise FileNotFoundError(f"Tests patch file {test_patch} not found.")
    return test_patch


def get_golden_feature_patch(task_folder: Path, feature_num: int) -> Path:
    """Get the path to the golden feature.patch file (hand-written implementation)."""
    golden_patch = get_feature_folder_path(task_folder, feature_num) / "feature.patch"
    if not golden_patch.exists():
        raise FileNotFoundError(f"Golden feature patch file {golden_patch} not found.")
    return golden_patch


def get_base_repo_path(task_folder: Path) -> Path | None:
    """Identify candidate base repo within the given task folder.

    Args:
        task_folder: Path to the task folder

    Returns:
        Candidate base repo folder or None if not found
    """
    repo_candidates = []
    for subfolder in task_folder.iterdir():
        if subfolder.is_dir() and (subfolder / ".git").exists():
            repo_candidates.append(subfolder)

    if len(repo_candidates) == 0:
        return None

    if len(repo_candidates) > 1:
        raise RuntimeError(f"Multiple git repositories found in {task_folder}, please specify one.")

    return repo_candidates[0]


def get_agent_workspace_path(
    setting: BenchSetting,
    task_folder_path: Path,
    repo_name: str,
    k: int,
    feature1_id: int,
    feature2_id: int | None = None,
) -> Path:
    """Get the agent workspace path for a given task and feature.
    
    We have a separate workspace per feature combination to avoid conflicts.
    """
    if setting == BenchSetting.SINGLE:
        return task_folder_path / "agent_workspace" / f"{repo_name}_feature{feature1_id}_k{k}"
    
    assert feature2_id is not None, "feature2_id must be provided for non-single settings"
    return task_folder_path / "agent_workspace" / f"{repo_name}_feature{feature1_id}_feature{feature2_id}_k{k}"


def get_container_name(
    setting: BenchSetting,
    repo_name: str,
    task_id: int,
    k: int,
    feature1_id: int,
    feature2_id: int | None = None,
) -> str:
    """Generate unique docker container name per feature combination."""
    if setting == BenchSetting.SINGLE:
        return f"openhands-app-{repo_name}-{task_id}-feature{feature1_id}_k{k}"
    
    assert feature2_id is not None, "feature2_id must be provided for non-single settings"
    return f"openhands-app-{repo_name}-{task_id}-feature{feature1_id}_feature{feature2_id}_k{k}"


def get_branch_name(
    setting: BenchSetting,
    k: int,
    feature1_id: int,
    feature2_id: int | None = None,
) -> str:
    """Create unique branch names for git worktrees.
    
    For multi-feature modes we include both feature_ids to avoid conflicts.
    """
    if setting == BenchSetting.SINGLE:
        return f"feature{feature1_id}_k{k}"
    
    assert feature2_id is not None, "feature2_id must be provided for non-single settings"
    return f"feature{feature1_id}_feature{feature2_id}_k{k}"


def _get_default_dir_structure(
    setting: BenchSetting,
    repo_name: str,
    task_id: int,
    feature1_id: int,
    feature2_id: int | None = None,
) -> str:
    """Generate a standardized directory structure based on the experiment setting.
    
    Structure:
    - single: single/<repo_name>/task<task_id>/feature<feature1_id>/
    - Multi-feature: <setting>/<repo_name>/task<task_id>/feature<i>_feature<j>/
    """
    if setting == BenchSetting.SINGLE:
        return f"{setting.value}/{repo_name}/task{task_id}/feature{feature1_id}/"
    
    assert feature2_id is not None, "feature2_id must be provided for non-single settings"
    i, j = sorted((feature1_id, feature2_id))
    return f"{setting.value}/{repo_name}/task{task_id}/feature{i}_feature{j}/"


def _get_path_for_file(
    file_name: str,
    setting: BenchSetting,
    project: str,
    task_id: int,
    feature1_id: int,
    feature2_id: int | None = None,
) -> str:
    """Generate a standardized path for a given file based on the experiment setting."""
    dir_structure = _get_default_dir_structure(setting, project, task_id, feature1_id, feature2_id)
    return f"{dir_structure}{file_name}"


def _get_paths_for_files(
    files: dict[str, str],
    setting: BenchSetting,
    repo_name: str,
    task_id: int,
    feature1_id: int,
    feature2_id: int | None = None,
) -> dict[str, Path]:
    """Convert a dictionary of file keys and names to full file paths."""
    return {
        key: Path(_get_path_for_file(name, setting, repo_name, task_id, feature1_id, feature2_id))
        for key, name in files.items()
    }


def clean_model_name(model: str) -> str:
    """Normalize model name for file naming."""
    model_lower = model.lower()
    if "minimax" in model_lower:
        return "minimax"
    if "claude" in model_lower:
        return "claude"
    if "gemini" in model_lower:
        return "gemini"
    if "qwen" in model_lower:
        if "coder" in model_lower:
            return "qwen_coder"
        return "qwen"
    return "gpt5"


def get_file_paths(
    setting: BenchSetting,
    repo_name: str,
    task_id: int,
    k: int,
    feature1_id: int,
    model1: str,
    feature2_id: int | None = None,
    model2: str | None = None,
) -> dict[str, Path]:
    """Generate all output and eval file paths for a task given a setting.
    
    Returns HF paths (without logs/ or .cache prefix).
    """
    model1_clean = clean_model_name(model1)
    
    if setting == BenchSetting.SINGLE:
        files = {
            "plan1": f"plan_{model1_clean}_k{k}_feature{feature1_id}.md",
            "planning_traj": f"planning_traj_{model1_clean}_k{k}_feature{feature1_id}.json",
            "patch1": f"patch_{model1_clean}_k{k}_feature{feature1_id}.patch",
            "execution_traj1": f"execution_traj_{model1_clean}_k{k}_feature{feature1_id}.json",
            "test_eval_result": f"test_result_{model1_clean}_{repo_name}_task{task_id}_f{feature1_id}_k{k}.txt",
        }
    elif setting == BenchSetting.SOLO:
        # Solo mode: single agent creates unified plan/patch for both features
        files = {
            "plan1": f"plan_{model1_clean}_k{k}_feature{feature1_id}.md",
            "planning_traj": f"planning_traj_{model1_clean}_k{k}_feature{feature1_id}_feature{feature2_id}.json",
            "patch1": f"patch_{model1_clean}_k{k}_feature{feature1_id}_feature{feature2_id}.patch",
            "execution_traj1": f"execution_traj_{model1_clean}_k{k}_feature{feature1_id}_feature{feature2_id}.json",
            "test_eval_result": f"test_result_{model1_clean}_{repo_name}_task{task_id}_f{feature1_id}_k{k}.txt",
            "solo_test_eval_result": f"solo_test_result_{model1_clean}_{repo_name}_task{task_id}_f{feature1_id}_f{feature2_id}_k{k}.json",
        }
    elif model2:
        # Coop or coop_ablation mode: two agents
        model2_clean = clean_model_name(model2)
        files = {
            "plan1": f"plan_{model1_clean}_k{k}_feature{feature1_id}.md",
            "plan2": f"plan_{model2_clean}_k{k}_feature{feature2_id}.md",
            "planning_traj": f"planning_traj_model1{model1_clean}_model2{model2_clean}_k{k}_feature{feature1_id}_feature{feature2_id}.json",
            "patch1": f"patch_{model1_clean}_k{k}_feature{feature1_id}.patch",
            "patch2": f"patch_{model2_clean}_k{k}_feature{feature2_id}.patch",
            "execution_traj1": f"execution_traj_{model1_clean}_k{k}_feature{feature1_id}.json",
            "execution_traj2": f"execution_traj_{model2_clean}_k{k}_feature{feature2_id}.json",
            "json_merge_report": f"{model1_clean}_{repo_name}_task{task_id}_merge_report_f{feature1_id}_into_f{feature2_id}_k{k}.json",
            "diff_file": f"{model1_clean}_{repo_name}_task{task_id}_merge_diff_f{feature1_id}_into_f{feature2_id}_k{k}.diff",
            "union_diff_file": f"{model1_clean}_{repo_name}_task{task_id}_merge_diff_f{feature1_id}_into_f{feature2_id}_k{k}_union.diff",
            "llm_diff_file": f"{model1_clean}_{repo_name}_task{task_id}_merge_diff_f{feature1_id}_into_f{feature2_id}_k{k}_llm.diff",
        }
    else:
        raise ValueError("model2 must be provided for non-single settings")
    
    return _get_paths_for_files(files, setting, repo_name, task_id, feature1_id, feature2_id)


def get_aggregate_eval_file_path(
    setting: BenchSetting,
    repo_name: str,
    task_id: int,
    k: int,
    model: str,
) -> Path:
    """Get the path for the aggregate evaluation results file."""
    dir_for_file = Path(_get_default_dir_structure(setting, repo_name, task_id, 1, 2)).parent
    return dir_for_file / f"aggregated_results_{clean_model_name(model)}_k{k}.json"
