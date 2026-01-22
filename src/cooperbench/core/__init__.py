"""
Core utilities and foundational components for CooperBench.

This subpackage contains essential modules for configuration, file management,
path conventions, patch handling, Git merge operations, workspace setup,
and HuggingFace integration.
"""

from cooperbench.core.settings import BenchSetting
from cooperbench.core.paths import (
    CACHE_DIR,
    LOGS_DIR,
    clean_model_name,
    ensure_dirs,
    get_aggregate_eval_file_path,
    get_agent_workspace_path,
    get_base_repo_path,
    get_branch_name,
    get_cache_dir,
    get_cache_path,
    get_container_name,
    get_feature_file_path,
    get_feature_folder_path,
    get_file_paths,
    get_golden_feature_patch,
    get_log_path,
    get_task_folder_path,
    get_test_script_path,
    get_tests_patch_path,
)
from cooperbench.core.logger import BenchLogger, get_logger
from cooperbench.core.git import (
    get_base_commit,
    make_dirs,
    run_git_command,
    setup_agent_workspace,
    setup_base_repo,
)
from cooperbench.core.patch import (
    apply_patch,
    categorize_files,
    count_code_lines_changed,
    generate_patch,
    parse_patch_file_paths,
    split_patch,
    split_patch_by_type,
)
from cooperbench.core.merge import (
    analyze_merge,
    analyze_merge_union,
    merge,
    merge_union,
)
from cooperbench.core.interface import FileInterface

__all__ = [
    # Interface
    "FileInterface",
    # Settings
    "BenchSetting",
    # Paths
    "CACHE_DIR",
    "LOGS_DIR",
    "clean_model_name",
    "ensure_dirs",
    "get_aggregate_eval_file_path",
    "get_agent_workspace_path",
    "get_base_repo_path",
    "get_branch_name",
    "get_cache_dir",
    "get_cache_path",
    "get_container_name",
    "get_feature_file_path",
    "get_feature_folder_path",
    "get_file_paths",
    "get_golden_feature_patch",
    "get_log_path",
    "get_task_folder_path",
    "get_test_script_path",
    "get_tests_patch_path",
    # Logger
    "BenchLogger",
    "get_logger",
    # Git
    "get_base_commit",
    "make_dirs",
    "run_git_command",
    "setup_agent_workspace",
    "setup_base_repo",
    # Patch
    "apply_patch",
    "categorize_files",
    "count_code_lines_changed",
    "generate_patch",
    "parse_patch_file_paths",
    "split_patch",
    "split_patch_by_type",
    # Merge
    "analyze_merge",
    "analyze_merge_union",
    "merge",
    "merge_union",
]
