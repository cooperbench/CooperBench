"""
File interface for managing experiment files and paths.

This module provides the main FileInterface class that coordinates all file
operations during CooperBench experiments, including setup, loading, saving,
and uploading to HuggingFace.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from cooperbench.core.git import (
    get_base_commit,
    make_dirs,
    setup_agent_workspace,
    setup_base_repo,
)
from cooperbench.core.huggingface import download_file_from_hf, upload_file_to_hf
from cooperbench.core.merge import merge
from cooperbench.core.patch import apply_patch
from cooperbench.core.paths import (
    _get_default_dir_structure,
    get_agent_workspace_path,
    get_aggregate_eval_file_path,
    get_branch_name,
    get_container_name,
    get_feature_file_path,
    get_file_paths,
    get_golden_feature_patch,
    get_log_path,
    get_task_folder_path,
    get_test_script_path,
    get_tests_patch_path,
)
from cooperbench.core.settings import BenchSetting


class FileInterface:
    """Main file interface for managing experiment files and state.

    Handles all loading and saving of files, and stores task details.
    Supports different experiment settings: single, solo, coop, coop_ablation.
    """

    def __init__(
        self,
        setting: BenchSetting,
        repo_name: str,
        task_id: int,
        k: int,
        feature1_id: int,
        model1: str,
        feature2_id: int | None = None,
        model2: str | None = None,
        save_to_hf: bool = False,
        create_pr: bool = False,
        defer_uploads: bool = False,
    ) -> None:
        """Initialize the file interface.

        Args:
            setting: Experiment setting (single, solo, coop, coop_ablation)
            repo_name: Repository name
            task_id: Task number
            k: Experiment run identifier
            feature1_id: First feature number
            model1: Model for first agent or single agent
            feature2_id: Second feature number (required for non-single settings)
            model2: Model for second agent (coop only, defaults to model1)
            save_to_hf: Whether to save experiment results to HuggingFace
            create_pr: Whether to create a PR when saving to HF
            defer_uploads: Whether to defer uploads until flush_pending_uploads
        """
        self.repo_name = repo_name
        self.task_id = task_id
        self.model1 = model1
        self.model2 = model2
        self.create_pr = create_pr
        self.setting = setting
        self.feature1_id = feature1_id
        self.feature2_id = feature2_id
        self.k = k
        self.save_to_hf = save_to_hf
        self.defer_uploads = defer_uploads
        self._pending_uploads: set[Path] = set()

        self.task_folder_path = get_task_folder_path(repo_name, task_id)

        self.file_paths = get_file_paths(
            setting=setting,
            repo_name=repo_name,
            task_id=task_id,
            k=k,
            feature1_id=feature1_id,
            model1=model1,
            feature2_id=feature2_id,
            model2=model2,
        )

        assert feature1_id != feature2_id, (
            f"feature1_id and feature2_id cannot be the same: {feature1_id} and {feature2_id}"
        )
        assert setting == BenchSetting.SINGLE or feature2_id, "feature2_id should be set for non-single settings"

        self.base_repo_path: Path
        self.base_commit: str
        self.agent_workspace1_path: Path
        self.agent_workspace2_path: Path
        self.has_conflicting_golden_features: bool = False

    def copy_with_params(
        self,
        setting: BenchSetting | None = None,
        repo_name: str | None = None,
        task_id: int | None = None,
        k: int | None = None,
        feature1_id: int | None = None,
        model1: str | None = None,
        feature2_id: int | None = None,
        model2: str | None = None,
        save_to_hf: bool | None = None,
        create_pr: bool | None = None,
        defer_uploads: bool | None = None,
    ) -> "FileInterface":
        """Create a copy of this FileInterface with potentially new parameters."""
        return FileInterface(
            setting=setting if setting is not None else self.setting,
            repo_name=repo_name if repo_name is not None else self.repo_name,
            task_id=task_id if task_id is not None else self.task_id,
            k=k if k is not None else self.k,
            feature1_id=feature1_id if feature1_id is not None else self.feature1_id,
            model1=model1 if model1 is not None else self.model1,
            feature2_id=feature2_id if feature2_id is not None else self.feature2_id,
            model2=model2 if model2 is not None else self.model2,
            save_to_hf=save_to_hf if save_to_hf is not None else self.save_to_hf,
            create_pr=create_pr if create_pr is not None else self.create_pr,
            defer_uploads=defer_uploads if defer_uploads is not None else self.defer_uploads,
        )

    def setup_filesystem(
        self,
        setup_both_agent_workspaces: bool = True,
        check_for_conflicts: bool = True,
    ) -> None:
        """Create new agent workspaces and initialize the repository.

        Args:
            setup_both_agent_workspaces: Also create workspace for second feature (for eval)
            check_for_conflicts: Check if golden features have merge conflicts
        """
        make_dirs(self.file_paths["patch1"])
        self.base_repo_path = setup_base_repo(self.task_folder_path)
        self.base_commit = get_base_commit(self.base_repo_path)

        self.agent_workspace1_path = setup_agent_workspace(
            self.base_repo_path,
            self.base_commit,
            get_branch_name(self.setting, self.k, self.feature1_id, self.feature2_id),
            get_agent_workspace_path(
                self.setting,
                self.task_folder_path,
                self.repo_name,
                self.k,
                self.feature1_id,
                self.feature2_id,
            ),
        )

        if setup_both_agent_workspaces and self.feature2_id:
            self.agent_workspace2_path = setup_agent_workspace(
                self.base_repo_path,
                self.base_commit,
                get_branch_name(self.setting, self.k, self.feature2_id, self.feature1_id),
                get_agent_workspace_path(
                    self.setting,
                    self.task_folder_path,
                    self.repo_name,
                    self.k,
                    self.feature2_id,
                    self.feature1_id,
                ),
            )
            if check_for_conflicts:
                self._check_for_conflict()

    def _check_for_conflict(self) -> None:
        """Check if the two golden features have a conflict."""
        self.apply_patch(file_location="golden", first=True)
        self.apply_patch(file_location="golden", first=False)
        _, conflict = merge(
            self.agent_workspace2_path,
            get_branch_name(self.setting, self.k, self.feature1_id, self.feature2_id),
        )
        self.has_conflicting_golden_features = conflict
        if conflict:
            self.setup_filesystem(setup_both_agent_workspaces=True, check_for_conflicts=False)

    def get_log_file_path(self, key: str) -> Path:
        """Get the log path for a file by key."""
        return get_log_path(self.file_paths[key])

    def _get_local_file_path(self, file_path: Path, location: str | None = None) -> Path:
        if location:
            file_path = download_file_from_hf(file_path, location=location)
        return file_path

    def _get_file_content(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")

    def _get_json_file_content(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    def _save_file(self, file_path: Path, content: str) -> None:
        get_log_path(file_path).write_text(content, encoding="utf-8")

    def _save_json_file(self, file_path: Path, content: dict[str, Any]) -> None:
        with open(get_log_path(file_path), "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

    def _queue_or_upload(self, file_path: Path) -> None:
        if not self.save_to_hf:
            return
        if self.defer_uploads:
            self._pending_uploads.add(file_path)
        else:
            upload_file_to_hf(file_path, self.create_pr)

    def flush_pending_uploads(self) -> None:
        """Upload any queued files to HF in a single batch."""
        if not self.save_to_hf or not self._pending_uploads:
            return
        for path in sorted(self._pending_uploads, key=lambda p: str(p)):
            upload_file_to_hf(path, self.create_pr)
        self._pending_uploads.clear()

    def _save_and_upload_file(self, file_path: Path, content: str | dict[str, Any], is_json: bool = False) -> None:
        if is_json:
            assert isinstance(content, dict)
            self._save_json_file(file_path, content)
        else:
            assert isinstance(content, str)
            self._save_file(file_path, content)
        self._queue_or_upload(file_path)

    def _get_feature_id(self, first: bool) -> int:
        if first:
            return self.feature1_id
        assert self.feature2_id, "feature2_id required for second feature"
        return self.feature2_id

    def apply_patch(
        self,
        file_location: Literal["logs", "cache", "hf", "golden"],
        first: bool = True,
        filter_patch: bool = True,
    ) -> None:
        """Apply a patch to the agent workspace."""
        if file_location == "golden":
            local_patch_path = self.get_golden_feature_patch(first=first)
        else:
            local_patch_path = self.get_patch(file_location, first=first)
        apply_patch(
            agent_workspace_path=self.agent_workspace1_path if first else self.agent_workspace2_path,
            local_patch_path=local_patch_path,
            filter_patch=filter_patch,
        )

    # INPUT FILES

    def get_feature_description(self, first: bool = True) -> str:
        """Get the content of the feature.md for the task."""
        return self._get_file_content(get_feature_file_path(self.task_folder_path, self._get_feature_id(first)))

    def get_tests_patch_path(self, first: bool) -> Path:
        """Get the path to the tests patch for the task."""
        return get_tests_patch_path(self.task_folder_path, self._get_feature_id(first))

    def get_golden_feature_patch(self, first: bool = True) -> Path:
        """Get the path to the golden feature patch file."""
        return get_golden_feature_patch(self.task_folder_path, self._get_feature_id(first))

    def get_test_script_path(self) -> Path:
        """Get the path to the run test shell script."""
        return get_test_script_path(self.task_folder_path)

    # PLANNING FILES

    def get_plan(self, plan_location: str | None = None, first: bool = True) -> str:
        """Get the plan for the task."""
        plan_key = "plan1" if first else "plan2"
        local_plan_path = self._get_local_file_path(self.file_paths[plan_key], plan_location)
        return self._get_file_content(local_plan_path)

    def save_plan(self, content: str, first: bool = True) -> None:
        """Save the plan content to the plan file."""
        plan_id = "plan1" if first else "plan2"
        content = "## Plan\n\n" + content
        self._save_and_upload_file(self.file_paths[plan_id], content)

    def save_planning_trajectory(self, content: dict[str, Any]) -> None:
        """Save the planning trajectory content."""
        self._save_and_upload_file(self.file_paths["planning_traj"], content, is_json=True)

    # EXECUTION FILES

    def get_patch(self, patch_location: str | None = None, first: bool = True) -> Path:
        """Get the path to the agent patch file."""
        return self._get_local_file_path(self.file_paths["patch1" if first else "patch2"], patch_location)

    def save_patch(self, diff_content: str, first: bool = True) -> None:
        """Save the patch content. Skips saving if patch is empty."""
        if not diff_content or not diff_content.strip():
            return
        patch_key = "patch1" if first else "patch2"
        self._save_and_upload_file(self.file_paths[patch_key], diff_content)

    def save_execution_trajectory(self) -> None:
        """Save the execution trajectory content."""
        if self.save_to_hf:
            self._queue_or_upload(self.file_paths["execution_traj1"])

    def save_coop_execution_files(
        self,
        execution_traj1_path: Path | None = None,
        execution_traj2_path: Path | None = None,
        conversation_json_path: Path | None = None,
        agent_log_paths: list[Path] | None = None,
    ) -> None:
        """Save and upload coop execution files (trajectories, patches, and conversation JSON)."""
        if self.save_to_hf:
            if execution_traj1_path and execution_traj1_path.exists() and "execution_traj1" in self.file_paths:
                self._queue_or_upload(self.file_paths["execution_traj1"])

            if execution_traj2_path and execution_traj2_path.exists() and "execution_traj2" in self.file_paths:
                self._queue_or_upload(self.file_paths["execution_traj2"])

            if "patch1" in self.file_paths:
                patch1_log_path = get_log_path(self.file_paths["patch1"])
                if patch1_log_path.exists():
                    self._queue_or_upload(self.file_paths["patch1"])

            if "patch2" in self.file_paths:
                patch2_log_path = get_log_path(self.file_paths["patch2"])
                if patch2_log_path.exists():
                    self._queue_or_upload(self.file_paths["patch2"])

            if conversation_json_path and conversation_json_path.exists():
                dir_structure = _get_default_dir_structure(
                    self.setting, self.repo_name, self.task_id, self.feature1_id, self.feature2_id
                )
                conversation_filename = conversation_json_path.name
                conversation_hf_path = Path(f"{dir_structure}{conversation_filename}")
                self._queue_or_upload(conversation_hf_path)

            if agent_log_paths:
                dir_structure = _get_default_dir_structure(
                    self.setting, self.repo_name, self.task_id, self.feature1_id, self.feature2_id
                )
                for log_path in agent_log_paths:
                    if log_path and log_path.exists():
                        log_filename = log_path.name
                        log_hf_path = Path(f"{dir_structure}{log_filename}")
                        self._queue_or_upload(log_hf_path)

    def get_container_name(self) -> str:
        """Get the name of the docker container for the execution agent."""
        return get_container_name(
            self.setting,
            self.repo_name,
            self.task_id,
            self.k,
            self.feature1_id,
            self.feature2_id,
        )

    # EVAL FILES

    def save_json_merge_report(self, report_content: dict[str, Any]) -> None:
        """Save the JSON merge report content."""
        self._save_and_upload_file(self.file_paths["json_merge_report"], report_content, is_json=True)

    def save_merge_diff_file(self, diff_content: str, merge_type: str) -> None:
        """Save the merge diff file content."""
        if merge_type == "naive":
            self._save_and_upload_file(self.file_paths["diff_file"], diff_content)
        elif merge_type == "union":
            self._save_and_upload_file(self.file_paths["union_diff_file"], diff_content)
        elif merge_type == "llm":
            self._save_and_upload_file(self.file_paths["llm_diff_file"], diff_content)

    def get_merge_diff_path(self, merge_type: str, file_location: str) -> Path:
        """Get the merge diff file path."""
        if merge_type == "union":
            file_key = "union_diff_file"
        elif merge_type == "llm":
            file_key = "llm_diff_file"
        else:
            file_key = "diff_file"
        return self._get_local_file_path(self.file_paths[file_key], file_location)

    def save_test_report(self, report_content: dict[str, Any]) -> None:
        """Save the test report content."""
        self._save_and_upload_file(self.file_paths["test_eval_result"], report_content, is_json=True)

    def get_test_report(self, location: str | None = None) -> dict[str, Any]:
        """Get the test report content."""
        file_path = self._get_local_file_path(self.file_paths["test_eval_result"], location)
        return self._get_json_file_content(file_path)

    def save_solo_test_report(self, report_content: dict[str, Any]) -> None:
        """Save the solo test report content."""
        self._save_and_upload_file(
            self.file_paths["solo_test_eval_result"],
            report_content,
            is_json=True,
        )

    def get_solo_test_report(self, location: str | None = None) -> dict[str, Any]:
        """Get the solo test report content."""
        file_path = self._get_local_file_path(self.file_paths["solo_test_eval_result"], location)
        return self._get_json_file_content(file_path)

    def get_json_merge_report(self, location: str | None = None) -> dict[str, Any]:
        """Get the JSON merge report content."""
        file_path = self._get_local_file_path(self.file_paths["json_merge_report"], location)
        return self._get_json_file_content(file_path)

    def save_aggregate_eval_file(self, results: dict[str, Any]) -> None:
        """Save the aggregate evaluation results."""
        aggregate_file_path = get_aggregate_eval_file_path(
            self.setting, self.repo_name, self.task_id, self.k, self.model1
        )
        self._save_and_upload_file(aggregate_file_path, results, is_json=True)

    # REPORT METADATA

    def get_merge_report_metadata(self) -> dict[str, Any]:
        """Generate metadata for the merge report."""
        return {
            "repo_name": self.repo_name,
            "task_id": self.task_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature1": {
                "number": self.feature1_id,
                "tests_passed": False,
                "test_output": None,
            },
            "feature2": {
                "number": self.feature2_id,
                "tests_passed": False,
                "test_output": None,
            },
            "merge_status": "clean",
            "conflict_score": 0,
            "conflict_details": {
                "conflict_sections": 0,
                "conflict_lines": 0,
                "avg_lines_per_conflict": 0,
            },
        }
