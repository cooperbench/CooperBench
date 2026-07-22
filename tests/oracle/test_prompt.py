"""Tests for oracle prompt injection (cooperbench.oracle.prompt)."""

from pathlib import Path

import pytest

from cooperbench.oracle.prompt import OracleMode, build_oracle_task


SAMPLE_PATCH = """\
diff --git a/src/foo.py b/src/foo.py
index 1234567..abcdef0 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,5 @@
 def hello():
+    # new comment
     return "hello"
+
+NEW_CONSTANT = 42
"""

SAMPLE_FEATURE_MD = """\
**Title**: Add NEW_CONSTANT and improve hello

**Description**: Adds a module-level constant and a comment to the hello function.
"""


@pytest.fixture
def feature_dir(tmp_path):
    """Minimal feature directory with patch and feature.md."""
    d = tmp_path / "feature1"
    d.mkdir()
    (d / "feature.patch").write_text(SAMPLE_PATCH)
    (d / "feature.md").write_text(SAMPLE_FEATURE_MD)
    return d


@pytest.fixture
def feature_dir_no_patch(tmp_path):
    """Feature directory with no patch file."""
    d = tmp_path / "feature2"
    d.mkdir()
    (d / "feature.md").write_text(SAMPLE_FEATURE_MD)
    return d


class TestBuildOracleTaskPatchMode:
    def test_prepends_oracle_block(self, feature_dir):
        base = "Implement feature X."
        result = build_oracle_task(base, feature_dir, mode=OracleMode.PATCH)
        assert "Oracle Solution" in result
        assert "Implement feature X." in result

    def test_patch_content_included(self, feature_dir):
        result = build_oracle_task("task", feature_dir, mode=OracleMode.PATCH)
        assert "NEW_CONSTANT = 42" in result

    def test_oracle_block_comes_before_task(self, feature_dir):
        base = "TASK_SENTINEL"
        result = build_oracle_task(base, feature_dir, mode=OracleMode.PATCH)
        oracle_pos = result.index("Oracle Solution")
        task_pos = result.index("TASK_SENTINEL")
        assert oracle_pos < task_pos

    def test_no_patch_returns_base_task_unchanged(self, feature_dir_no_patch):
        base = "original task"
        result = build_oracle_task(base, feature_dir_no_patch, mode=OracleMode.PATCH)
        assert result == base

    def test_empty_patch_returns_base_task_unchanged(self, tmp_path):
        d = tmp_path / "feature3"
        d.mkdir()
        (d / "feature.patch").write_text("")
        base = "original task"
        result = build_oracle_task(base, d, mode=OracleMode.PATCH)
        assert result == base


class TestBuildOracleTaskIntentMode:
    def test_includes_description_and_patch(self, feature_dir):
        result = build_oracle_task("task", feature_dir, mode=OracleMode.INTENT)
        assert "NEW_CONSTANT = 42" in result
        assert "NEW_CONSTANT" in result  # description also mentions it

    def test_feature_md_content_present(self, feature_dir):
        result = build_oracle_task("task", feature_dir, mode=OracleMode.INTENT)
        assert "Add NEW_CONSTANT" in result


class TestBuildOracleTaskCodeMode:
    def test_falls_back_gracefully_when_no_base_dir(self, feature_dir):
        # Without repo_base_dir, code mode cannot apply the patch and will show a
        # fallback message rather than crashing.
        result = build_oracle_task("task", feature_dir, mode=OracleMode.CODE)
        assert "Oracle Solution" in result
        assert "task" in result

    def test_code_mode_does_not_crash_with_invalid_base_dir(self, feature_dir, tmp_path):
        # A base dir that is not a git repo → _apply_patch_to_temp returns {} → fallback message.
        result = build_oracle_task("task", feature_dir, mode=OracleMode.CODE, repo_base_dir=tmp_path)
        assert "Oracle Solution" in result


class TestOracleModeEnum:
    def test_all_values_are_valid_strings(self):
        for mode in OracleMode:
            assert isinstance(mode.value, str)

    def test_default_mode_is_patch(self):
        assert OracleMode.PATCH == OracleMode("patch")
