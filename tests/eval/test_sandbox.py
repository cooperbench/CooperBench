"""Unit tests for cooperbench.eval.sandbox module.

These are pure function tests that don't require Modal.
For integration tests, see tests/integration/eval/test_sandbox.py
"""

import tempfile
from pathlib import Path

from cooperbench.eval.sandbox import (
    _error_result,
    _filter_test_files,
    _load_patch,
    _merged_error_result,
    _parse_results,
    _sanitize_patch,
    _solo_error_result,
)


class TestParseResults:
    """Tests for _parse_results output parsing."""

    def test_parse_pytest_passed_only(self):
        """Test parsing pytest output with only passed tests."""
        output = "===== 5 passed in 1.23s ====="
        result = _parse_results(output)
        assert result["passed"] == 5
        assert result["failed"] == 0

    def test_parse_pytest_mixed(self):
        """Test parsing pytest output with passed and failed."""
        output = "===== 3 passed, 2 failed in 1.23s ====="
        result = _parse_results(output)
        assert result["passed"] == 3
        assert result["failed"] == 2

    def test_parse_pytest_with_errors(self):
        """Test parsing pytest output with errors."""
        output = "===== 1 passed, 1 failed, 2 error in 1.23s ====="
        result = _parse_results(output)
        assert result["passed"] == 1
        assert result["failed"] == 3  # 1 failed + 2 errors

    def test_parse_go_test_passed(self):
        """Test parsing go test output with passes."""
        output = """
--- PASS: TestFoo (0.00s)
--- PASS: TestBar (0.01s)
--- PASS: TestBaz (0.00s)
PASS
"""
        result = _parse_results(output)
        assert result["passed"] == 3
        assert result["failed"] == 0

    def test_parse_go_test_mixed(self):
        """Test parsing go test output with mixed results."""
        output = """
--- PASS: TestFoo (0.00s)
--- FAIL: TestBar (0.01s)
--- PASS: TestBaz (0.00s)
FAIL
"""
        result = _parse_results(output)
        assert result["passed"] == 2
        assert result["failed"] == 1

    def test_parse_go_test_non_verbose_passed(self):
        """Test parsing go test non-verbose output with all packages passing."""
        output = """
ok  	github.com/go-chi/chi/v5	0.022s
ok  	github.com/go-chi/chi/v5/middleware	0.011s [no tests to run]
"""
        result = _parse_results(output)
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_parse_go_test_non_verbose_failed(self):
        """Test parsing go test non-verbose output with build failures."""
        output = """
FAIL	github.com/go-chi/chi/v5 [build failed]
FAIL	github.com/go-chi/chi/v5/middleware [build failed]
FAIL
"""
        result = _parse_results(output)
        assert result["passed"] == 0
        assert result["failed"] == 2

    def test_parse_go_test_non_verbose_mixed(self):
        """Test parsing go test non-verbose output with mixed results."""
        output = """
ok  	github.com/go-chi/chi/v5	0.022s
FAIL	github.com/go-chi/chi/v5/middleware [build failed]
"""
        result = _parse_results(output)
        assert result["passed"] == 0  # Any failure means overall failure
        assert result["failed"] == 1

    def test_parse_cargo_test(self):
        """Test parsing cargo test output."""
        output = """
running 5 tests
test tests::test_foo ... ok
test tests::test_bar ... ok
test tests::test_baz ... FAILED
test tests::test_qux ... ok
test tests::test_quux ... ok

test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured
"""
        result = _parse_results(output)
        assert result["passed"] == 4
        assert result["failed"] == 1

    def test_parse_jest_passed_only(self):
        """Test parsing jest output with only passed tests."""
        output = """
PASS  src/__tests__/useForm/handleSubmit.test.tsx
  handleSubmit
    ✓ should handle form submission (15 ms)
    ✓ should validate fields (8 ms)

Test Suites: 1 passed, 1 total
Tests:       15 passed, 15 total
Snapshots:   0 total
Time:        2.345 s
"""
        result = _parse_results(output)
        assert result["passed"] == 15
        assert result["failed"] == 0

    def test_parse_jest_mixed(self):
        """Test parsing jest output with passed and failed tests."""
        output = """
FAIL  src/__tests__/useForm/handleSubmit.test.tsx
  handleSubmit
    ✓ should handle form submission (15 ms)
    ✕ should validate fields (8 ms)

Test Suites: 1 failed, 1 total
Tests:       2 failed, 15 passed, 17 total
Snapshots:   0 total
Time:        2.345 s
"""
        result = _parse_results(output)
        assert result["passed"] == 15
        assert result["failed"] == 2

    def test_parse_empty_output(self):
        """Test parsing empty output."""
        result = _parse_results("")
        assert result["passed"] == 0
        assert result["failed"] == 0

    def test_parse_no_test_output(self):
        """Test parsing output with no test results."""
        output = "Just some random log output\nNothing about tests"
        result = _parse_results(output)
        assert result["passed"] == 0
        assert result["failed"] == 0


class TestFilterTestFiles:
    """Tests for _filter_test_files patch filtering."""

    def test_filter_removes_test_files(self):
        """Test that test files are removed from patch."""
        patch = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
 def foo():
     pass
+    return 42

diff --git a/tests/test_main.py b/tests/test_main.py
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,4 @@
 def test_foo():
     pass
+    assert True
"""
        filtered = _filter_test_files(patch)
        assert "src/main.py" in filtered
        assert "tests/test_main.py" not in filtered
        assert "return 42" in filtered
        assert "assert True" not in filtered

    def test_filter_removes_test_underscore_prefix(self):
        """Test that test_ prefix files are removed."""
        patch = """diff --git a/src/util.py b/src/util.py
--- a/src/util.py
+++ b/src/util.py
@@ -1 +1 @@
-old
+new

diff --git a/test_util.py b/test_util.py
--- a/test_util.py
+++ b/test_util.py
@@ -1 +1 @@
-test old
+test new
"""
        filtered = _filter_test_files(patch)
        assert "src/util.py" in filtered
        assert "test_util.py" not in filtered

    def test_filter_removes_test_directory(self):
        """Test that /test/ directory files are removed."""
        patch = """diff --git a/lib/core.py b/lib/core.py
--- a/lib/core.py
+++ b/lib/core.py
@@ -1 +1 @@
-a
+b

diff --git a/test/core_test.py b/test/core_test.py
--- a/test/core_test.py
+++ b/test/core_test.py
@@ -1 +1 @@
-x
+y
"""
        filtered = _filter_test_files(patch)
        assert "lib/core.py" in filtered
        assert "test/core_test.py" not in filtered

    def test_filter_removes_tests_py_suffix(self):
        """Test that _test.py suffix files are removed."""
        patch = """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1 +1 @@
-foo
+bar

diff --git a/module_test.py b/module_test.py
--- a/module_test.py
+++ b/module_test.py
@@ -1 +1 @@
-test foo
+test bar
"""
        filtered = _filter_test_files(patch)
        assert "module.py" in filtered
        assert "module_test.py" not in filtered

    def test_filter_empty_patch(self):
        """Test filtering empty patch."""
        assert _filter_test_files("") == ""
        assert _filter_test_files(None) is None

    def test_filter_preserves_non_test_files(self):
        """Test that non-test files are preserved."""
        patch = """diff --git a/src/api.py b/src/api.py
--- a/src/api.py
+++ b/src/api.py
@@ -1,5 +1,6 @@
 class API:
     def get(self):
-        return None
+        return {}
+
+    def post(self):
+        return {}
"""
        filtered = _filter_test_files(patch)
        assert filtered.strip() == patch.strip()

    def test_filter_adds_trailing_newline(self):
        """Test that filtered patch ends with newline."""
        patch = "diff --git a/f.py b/f.py\n+content"
        filtered = _filter_test_files(patch)
        assert filtered.endswith("\n")


class TestSanitizePatch:
    """Tests for _sanitize_patch."""

    def test_sanitize_fixes_shell_escaped_quotes(self):
        """Test fixing shell-escaped quotes."""
        patch = "don'\\''t break"
        sanitized = _sanitize_patch(patch)
        assert sanitized == "don't break\n"

    def test_sanitize_adds_trailing_newline(self):
        """Test adding trailing newline."""
        patch = "content without newline"
        sanitized = _sanitize_patch(patch)
        assert sanitized.endswith("\n")

    def test_sanitize_preserves_existing_newline(self):
        """Test preserving existing trailing newline."""
        patch = "content with newline\n"
        sanitized = _sanitize_patch(patch)
        assert sanitized == "content with newline\n"

    def test_sanitize_empty_string(self):
        """Test sanitizing empty string."""
        assert _sanitize_patch("") == ""


class TestLoadPatch:
    """Tests for _load_patch."""

    def test_load_none_returns_none(self):
        """Test loading None returns None."""
        assert _load_patch(None) is None

    def test_load_empty_string_returns_none(self):
        """Test loading empty string returns None."""
        assert _load_patch("") is None
        assert _load_patch("   ") is None

    def test_load_string_content(self):
        """Test loading patch from string content."""
        patch_content = "diff --git a/f.py b/f.py\n+new line"
        # Long enough to not be treated as a file path
        result = _load_patch(patch_content)
        assert "diff --git" in result

    def test_load_from_path_object(self):
        """Test loading patch from Path object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write("diff --git a/test.py b/test.py\n+test content")
            f.flush()
            path = Path(f.name)

        try:
            result = _load_patch(path)
            assert "diff --git" in result
            assert "test content" in result
        finally:
            path.unlink()

    def test_load_from_file_path_string(self):
        """Test loading patch from file path string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write("patch from file")
            f.flush()
            path = f.name

        try:
            result = _load_patch(path)
            assert "patch from file" in result
        finally:
            Path(path).unlink()

    def test_load_sanitizes_content(self):
        """Test that loaded content is sanitized."""
        patch = "won'\\''t break"
        result = _load_patch(patch * 100)  # Make it long enough
        assert "won't break" in result


class TestErrorResults:
    """Tests for error result factory functions."""

    def test_error_result_structure(self):
        """Test _error_result returns correct structure."""
        result = _error_result("Something went wrong")
        assert result["passed"] is False
        assert result["tests_passed"] == 0
        assert result["tests_failed"] == 0
        assert result["tests_total"] == 0
        assert result["output"] == ""
        assert result["error"] == "Something went wrong"

    def test_merged_error_result_structure(self):
        """Test _merged_error_result returns correct structure."""
        result = _merged_error_result("Merge failed")
        assert result["merge"]["status"] == "error"
        assert result["merge"]["strategy"] is None
        assert result["merge"]["diff"] == ""
        assert result["feature1"]["passed"] is False
        assert result["feature2"]["passed"] is False
        assert result["both_passed"] is False
        assert result["error"] == "Merge failed"

    def test_solo_error_result_structure(self):
        """Test _solo_error_result returns correct structure."""
        result = _solo_error_result("Solo agent failed")
        assert result["setting"] == "solo"
        assert result["patch_lines"] == 0
        assert result["feature1"]["passed"] is False
        assert result["feature2"]["passed"] is False
        assert result["both_passed"] is False
        assert result["error"] == "Solo agent failed"
