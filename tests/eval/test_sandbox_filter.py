"""Test-file stripping must work for every language, not just Python.

An agent's own test file that survives into the graded patch collides with the hidden
tests.patch, and `git apply` then rejects the WHOLE patch — failing both features for a reason
unrelated to their code. The original rule only knew Python (`_test.py`), so `metrics_test.go`
went through: that, not the merge, is what killed 3 of 4 runs on go_chi/26.
"""

import pytest

from cooperbench.eval.sandbox import _filter_test_files, _is_test_path


@pytest.mark.parametrize("path", [
    "metrics_test.go",          # the go_chi killer
    "src/foo_test.rs",
    "tests/test_ext.py",
    "src/api.test.ts",
    "src/api.spec.js",
    "java/FooTest.java",
    "__tests__/x.js",
    "tests.py",
    "pkg/testdata/golden.json",
])
def test_test_paths_are_stripped(path):
    assert _is_test_path(f"diff --git a/{path} b/{path}")


@pytest.mark.parametrize("path", [
    "mux.go",
    "src/click/core.py",
    "crates/typst/src/foundations/str.rs",
    "src/pytest_helper.py",     # contains "test" but is source
    "latest.py",                # substring only
    "contest/main.go",          # directory named contest, not test
    "protest.py",
])
def test_source_files_survive(path):
    assert not _is_test_path(f"diff --git a/{path} b/{path}")


def test_filter_drops_only_the_test_hunk():
    patch = (
        "diff --git a/mux.go b/mux.go\n--- a/mux.go\n+++ b/mux.go\n@@ -1 +1 @@\n+real change\n"
        "diff --git a/metrics_test.go b/metrics_test.go\n--- a/metrics_test.go\n"
        "+++ b/metrics_test.go\n@@ -1 +1 @@\n+agent's own test\n"
    )
    out = _filter_test_files(patch)
    assert "real change" in out
    assert "agent's own test" not in out
    assert "metrics_test.go" not in out
