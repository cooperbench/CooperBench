"""Tests for CLI functionality."""

import sys
from unittest.mock import patch

import pytest

from cooperbench.cli import _generate_run_name


class TestGenerateRunName:
    """Tests for _generate_run_name function."""

    def test_basic_name_generation(self):
        """Test basic run name generation."""
        name = _generate_run_name("solo", "gpt-4o")
        assert name == "solo-gpt-4o"

    def test_with_subset(self):
        """Test name generation with subset."""
        name = _generate_run_name("solo", "gpt-4o", subset="lite")
        assert name == "solo-gpt-4o-lite"

    def test_with_repo(self):
        """Test name generation with repo filter."""
        name = _generate_run_name("coop", "gpt-4o", repo="llama_index_task")
        assert name == "coop-gpt-4o-llama-index"

    def test_with_task(self):
        """Test name generation with task filter."""
        name = _generate_run_name("solo", "gpt-4o", repo="pillow_task", task=25)
        assert name == "solo-gpt-4o-pillow-25"

    def test_with_task_zero(self):
        """Test name generation with task ID 0 (valid task ID)."""
        name = _generate_run_name("solo", "gpt-4o", repo="openai_tiktoken_task", task=0)
        assert name == "solo-gpt-4o-openai-tiktoken-0"

    def test_with_all_options(self):
        """Test name generation with all options."""
        name = _generate_run_name(
            "coop", "gemini/gemini-3-flash-preview", subset="lite", repo="pillow_task", task=25
        )
        assert name == "coop-gemini-3-flash-lite-pillow-25"

    def test_cleans_model_name(self):
        """Test that model names are cleaned."""
        name = _generate_run_name("solo", "gemini/gemini-3-flash-preview")
        assert "gemini-3-flash" in name
        assert "preview" not in name
        assert "/" not in name

    def test_cleans_repo_name(self):
        """Test that repo names are cleaned."""
        name = _generate_run_name("solo", "gpt-4o", repo="llama_index_task")
        assert "llama-index" in name
        assert "_task" not in name

    def test_different_settings(self):
        """Test coop vs solo settings."""
        solo_name = _generate_run_name("solo", "gpt-4o")
        coop_name = _generate_run_name("coop", "gpt-4o")
        assert solo_name.startswith("solo-")
        assert coop_name.startswith("coop-")


class TestCLI:
    """Tests for CLI."""

    def test_cli_module_importable(self):
        """Test that CLI module is importable."""
        from cooperbench import cli

        assert hasattr(cli, "main")

    def test_cli_help(self):
        """Test CLI help output."""
        from cooperbench.cli import main

        with patch.object(sys, "argv", ["cooperbench", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # --help should exit with 0
            assert exc_info.value.code == 0

    def test_cli_run_subcommand_exists(self):
        """Test run subcommand exists."""
        from cooperbench.cli import main

        with patch.object(sys, "argv", ["cooperbench", "run", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_cli_eval_subcommand_exists(self):
        """Test eval subcommand exists."""
        from cooperbench.cli import main

        with patch.object(sys, "argv", ["cooperbench", "eval", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
