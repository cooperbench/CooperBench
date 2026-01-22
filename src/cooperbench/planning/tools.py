"""
Planning tools for CooperBench agents.

This module defines the tool interfaces and implementations that agents use
during the planning phase to explore codebases and coordinate on implementation plans.
"""

import glob
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ToolResult:
    """Result of a tool execution."""

    def __init__(self, success: bool, content: str = "", error: str = "") -> None:
        self.success = success
        self.content = content
        self.error = error


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self) -> None:
        self.name = self.get_name()
        self.description = self.get_description()
        self.parameters = self.get_parameters()

    @abstractmethod
    def get_name(self) -> str:
        """Return the tool name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return the tool description."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return the tool parameters schema."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def get_schema(self) -> dict[str, Any]:
        """Return the OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def validate_path(path: str, agent_workspace_path: Path) -> Path:
    """Validate and resolve a path within the agent workspace."""
    target_path = (agent_workspace_path / path).resolve()
    try:
        target_path.relative_to(agent_workspace_path)
    except ValueError:
        raise ValueError(f"Path {path} is outside agent workspace directory")
    return target_path


def get_file_with_line_numbers(file_path: Path, start_line: int | None = None, end_line: int | None = None) -> str:
    """Read file content with line numbers."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")

        start_idx = max(0, start_line - 1) if start_line else 0
        end_idx = min(len(lines), end_line) if end_line else len(lines)

        result_lines = []
        for i, line in enumerate(lines[start_idx:end_idx], start=start_idx + 1):
            result_lines.append(f"{i:4d} {line}")

        return "\n".join(result_lines)
    except Exception as e:
        return f"Error reading file: {str(e)}"


class ListFilesTool(BaseTool):
    """Tool to list files and directories."""

    def get_name(self) -> str:
        return "list_files"

    def get_description(self) -> str:
        return "List files and directories (including hidden files) in the agent workspace, similar to 'ls -a'."

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to list (defaults to current directory)",
                },
            },
            "required": ["explanation"],
        }

    async def execute(
        self,
        explanation: str,
        agent_workspace_path: Path,
        path: str = ".",
        **kwargs: object,
    ) -> ToolResult:
        try:
            dir_path = validate_path(path, agent_workspace_path)
            if not dir_path.exists():
                return ToolResult(success=False, error=f"Directory {path} not found")

            items = []
            all_items = list(dir_path.iterdir())
            for item in sorted(all_items):
                item_name = item.name
                if item.is_dir():
                    items.append(f"{item_name}/")
                else:
                    items.append(item_name)

            if not items:
                return ToolResult(success=True, content="Directory is empty")

            result_content = f"Found {len(items)} items:\n\n" + "\n".join(items)
            return ToolResult(success=True, content=result_content)
        except Exception as e:
            return ToolResult(success=False, error=f"ListFiles error: {str(e)}")


class ReadFileTool(BaseTool):
    """Tool to read content from a file."""

    def get_name(self) -> str:
        return "read_file"

    def get_description(self) -> str:
        return "Read file content with line numbers. By default reads first 100 lines."

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used.",
                },
                "filename": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, optional).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (1-indexed, optional).",
                },
            },
            "required": ["explanation", "filename"],
        }

    async def execute(
        self,
        explanation: str,
        filename: str,
        agent_workspace_path: Path,
        start_line: int | None = None,
        end_line: int | None = None,
        **kwargs: object,
    ) -> ToolResult:
        try:
            file_path = validate_path(filename, agent_workspace_path)
            if not file_path.exists():
                return ToolResult(success=False, error=f"File {filename} not found")

            content = file_path.read_text(encoding="utf-8", errors="ignore")
            total_lines = len(content.split("\n"))

            if start_line is None and end_line is None:
                end_line = min(100, total_lines)

            numbered_content = get_file_with_line_numbers(file_path, start_line, end_line)

            if start_line is None and end_line is None:
                header = f"File: {filename} (showing first {min(100, total_lines)} of {total_lines} lines)\n\n"
            elif start_line is not None and end_line is not None:
                header = f"File: {filename} (lines {start_line}-{end_line} of {total_lines} total)\n\n"
            elif start_line is not None:
                header = f"File: {filename} (from line {start_line} of {total_lines} total)\n\n"
            else:
                header = f"File: {filename} (first {end_line} lines of {total_lines} total)\n\n"

            return ToolResult(success=True, content=header + numbered_content)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GrepSearchTool(BaseTool):
    """Tool for fast regex search using grep functionality."""

    def get_name(self) -> str:
        return "grep_search"

    def get_description(self) -> str:
        return "Search for text patterns in files using regex. Results limited to 20 matches."

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "One sentence explanation as to why this tool is being used.",
                },
                "query": {
                    "type": "string",
                    "description": "Text pattern or regex to search for",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to search (e.g., '*.py', '*.js')",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive (default: false)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of lines to show before and after each match (default: 2)",
                },
            },
            "required": ["explanation", "query"],
        }

    async def execute(
        self,
        explanation: str,
        query: str,
        agent_workspace_path: Path,
        file_pattern: str = "*",
        case_sensitive: bool = False,
        context_lines: int = 2,
        **kwargs: object,
    ) -> ToolResult:
        try:
            search_pattern = str(agent_workspace_path / "**" / file_pattern)
            files = glob.glob(search_pattern, recursive=True)

            text_files = [
                Path(file_path)
                for file_path in files
                if Path(file_path).is_file() and self._is_text_file(Path(file_path))
            ]

            if not text_files:
                return ToolResult(
                    success=True,
                    content=f"No files found matching pattern: {file_pattern}",
                )

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(query, flags)
            except re.error as e:
                return ToolResult(success=False, error=f"Invalid regex pattern: {e}")

            matches = []
            match_count = 0

            for file_path in text_files:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.split("\n")
                    file_matches = []

                    for line_num, line in enumerate(lines, 1):
                        if pattern.search(line):
                            start_idx = max(0, line_num - 1 - context_lines)
                            end_idx = min(len(lines), line_num + context_lines)

                            context_block = []
                            for i in range(start_idx, end_idx):
                                line_number = i + 1
                                line_content = lines[i]
                                if len(line_content) > 500:
                                    line_content = line_content[:500] + "... [truncated]"
                                if line_number == line_num:
                                    context_block.append(f"{line_number:4d} {line_content}  # <-- MATCH")
                                else:
                                    context_block.append(f"{line_number:4d} {line_content}")

                            file_matches.append("\n".join(context_block))
                            match_count += 1

                            if match_count >= 20:
                                break

                    if file_matches:
                        relative_path = file_path.relative_to(agent_workspace_path)
                        matches.append(f"{relative_path}:\n" + "\n\n".join(file_matches))

                    if match_count >= 20:
                        break

                except Exception:
                    continue

            if not matches:
                return ToolResult(
                    success=True,
                    content=f"No matches found for pattern: {query}",
                )

            result_header = f"Found {min(match_count, 20)} matches for '{query}':\n\n"
            if match_count >= 20:
                result_header += "... (results truncated at 20 matches)\n\n"

            result = result_header + "\n\n".join(matches)
            return ToolResult(success=True, content=result)

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is likely a text file."""
        binary_extensions = {
            ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip",
            ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc",
        }

        if file_path.suffix.lower() in binary_extensions:
            return False

        try:
            if file_path.stat().st_size > 1_000_000:
                return False
        except (OSError, ValueError):
            return False

        return True


class AgreementTool(BaseTool):
    """Tool for agents to signal they have reached an agreement on implementation plans."""

    def get_name(self) -> str:
        return "agreement_reached"

    def get_description(self) -> str:
        return """Provide an extremely detailed implementation plan that a weaker model can execute step-by-step without guesswork.

        REQUIRED FORMAT:

        # FEATURE IMPLEMENTATION PLAN
        ## CONTEXT: Problem, Goal, Requirements
        ## TECHNICAL DETAILS: Architecture, Critical Notes
        ## STEP-BY-STEP IMPLEMENTATION (Max 8 steps)

        Example steps:

        ### Step 1: Add authentication middleware
        File: `src/app.py`
        Location: Line 15 (after imports)
        Action: Add
        Before Code: [none - new addition]
        After Code:
        ```
        def authenticate_user(request):
            return validate_token(request.headers.get('auth'))
        ```
        Critical Details: Leave line 16-17 blank, add import at top

        Must include exact file paths, line numbers, complete code blocks."""

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "Complete implementation plan with exact file paths, line numbers, and code blocks.",
                }
            },
            "required": ["plan"],
        }

    async def execute(self, plan: str, **kwargs: object) -> ToolResult:
        if not plan or not plan.strip():
            return ToolResult(
                success=False,
                error="Missing required parameter 'plan'. You must provide a complete implementation plan.",
            )
        return ToolResult(success=True, content=plan)


class CommunicateTool(BaseTool):
    """Tool for agents to communicate intermediate thoughts and partial plans to each other."""

    def get_name(self) -> str:
        return "communicate_with_agent"

    def get_description(self) -> str:
        return """Communicate with the other agent during the planning process.

        Use this to share thoughts, ask questions, propose approaches, raise concerns, or reach agreement.
        Include specific details like file paths, line numbers, and code structure in your message content."""

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Your message to the other agent",
                },
                "message_type": {
                    "type": "string",
                    "enum": ["analysis", "proposal", "question", "concern", "agreement"],
                    "description": "Type of communication",
                },
            },
            "required": ["message", "message_type"],
        }

    async def execute(self, message: str, message_type: str, **kwargs: object) -> ToolResult:
        formatted_message = f"[{message_type.upper()}] {message}"
        return ToolResult(success=True, content=formatted_message)


class DualAgreementTool(BaseTool):
    """Tool for a single agent to provide implementation plans for both features (solo mode)."""

    def get_name(self) -> str:
        return "dual_agreement_reached"

    def get_description(self) -> str:
        return """Provide extremely detailed implementation plans for BOTH features.
        
        Follow the same format as agreement_reached but provide separate plans for each feature."""

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan1": {
                    "type": "string",
                    "description": "Complete implementation plan for feature 1",
                },
                "plan2": {
                    "type": "string",
                    "description": "Complete implementation plan for feature 2",
                },
            },
            "required": ["plan1", "plan2"],
        }

    async def execute(self, plan1: str, plan2: str, **kwargs: object) -> ToolResult:
        if not plan1 or not plan1.strip():
            return ToolResult(
                success=False,
                error="Missing required parameter 'plan1'.",
            )

        if not plan2 or not plan2.strip():
            return ToolResult(
                success=False,
                error="Missing required parameter 'plan2'.",
            )

        return ToolResult(success=True, content=json.dumps({"plan1": plan1, "plan2": plan2}))
