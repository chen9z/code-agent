from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base import BaseTool

MAX_TIMEOUT_MS = 600_000


def _set_nonblocking(pipe) -> None:
    if pipe is None:
        return
    fd = pipe.fileno()
    os.set_blocking(fd, False)


def _read_available(stream) -> str:
    if stream is None:
        return ""
    try:
        data = stream.read()
    except (BlockingIOError, ValueError):
        return ""
    if data in (None, ""):
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _split_complete_lines(buffer: str) -> tuple[List[str], str]:
    if not buffer:
        return [], ""
    lines: List[str] = []
    remainder = buffer
    while True:
        newline_index = remainder.find("\n")
        if newline_index == -1:
            break
        lines.append(remainder[:newline_index])
        remainder = remainder[newline_index + 1 :]
    return lines, remainder


@dataclass
class BackgroundShell:
    shell_id: str
    process: subprocess.Popen
    stdout_buffer: str = ""
    stderr_buffer: str = ""
    status: str = "running"

    def refresh(self) -> None:
        self.stdout_buffer += _read_available(self.process.stdout)
        self.stderr_buffer += _read_available(self.process.stderr)
        if self.process.poll() is not None and self.status == "running":
            self.status = "completed"

    def consume(self, buffer: str, pattern: str | None) -> tuple[str, str]:
        if not buffer:
            return "", ""
        if pattern is None:
            return buffer, ""
        complete_lines, remainder = _split_complete_lines(buffer)
        if not complete_lines:
            return "", remainder
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid filter regex: {exc}") from exc
        matched_lines: List[str] = []
        for line in complete_lines:
            if regex.search(line):
                matched_lines.append(line)
        output = "\n".join(matched_lines)
        if matched_lines:
            output += "\n"
        return output, remainder

    def close(self) -> None:
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()
        self.process.stdout = None
        self.process.stderr = None


BACKGROUND_SHELLS: Dict[str, BackgroundShell] = {}


class BashTool(BaseTool):
    """Executes bash commands, optionally in background."""

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return """Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use the LS tool to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use LS to check that "foo" exists and is the intended parent directory

2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command.
   - Capture the output of the command.

Usage notes:
  - The command argument is required.
  - You can specify an optional timeout in milliseconds (up to 600000ms / 10 minutes). If not specified, commands will timeout after 120000ms (2 minutes).
  - It is very helpful if you write a clear, concise description of what this command does in 5-10 words.
  - If the output exceeds 30000 characters, output will be truncated before being returned to you.
  - You can use the `run_in_background` parameter to run the command in the background, which allows you to continue working while the command runs. You can monitor the output using the BashOutput tool as it becomes available. Never use `run_in_background` to run 'sleep' as it will return immediately. You do not need to use '&' at the end of the command when using this parameter.
  - VERY IMPORTANT: You MUST avoid using search commands like `find` and `grep`. Instead use Grep, Glob, or Task to search. You MUST avoid read tools like `cat`, `head`, `tail`, and `ls`, and use Read and LS to read files.
  - If you still need to run `grep`, STOP. ALWAYS USE ripgrep at `rg` first, which all Claude Code users have pre-installed.
  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines.
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in milliseconds (max 600000)",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Clear, concise description of what this command does in 5-10 words."
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "Set to true to run this command in the background. Use BashOutput to read the output later."
                    ),
                },
            },
            "required": ["command"],
        }

    def execute(
        self,
        *,
        command: str,
        timeout: float | int | None = None,
        description: str | None = None,
        run_in_background: bool | None = None,
    ) -> Dict[str, Any]:
        if not command:
            return {"error": "command must be a non-empty string", "command": command}

        if timeout is not None:
            try:
                timeout_val = float(timeout) / 1000.0
            except (TypeError, ValueError):
                return {"error": "timeout must be a number", "command": command}
            if timeout_val <= 0 or timeout_val > MAX_TIMEOUT_MS / 1000:
                return {"error": "timeout must be between 1 and 600000 milliseconds", "command": command}
        else:
            timeout_val = 120

        run_in_background = bool(run_in_background) if run_in_background is not None else False

        if run_in_background:
            try:
                process = subprocess.Popen(
                    ["/bin/bash", "-lc", command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    bufsize=0,
                )
            except FileNotFoundError:
                return {"error": "bash executable not found", "command": command}

            if process.stdout:
                _set_nonblocking(process.stdout)
            if process.stderr:
                _set_nonblocking(process.stderr)

            shell_id = uuid.uuid4().hex
            shell = BackgroundShell(shell_id=shell_id, process=process)
            BACKGROUND_SHELLS[shell_id] = shell

            return {
                "shell_id": shell_id,
                "status": "running",
                "result": "started",
                "command": command,
                "description": description,
            }

        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_val,
            )
            return {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
                "timed_out": False,
                "command": command,
                "description": description,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "exit_code": None,
                "timed_out": True,
                "command": command,
                "description": description,
            }


class BashOutputTool(BaseTool):
    """Retrieves incremental output from background bash shells."""

    @property
    def name(self) -> str:
        return "BashOutput"

    @property
    def description(self) -> str:
        return """Retrieves output from a running or completed background bash shell.
- Always returns only new output since the last check.
- Supports optional regex filtering to show only matching lines."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bash_id": {
                    "type": "string",
                    "description": "The ID of the background shell to retrieve output from",
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Optional regular expression to filter the output lines. Only lines matching this regex will be included in the result."
                    ),
                },
            },
            "required": ["bash_id"],
        }

    def execute(self, *, bash_id: str, filter: str | None = None) -> Dict[str, Any]:
        shell = BACKGROUND_SHELLS.get(bash_id)
        if shell is None:
            return {"error": f"No background shell found for id {bash_id}", "bash_id": bash_id}

        if filter is not None:
            try:
                re.compile(filter)
            except re.error as exc:
                return {"error": f"Invalid filter regex: {exc}", "bash_id": bash_id}

        try:
            shell.refresh()
            stdout, new_stdout_buffer = shell.consume(shell.stdout_buffer, filter)
            stderr, new_stderr_buffer = shell.consume(shell.stderr_buffer, filter)
            shell.stdout_buffer = new_stdout_buffer
            shell.stderr_buffer = new_stderr_buffer
            done = shell.status != "running"
            exit_code = shell.process.poll()
            if done and exit_code is not None and shell.stdout_buffer == "" and shell.stderr_buffer == "":
                shell.close()
            return {
                "bash_id": bash_id,
                "stdout": stdout,
                "stderr": stderr,
                "status": shell.status,
                "exit_code": exit_code,
            }
        except ValueError as exc:
            return {"error": str(exc), "bash_id": bash_id}


class KillBashTool(BaseTool):
    """Terminates background bash shells by ID."""

    @property
    def name(self) -> str:
        return "KillBash"

    @property
    def description(self) -> str:
        return """Kills a running background bash shell by its ID.
Use this tool when you need to terminate a long-running shell."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "shell_id": {
                    "type": "string",
                    "description": "The ID of the background shell to kill",
                }
            },
            "required": ["shell_id"],
        }

    def execute(self, *, shell_id: str) -> Dict[str, Any]:
        shell = BACKGROUND_SHELLS.get(shell_id)
        if shell is None:
            return {"error": f"No background shell found for id {shell_id}", "shell_id": shell_id}

        if shell.status != "running":
            shell.close()
            BACKGROUND_SHELLS.pop(shell_id, None)
            return {
                "shell_id": shell_id,
                "status": shell.status,
                "result": "already finished",
            }

        shell.process.terminate()
        try:
            shell.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            shell.process.kill()
            shell.process.wait(timeout=5)
        shell.refresh()
        shell.status = "terminated"
        shell.close()
        BACKGROUND_SHELLS.pop(shell_id, None)

        return {
            "shell_id": shell_id,
            "status": "terminated",
            "stdout": shell.stdout_buffer,
            "stderr": shell.stderr_buffer,
            "result": "killed",
        }
