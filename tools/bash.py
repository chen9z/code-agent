from __future__ import annotations

import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

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
        remainder = remainder[newline_index + 1:]
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

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use `ls` to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use `ls foo` to check that "foo" exists and is the intended parent directory

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
  - You can use the `run_in_background` parameter to run the command in the background, which allows you to continue working while the command runs. You can monitor the output using the Bash tool as it becomes available. You do not need to use '&' at the end of the command when using this parameter.
  
  - Avoid using Bash with the `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or when these commands are truly necessary for the task. Instead, always prefer using the dedicated tools for these commands:
    - File search: Use Glob (NOT find or ls)
    - Content search: Use Grep (NOT grep or rg)
    - Read files: Use Read (NOT cat/head/tail)
    - Edit files: Use Edit (NOT sed/awk)
    - Write files: Use Write (NOT echo >/cat <<EOF)
    - Communication: Output text directly (NOT echo/printf)
  - When issuing multiple commands:
    - If the commands are independent and can run in parallel, make multiple Bash tool calls in a single message. For example, if you need to run "git status" and "git diff", send a single message with two Bash tool calls in parallel.
    - If the commands depend on each other and must run sequentially, use a single Bash call with '&&' to chain them together (e.g., `git add . && git commit -m "message" && git push`). For instance, if one operation must complete before another starts (like mkdir before cp, Write before Bash for git operations, or git add before git commit), run these operations sequentially instead.
    - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
    - DO NOT use newlines to separate commands (newlines are ok in quoted strings)
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.
    <good-example>
    pytest /foo/bar/tests
    </good-example>
    <bad-example>
    cd /foo/bar && pytest tests
    </bad-example>

# Committing changes with git

Only create commits when requested by the user. If unclear, ask first. When the user asks you to create a new git commit, follow these steps carefully:

Git Safety Protocol:
- NEVER update the git config
- NEVER run destructive/irreversible git commands (like push --force, hard reset, etc) unless the user explicitly requests them 
- NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless the user explicitly requests it
- NEVER run force push to main/master, warn the user if they request it
- Avoid git commit --amend.  ONLY use --amend when either (1) user explicitly requested amend OR (2) adding edits from pre-commit hook (additional instructions below) 
- Before amending: ALWAYS check authorship (git log -1 --format='%an %ae')
- NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.

1. You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. run the following bash commands in parallel, each using the Bash tool:
  - Run a git status command to see all untracked files.
  - Run a git diff command to see both staged and unstaged changes that will be committed.
  - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.
2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:
  - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. "add" means a wholly new feature, "update" means an enhancement to an existing feature, "fix" means a bug fix, etc.).
  - Do not commit files that likely contain secrets (.env, credentials.json, etc). Warn the user if they specifically request to commit those files
  - Draft a concise (1-2 sentences) commit message that focuses on the "why" rather than the "what"
  - Ensure it accurately reflects the changes and their purpose
3. You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. run the following commands:
   - Add relevant untracked files to the staging area.
   - Create the commit with a message ending with:
   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   - Run git status after the commit completes to verify success.
   Note: git status depends on the commit completing, so run it sequentially after the commit.
4. If the commit fails due to pre-commit hook changes, retry ONCE. If it succeeds but files were modified by the hook, verify it's safe to amend:
   - Check authorship: git log -1 --format='%an %ae'
   - Check not pushed: git status shows "Your branch is ahead"
   - If both true: amend your commit. Otherwise: create NEW commit (never amend other developers' commits)

Important notes:
- NEVER run additional commands to read or explore code, besides git bash commands
- NEVER use the TodoWrite or Task tools
- DO NOT push to the remote repository unless the user explicitly asks you to do so
- IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.
- If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit
- In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:
<example>
git commit -m "$(cat <<'EOF'
   Commit message here.

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
</example>

# Creating pull requests
Use the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases. If given a Github URL use the gh command to get the information needed.

IMPORTANT: When the user asks you to create a pull request, follow these steps carefully:

1. You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. run the following bash commands in parallel using the Bash tool, in order to understand the current state of the branch since it diverged from the main branch:
   - Run a git status command to see all untracked files
   - Run a git diff command to see both staged and unstaged changes that will be committed
   - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote
   - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch)
2. Analyze all changes that will be included in the pull request, making sure to look at all relevant commits (NOT just the latest commit, but ALL commits that will be included in the pull request!!!), and draft a pull request summary
3. You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. run the following commands in parallel:
   - Create new branch if needed
   - Push to remote with -u flag if needed
   - Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting.
<example>
gh pr create --title "the pr title" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points>

## Test plan
[Bulleted markdown checklist of TODOs for testing the pull request...]

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
</example>

Important:
- DO NOT use the TodoWrite or Task tools
- Return the PR URL when you're done, so the user can see it

# Other common operations
- View comments on a Github PR: gh api repos/foo/bar/pulls/123/comments"""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "required": ["command"],
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
                        "Clear, concise description of what this command does in 5-10 words, in active voice. Examples:\n"
                        "Input: ls\nOutput: List files in current directory\n\n"
                        "Input: git status\nOutput: Show working tree status\n\n"
                        "Input: npm install\nOutput: Install package dependencies\n\n"
                        "Input: mkdir foo\nOutput: Create directory 'foo'"
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "Set to true to run this command in the background. Use BashOutput to read the output later."
                    ),
                },
                "dangerouslyDisableSandbox": {
                    "type": "boolean",
                    "description": (
                        "Set this to true to dangerously override sandbox mode and run commands without sandboxing."
                    ),
                },
            },
        }

    def execute(
            self,
            *,
            command: str,
            timeout: float | int | None = None,
            description: str | None = None,
            run_in_background: bool | None = None,
            dangerouslyDisableSandbox: bool | None = None,
    ) -> Dict[str, Any]:
        description_text = description.strip() if isinstance(description, str) else ""

        def build_payload(status: str, content: str, data: Dict[str, Any]) -> Dict[str, Any]:
            payload = dict(data)
            if description_text:
                payload["description"] = description_text
            if dangerouslyDisableSandbox:
                payload["dangerouslyDisableSandbox"] = bool(dangerouslyDisableSandbox)
            return {
                "status": status,
                "content": content,
                "data": payload,
            }

        def summarize_output(stdout: str, stderr: str, fallback: str) -> str:
            for candidate in (stdout.strip(), stderr.strip()):
                if candidate:
                    return candidate
            return fallback

        if not command:
            return build_payload("error", "command must be a non-empty string", {"command": command})

        if timeout is not None:
            try:
                timeout_val = float(timeout) / 1000.0
            except (TypeError, ValueError):
                return build_payload("error", "timeout must be a number", {"command": command})
            if timeout_val <= 0 or timeout_val > MAX_TIMEOUT_MS / 1000:
                return build_payload(
                    "error",
                    "timeout must be between 1 and 600000 milliseconds",
                    {"command": command},
                )
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
                return build_payload("error", "bash executable not found", {"command": command})

            if process.stdout:
                _set_nonblocking(process.stdout)
            if process.stderr:
                _set_nonblocking(process.stderr)

            shell_id = uuid.uuid4().hex
            shell = BackgroundShell(shell_id=shell_id, process=process)
            BACKGROUND_SHELLS[shell_id] = shell

            data = {
                "shell_id": shell_id,
                "status": "running",
                "content": "started",
                "command": command,
            }
            return build_payload("success", f"Started background shell {shell_id}", data)

        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_val,
            )
            data = {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
                "timed_out": False,
                "command": command,
            }
            status = "success" if completed.returncode == 0 else "error"
            content = summarize_output(
                completed.stdout,
                completed.stderr,
                f"Command exited with code {completed.returncode}",
            )
            return build_payload(status, content, data)
        except subprocess.TimeoutExpired as exc:
            data = {
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "exit_code": None,
                "timed_out": True,
                "command": command,
            }
            return build_payload(
                "error",
                f"Command timed out after {timeout_val:.0f}s",
                data,
            )


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
            message = f"No background shell found for id {bash_id}"
            return {"error": message, "bash_id": bash_id, "content": message}

        if filter is not None:
            try:
                re.compile(filter)
            except re.error as exc:
                message = f"Invalid filter regex: {exc}"
                return {"error": message, "bash_id": bash_id, "content": message}

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
            result_parts: List[str] = []
            if stdout:
                result_parts.append(stdout.rstrip("\n"))
            if stderr:
                result_parts.append(f"STDERR:\n{stderr.rstrip('\n')}")
            result_text = "\n\n".join(part for part in result_parts if part)
            if not result_text:
                result_text = shell.status
            return {
                "bash_id": bash_id,
                "stdout": stdout,
                "stderr": stderr,
                "status": shell.status,
                "exit_code": exit_code,
                "content": result_text,
            }
        except ValueError as exc:
            message = str(exc)
            return {"error": message, "bash_id": bash_id, "content": message}


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
            message = f"No background shell found for id {shell_id}"
            return {"error": message, "shell_id": shell_id, "content": message}

        if shell.status != "running":
            shell.close()
            BACKGROUND_SHELLS.pop(shell_id, None)
            return {
                "shell_id": shell_id,
                "status": shell.status,
                "content": "already finished",
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
            "content": "killed",
        }
