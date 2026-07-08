"""File, git, GitHub, and execution tool handlers for the agent."""

import json
import logging
import subprocess
import os
from pathlib import Path
from glob import glob as _glob

logger = logging.getLogger(__name__)


class OperationToolSet:
    """Tool handlers for file operations, git, GitHub API, and command execution."""

    def __init__(self, repo_path: str | Path = ".", github_token: str = "", github_repo: str = ""):
        self.repo_path = Path(repo_path).resolve()
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.github_repo = github_repo or os.environ.get("GITHUB_WATCHED_REPO", "")
        self._current_issue_number: int | None = None
        logger.info("OperationToolSet initialized: repo=%s github_repo=%s", self.repo_path, self.github_repo)

    def set_current_issue(self, number: int):
        self._current_issue_number = number

    # ── File operations ───────────────────────────────────────────

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        """Read a file from the repository."""
        full = self.repo_path / path
        if not full.exists():
            return f"File not found: {path}"
        if not full.is_file():
            return f"Not a file: {path}"
        try:
            lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return f"Error reading {path}: {e}"

        start = max(0, offset - 1)
        end = min(len(lines), start + limit)
        snippet = lines[start:end]
        meta = f"File: {path} ({len(lines)} lines, showing {start+1}-{end})\n"
        return meta + "\n".join(f"{i+1:>6} | {l}" for i, l in enumerate(snippet, start=start))

    def search_code(self, pattern: str, include: str = "", path: str = "") -> str:
        """Regex search in the codebase."""
        search_root = self.repo_path / path if path else self.repo_path
        if not search_root.exists():
            return f"Path not found: {path or self.repo_path}"

        # Try ripgrep first, fall back to grep
        cmd = ["rg", "-n", pattern, str(search_root)]
        if include:
            cmd.extend(["-g", include])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            # ripgrep not installed — fall back to standard grep
            cmd = ["grep", "-rn", pattern, str(search_root)]
            if include:
                cmd.extend(["--include", include])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except FileNotFoundError:
                return "No search tool available (install ripgrep or grep)"
            except subprocess.TimeoutExpired:
                return f"Search timed out for '{pattern}'"

        if result.returncode == 0:
            output = result.stdout.strip()
            if not output:
                return f"No matches for '{pattern}'" + (f" in {include}" if include else "")
            lines = output.splitlines()
            return f"Found {len(lines)} match(es):\n" + "\n".join(lines[:50]) + ("\n..." if len(lines) > 50 else "")
        elif result.returncode == 1:
            return f"No matches for '{pattern}'" + (f" in {include}" if include else "")
        else:
            return f"Search failed (code {result.returncode}): {result.stderr.strip() or result.stdout.strip()[:200]}"

    def glob(self, pattern: str) -> str:
        """List files matching a glob pattern."""
        matches = [str(p.relative_to(self.repo_path)) for p in self.repo_path.rglob(pattern)]
        if not matches:
            return f"No files match '{pattern}'"
        matches.sort()
        return f"Found {len(matches)} file(s) matching '{pattern}':\n" + "\n".join(matches[:100]) + ("\n..." if len(matches) > 100 else "")

    # ── Execution ─────────────────────────────────────────────────

    def run_tests(self, test_command: str = "pytest") -> str:
        """Run tests in the repo."""
        try:
            result = subprocess.run(
                test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.repo_path),
            )
            output = (result.stdout or "") + (result.stderr or "")
            output = output.strip()[:3000]
            status = "passed" if result.returncode == 0 else "failed"
            return f"Tests {status} (exit code {result.returncode}):\n{output}"
        except subprocess.TimeoutExpired:
            return "Tests timed out after 120s"
        except Exception as e:
            return f"Failed to run tests: {e}"

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Run a command in the repo sandbox."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.repo_path),
            )
            output = (result.stdout or "") + (result.stderr or "")
            output = output.strip()[:3000]
            return f"Command completed (exit code {result.returncode}):\n{output}"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Command failed: {e}"

    # ── GitHub operations ─────────────────────────────────────────

    def _github_client(self):
        from core.github import GitHubClient, GitHubConfig
        return GitHubClient(GitHubConfig(
            token=self.github_token,
            repo_full_name=self.github_repo,
        ))

    def comment_on_issue(self, body: str) -> str:
        """Post a comment on the current GitHub issue."""
        if not self._current_issue_number:
            return "No current issue set — cannot comment"
        try:
            client = self._github_client()
            result = client.comment_on_issue(self._current_issue_number, body)
            return json.dumps(result)
        except Exception as e:
            return f"Failed to comment: {e}"

    def add_label(self, label: str) -> str:
        """Add a label to the current GitHub issue."""
        if not self._current_issue_number:
            return "No current issue set — cannot add label"
        try:
            client = self._github_client()
            client.add_label(self._current_issue_number, label)
            return f"Added label '{label}' to issue #{self._current_issue_number}"
        except Exception as e:
            return f"Failed to add label: {e}"

    def open_draft_pr(self, title: str, body: str, base: str = "main") -> str:
        """Push the branch, then open a draft pull request."""
        try:
            import git
            repo = git.Repo(self.repo_path)
            head = repo.active_branch.name
            # Auto-push before opening PR (branch must exist on remote)
            push_result = self.push_branch()
            if "Failed" in push_result:
                return f"Push failed before opening PR: {push_result}"
            client = self._github_client()
            result = client.open_draft_pr(title=title, body=body, head=head, base=base)
            return json.dumps(result)
        except Exception as e:
            return f"Failed to open PR: {e}"

    # ── Git operations ────────────────────────────────────────────

    def _git_client(self):
        from core.git import GitClient
        return GitClient(self.repo_path)

    def create_branch(self, branch_name: str) -> str:
        """Create a new branch and switch to it."""
        try:
            client = self._git_client()
            client.checkout_branch(branch_name)
            return f"Created and switched to branch '{branch_name}'"
        except Exception as e:
            return f"Failed to create branch: {e}"

    def push_branch(self) -> str:
        """Push the current branch to the remote."""
        try:
            client = self._git_client()
            import git
            repo = git.Repo(self.repo_path)
            branch = repo.active_branch.name
            client.push(branch=branch)
            return f"Pushed branch '{branch}' to origin"
        except Exception as e:
            return f"Failed to push: {e}"

    def commit_changes(self, message: str) -> str:
        """Stage and commit all local changes."""
        try:
            client = self._git_client()
            client.stage_files(["."])
            sha = client.commit(message)
            return f"Committed changes: {sha}"
        except Exception as e:
            return f"Failed to commit: {e}"

    # ── Registration helper ───────────────────────────────────────

    def get_tool_handlers(self) -> dict:
        return {
            "read_file": self.read_file,
            "search_code": self.search_code,
            "glob": self.glob,
            "run_tests": self.run_tests,
            "run_command": self.run_command,
            "comment_on_issue": self.comment_on_issue,
            "add_label": self.add_label,
            "open_draft_pr": self.open_draft_pr,
            "create_branch": self.create_branch,
            "commit_changes": self.commit_changes,
            "push_branch": self.push_branch,
        }
