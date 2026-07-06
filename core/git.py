import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GitClient:
    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)

    def clone(self, url: str, path: Path | None = None) -> None:
        target = path or self.repo_path
        if target.exists():
            logger.info("Repo already exists at %s", target)
            return
        try:
            import git
            git.Repo.clone_from(url, target)
            logger.info("Cloned %s to %s", url, target)
        except ImportError:
            logger.warning("GitPython not installed — using subprocess")
            self._run(f"git clone {url} {target}")

    def checkout_branch(self, name: str, base: str = "main") -> None:
        import git
        repo = git.Repo(self.repo_path)
        current = repo.create_head(name, commit=repo.head.commit)
        current.checkout()
        logger.info("Checked out branch '%s' from '%s'", name, base)

    def stage_files(self, paths: list[str]) -> None:
        import git
        repo = git.Repo(self.repo_path)
        repo.index.add(paths)
        logger.info("Staged %d files", len(paths))

    def commit(self, message: str) -> str:
        import git
        repo = git.Repo(self.repo_path)
        commit = repo.index.commit(message)
        logger.info("Committed: %s", commit.hexsha[:8])
        return commit.hexsha

    def push(self, remote: str = "origin", branch: str | None = None) -> None:
        import git
        repo = git.Repo(self.repo_path)
        branch = branch or repo.active_branch.name
        repo.remote(remote).push(branch)
        logger.info("Pushed %s to %s/%s", branch, remote, branch)

    def diff(self, base: str = "main", head: str | None = None) -> str:
        import git
        repo = git.Repo(self.repo_path)
        head = head or repo.active_branch.name
        return repo.git.diff(f"{base}...{head}")

    def _run(self, cmd: str) -> str:
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout or result.stderr
