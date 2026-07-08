import os
import logging
from dataclasses import dataclass

from models.issue import Issue

logger = logging.getLogger(__name__)


@dataclass
class GitHubConfig:
    token: str
    repo_full_name: str


class GitHubClient:
    def __init__(self, config: GitHubConfig | None = None):
        self.config = config or GitHubConfig(
            token=os.environ.get("GITHUB_TOKEN", ""),
            repo_full_name=os.environ.get("GITHUB_WATCHED_REPO", ""),
        )
        self._client = self._build_client()

    def _build_client(self):
        if not self.config.token:
            logger.warning("No GITHUB_TOKEN set — using mock client")
            return _MockGitHub()
        try:
            from github import Github
            return Github(self.config.token)
        except ImportError:
            logger.warning("PyGithub not installed — using mock client")
            return _MockGitHub()

    def get_repo(self):
        return self._client.get_repo(self.config.repo_full_name)

    def get_issue(self, number: int) -> Issue:
        gh_issue = self._client.get_issue(self.config.repo_full_name, number)
        return Issue(
            id=gh_issue.id,
            repo=self.config.repo_full_name,
            number=gh_issue.number,
            title=gh_issue.title,
            body=gh_issue.body or "",
            author=gh_issue.user.login,
            labels=[l.name for l in gh_issue.labels],
            url=gh_issue.html_url,
        )

    def comment_on_issue(self, issue_number: int, body: str) -> dict:
        repo = self.get_repo()
        issue = repo.get_issue(issue_number)
        comment = issue.create_comment(body)
        logger.info("Commented on issue #%d (comment id: %d)", issue_number, comment.id)
        return {"comment_id": comment.id, "url": comment.html_url}

    def add_label(self, issue_number: int, label: str) -> None:
        repo = self.get_repo()
        issue = repo.get_issue(issue_number)
        issue.add_to_labels(label)
        logger.info("Added label '%s' to issue #%d", label, issue_number)

    def open_draft_pr(
        self, title: str, body: str, head: str, base: str = "main"
    ) -> dict:
        repo = self.get_repo()
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head,
            base=base,
            draft=True,
        )
        logger.info("Opened draft PR #%d", pr.number)
        return {"pr_number": pr.number, "pr_url": pr.html_url}


class _MockGitHub:
    def get_repo(self, full_name):
        return _MockRepo()

    def get_issue(self, full_name, number):
        return _MockGHIssue()


class _MockRepo:
    def get_issue(self, number):
        return _MockGHIssue()

    def create_pull(self, **kwargs):
        return _MockPR()

    def get_pulls(self, state="open", sort="updated", direction="desc"):
        return []


class _MockGHIssue:
    id = 0
    number = 0
    title = "Mock Issue"
    body = "Mock body"
    html_url = "https://github.com/mock/repo/issues/0"

    class MockUser:
        login = "mock-user"

    user = MockUser()
    labels = []

    def create_comment(self, body):
        return type("Comment", (), {"id": 0, "html_url": ""})()

    def add_to_labels(self, label):
        pass


class _MockPR:
    number = 0
    html_url = "https://github.com/mock/repo/pull/0"
