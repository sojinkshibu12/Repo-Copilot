"""GitHub webhook handler — verifies signatures, parses payloads, triggers the agent."""

import hashlib
import hmac
import logging
import os

from fastapi import Request, HTTPException

from models.issue import Issue

logger = logging.getLogger(__name__)


async def verify_github_signature(request: Request, payload_bytes: bytes) -> bool:
    """Verify X-Hub-Signature-256 against the webhook secret.

    If no secret is configured, skip verification (dev mode).
    """
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        logger.warning("No GITHUB_WEBHOOK_SECRET set — skipping signature verification")
        return True

    signature_header = request.headers.get("X-Hub-Signature-256", "")
    if not signature_header.startswith("sha256="):
        raise HTTPException(status_code=400, detail="Missing or invalid signature header")

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload_bytes, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    return True


def parse_issue_from_webhook(payload: dict) -> Issue | None:
    """Extract an Issue model from a GitHub issues.opened webhook payload.

    Returns None if the event is not an issue being opened.
    """
    if payload.get("action") != "opened":
        return None

    gh_issue = payload.get("issue")
    if not gh_issue:
        return None

    repo = payload.get("repository", {})
    repo_full_name = repo.get("full_name", "unknown/unknown")

    return Issue(
        id=gh_issue["id"],
        repo=repo_full_name,
        number=gh_issue["number"],
        title=gh_issue["title"],
        body=gh_issue.get("body") or "",
        author=gh_issue["user"]["login"],
        labels=[l["name"] for l in gh_issue.get("labels", [])],
        created_at=None,
        url=gh_issue["html_url"],
    )


def parse_issue_from_dict(data: dict) -> Issue:
    """Build an Issue from a flat dict (e.g., test fixture or API input)."""
    return Issue(
        id=data.get("id", 0),
        repo=data.get("repo", "owner/repo"),
        number=data.get("number", 0),
        title=data.get("title", ""),
        body=data.get("body", ""),
        author=data.get("author", "unknown"),
        labels=data.get("labels", []),
        url=data.get("url", ""),
    )
