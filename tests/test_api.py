"""API tests — uses FastAPI TestClient with an in-memory mock store."""

import json
import pytest
from datetime import datetime, timezone
from typing import Optional

from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_store, init_store
from storage.postgres import PostgresStore
from models.issue import Issue
from models.decision import Decision, Classification, DecisionAction


# ── Mock Store ───────────────────────────────────────────────────────────

class MockStore:
    """In-memory mock of PostgresStore for API testing."""

    def __init__(self):
        self.issues: dict[int, dict] = {}
        self.decisions: dict[int, dict] = {}
        self.eval_runs: dict[str, dict] = {}
        self.eval_results: list[dict] = []
        self._next_issue_id = 1
        self._next_decision_id = 1
        self._next_eval_id = 1

    def save_issue(self, issue: Issue) -> int:
        iid = self._next_issue_id
        self._next_issue_id += 1
        self.issues[iid] = {
            "id": iid,
            "gh_id": issue.id,
            "repo": issue.repo,
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "author": issue.author,
            "labels": issue.labels,
            "url": issue.url,
            "created_at": datetime.now(timezone.utc),
        }
        return iid

    def get_issue(self, issue_id: int) -> Optional[dict]:
        return self.issues.get(issue_id)

    def get_issue_by_number(self, repo: str, number: int) -> Optional[dict]:
        for i in self.issues.values():
            if i["repo"] == repo and i["number"] == number:
                return i
        return None

    def list_issues(self, page=1, per_page=20, repo=None, classification=None):
        items = list(self.issues.values())
        total = len(items)
        offset = (page - 1) * per_page
        page_items = items[offset:offset + per_page]
        from storage.postgres import PaginatedResult
        return PaginatedResult(
            items=page_items, total=total, page=page, per_page=per_page,
            pages=max(1, (total + per_page - 1) // per_page),
        )

    def save_decision(self, decision: Decision) -> int:
        did = self._next_decision_id
        self._next_decision_id += 1
        self.decisions[did] = {
            "id": did,
            "issue_id": decision.issue_id,
            "classification": decision.classification.value,
            "action_taken": decision.action.value,
            "confidence": decision.confidence,
            "pr_url": decision.pr_url or "",
            "explanation": decision.explanation,
            "session_id": "test-session-123",
            "created_at": datetime.now(timezone.utc),
        }
        return did

    def get_decision(self, decision_id: int) -> Optional[dict]:
        return self.decisions.get(decision_id)

    def get_decisions_for_issue(self, issue_id: int) -> list[dict]:
        return [d for d in self.decisions.values() if d["issue_id"] == issue_id]

    def log_tool_call(self, session_id, tool_name, input_data, output_data, latency_ms=0):
        pass

    def get_tool_calls_for_session(self, session_id: str) -> list[dict]:
        return [
            {"id": 1, "session_id": session_id, "tool_name": "classify_issue",
             "input_data": {"classification": "bug"}, "output_data": {},
             "latency_ms": 100, "created_at": datetime.now(timezone.utc).isoformat()},
        ]

    def save_eval_run(self, run_id, overall_score, classification_accuracy,
                      action_accuracy, total_cases, failures, summary=None):
        eid = self._next_eval_id
        self._next_eval_id += 1
        self.eval_runs[run_id] = {
            "id": eid, "run_id": run_id, "overall_score": overall_score,
            "classification_accuracy": classification_accuracy,
            "action_accuracy": action_accuracy, "total_cases": total_cases,
            "failures": failures, "summary": summary or {},
            "created_at": datetime.now(timezone.utc),
        }
        return run_id

    def save_eval_result(self, run_id, case_id, category,
                         classification_correct, action_correct,
                         expected, actual, errors, judge_scores=None):
        self.eval_results.append({
            "case_id": case_id, "category": category,
            "classification_correct": classification_correct,
            "action_correct": action_correct,
            "expected": expected, "actual": actual,
            "errors": errors, "judge_scores": judge_scores or {},
        })

    def list_eval_runs(self, limit=20):
        return list(self.eval_runs.values())[:limit]

    def get_eval_run(self, run_id: str) -> Optional[dict]:
        run = self.eval_runs.get(run_id)
        if run:
            run["results"] = self.eval_results
        return run

    def get_stats(self) -> dict:
        return {
            "total_issues": len(self.issues),
            "total_decisions": len(self.decisions),
            "classification_distribution": {"bug": 1},
            "action_distribution": {"opened_pr": 1},
            "prs_opened": 1,
            "eval_runs": len(self.eval_runs),
            "latest_eval_score": 1.0,
        }

    def close(self):
        pass


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_store():
    return MockStore()


@pytest.fixture
def client(mock_store):
    app.dependency_overrides[get_store] = lambda: mock_store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Tests ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["version"] == "0.1.0"

    def test_health_has_uptime(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] >= 0


class TestWebhook:
    def test_webhook_ignores_non_opened(self, client):
        payload = {"action": "edited", "issue": {"id": 1, "title": "Test"}}
        resp = client.post("/webhook/github", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_webhook_processes_opened_issue(self, client):
        payload = {
            "action": "opened",
            "issue": {
                "id": 42,
                "number": 1,
                "title": "Bug: typo in README",
                "body": "There's a typo.",
                "user": {"login": "test-user"},
                "labels": [],
                "html_url": "https://github.com/owner/repo/issues/1",
            },
            "repository": {"full_name": "owner/repo"},
        }
        resp = client.post("/webhook/github", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processed"
        assert "1" in data["message"]


class TestIssues:
    def test_list_issues_empty(self, client):
        resp = client.get("/api/issues")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_issues_with_data(self, client, mock_store):
        # Seed an issue
        mock_store.save_issue(Issue(
            id=100, repo="owner/repo", number=1,
            title="Test issue", body="Body", author="user",
        ))
        resp = client.get("/api/issues")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["title"] == "Test issue"

    def test_get_issue_not_found(self, client):
        resp = client.get("/api/issues/999")
        assert resp.status_code == 404

    def test_get_issue_with_decisions(self, client, mock_store):
        iid = mock_store.save_issue(Issue(
            id=200, repo="owner/repo", number=2,
            title="Bug", body="Desc", author="user",
        ))
        mock_store.save_decision(Decision(
            issue_id=iid, classification=Classification.BUG,
            action=DecisionAction.OPENED_PR,
            explanation="Fixed typo",
        ))
        resp = client.get(f"/api/issues/{iid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Bug"
        assert len(data["decisions"]) == 1
        assert data["decisions"][0]["classification"] == "bug"


class TestDecisions:
    def test_get_decision_not_found(self, client):
        resp = client.get("/api/decisions/999")
        assert resp.status_code == 404

    def test_get_decision_with_trace(self, client, mock_store):
        iid = mock_store.save_issue(Issue(
            id=300, repo="owner/repo", number=3,
            title="Feature request", body="Add dark mode", author="user",
        ))
        did = mock_store.save_decision(Decision(
            issue_id=iid, classification=Classification.FEATURE,
            action=DecisionAction.COMMENTED,
            explanation="This is a feature request, not a bug.",
        ))
        resp = client.get(f"/api/decisions/{did}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"] == "feature"
        assert data["issue"] is not None
        assert len(data["tool_calls"]) > 0


class TestStats:
    def test_stats_returns_counts(self, client, mock_store):
        iid = mock_store.save_issue(Issue(
            id=400, repo="owner/repo", number=4, title="Test", body="", author="u",
        ))
        mock_store.save_decision(Decision(
            issue_id=iid, classification=Classification.BUG,
            action=DecisionAction.OPENED_PR, explanation="",
        ))

        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_issues"] >= 1
        assert data["total_decisions"] >= 1


class TestEval:
    def test_trigger_eval(self, client):
        resp = client.post("/api/eval/run", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["total_cases"] > 0

    def test_list_eval_runs_empty(self, client):
        resp = client.get("/api/eval/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_eval_run_not_found(self, client):
        resp = client.get("/api/eval/runs/nonexistent")
        assert resp.status_code == 404

    def test_get_eval_run_with_results(self, client, mock_store):
        mock_store.save_eval_run(
            run_id="test-run-001", overall_score=0.95,
            classification_accuracy=1.0, action_accuracy=0.9,
            total_cases=20, failures=1,
        )
        mock_store.save_eval_result(
            run_id="test-run-001", case_id="001-bug", category="bug-scoped",
            classification_correct=True, action_correct=True,
            expected={"classification": "bug"}, actual={"classification": "bug"},
            errors=[],
        )
        resp = client.get("/api/eval/runs/test-run-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_score"] == 0.95
        assert len(data["results"]) >= 1


class TestPagination:
    def test_pagination_returns_correct_pages(self, client, mock_store):
        for i in range(5):
            mock_store.save_issue(Issue(
                id=500 + i, repo="owner/repo", number=10 + i,
                title=f"Issue {i}", body="", author="user",
            ))
        resp = client.get("/api/issues?per_page=2&page=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["pages"] == 3

    def test_pagination_page_2(self, client, mock_store):
        for i in range(5):
            mock_store.save_issue(Issue(
                id=600 + i, repo="owner/repo", number=20 + i,
                title=f"Issue {i}", body="", author="user",
            ))
        resp = client.get("/api/issues?per_page=2&page=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["page"] == 2
