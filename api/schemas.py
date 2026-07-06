from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


# ── Health ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    uptime_seconds: float = 0.0
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    tools_registered: int = 0


# ── Issues ─────────────────────────────────────────────────────────────

class IssueOut(BaseModel):
    id: int
    gh_id: int
    repo: str
    number: int
    title: str
    body: str
    author: str
    labels: list[str] = []
    url: str = ""
    created_at: Optional[datetime] = None

    classification: Optional[str] = None
    action_taken: Optional[str] = None
    confidence: Optional[float] = None
    pr_url: Optional[str] = None
    explanation: Optional[str] = None
    decision_created_at: Optional[datetime] = None


class IssueDetail(BaseModel):
    id: int
    gh_id: int
    repo: str
    number: int
    title: str
    body: str
    author: str
    labels: list[str]
    url: str
    created_at: Optional[datetime] = None
    decisions: list["DecisionOut"] = []


# ── Decisions ──────────────────────────────────────────────────────────

class DecisionOut(BaseModel):
    id: int
    issue_id: int
    classification: str
    action_taken: str
    confidence: float = 0.0
    pr_url: Optional[str] = None
    explanation: str = ""
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None


class DecisionDetail(BaseModel):
    id: int
    issue_id: int
    classification: str
    action_taken: str
    confidence: float
    pr_url: Optional[str] = None
    explanation: str
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    issue: Optional[IssueOut] = None
    tool_calls: list[dict] = []


# ── Webhook ────────────────────────────────────────────────────────────

class WebhookEvent(BaseModel):
    action: str
    issue: Optional[dict] = None
    repository: Optional[dict] = None
    sender: Optional[dict] = None


class WebhookResponse(BaseModel):
    status: str
    decision_id: Optional[int] = None
    message: str = ""


# ── Pagination ─────────────────────────────────────────────────────────

class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int
    per_page: int
    pages: int


# ── Eval ───────────────────────────────────────────────────────────────

class EvalRunOut(BaseModel):
    id: int
    run_id: str
    overall_score: Optional[float] = None
    classification_accuracy: Optional[float] = None
    action_accuracy: Optional[float] = None
    total_cases: Optional[int] = None
    failures: Optional[int] = None
    summary: dict = {}
    created_at: Optional[datetime] = None


class EvalRunDetail(BaseModel):
    id: int
    run_id: str
    overall_score: Optional[float] = None
    classification_accuracy: Optional[float] = None
    action_accuracy: Optional[float] = None
    total_cases: Optional[int] = None
    failures: Optional[int] = None
    summary: dict = {}
    created_at: Optional[datetime] = None
    results: list[dict] = []


# ── Stats ──────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_issues: int = 0
    total_decisions: int = 0
    classification_distribution: dict[str, int] = {}
    action_distribution: dict[str, int] = {}
    prs_opened: int = 0
    eval_runs: int = 0
    latest_eval_score: Optional[float] = None


# ── Triggers ───────────────────────────────────────────────────────────

class TriggerEvalRequest(BaseModel):
    results_path: Optional[str] = None
    min_accuracy: Optional[float] = None
