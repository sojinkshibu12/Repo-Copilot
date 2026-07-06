"""Repo Copilot API — FastAPI application.

Exposes:
  - GET  /                    — dashboard UI (single-page HTML)
  - POST /webhook/github     — receive GitHub issue events and trigger the agent
  - GET  /api/issues          — list processed issues with latest decision
  - GET  /api/issues/{id}     — issue detail with all decisions
  - GET  /api/decisions/{id}  — decision detail with tool call trace
  - GET  /api/stats           — dashboard statistics
  - POST /api/eval/run        — trigger an eval run
  - GET  /api/eval/runs       — list eval runs
  - GET  /api/eval/runs/{id}  — eval run detail with per-case results
  - GET  /health              — health check
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from core.tracer import get_tracer, set_tracer, Tracer

from api.schemas import (
    HealthResponse, IssueOut, IssueDetail, DecisionOut, DecisionDetail,
    WebhookEvent, WebhookResponse, PaginatedResponse,
    EvalRunOut, EvalRunDetail, StatsResponse, TriggerEvalRequest,
)
from api.webhook import verify_github_signature, parse_issue_from_webhook
from api.dependencies import get_store, init_store, init_orchestrator
from models.issue import Issue
from models.decision import Decision, Classification, DecisionAction
from storage.postgres import PostgresStore

logger = logging.getLogger(__name__)

# ── Global state (set during lifespan) ─────────────────────────────

orchestrator = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()

    tracer = Tracer(service="repo-copilot")
    set_tracer(tracer)
    logger.info("Tracer initialized: emitter=%s", type(tracer.emitter).__name__)

    store = PostgresStore()
    init_store(store)
    logger.info("API started — Postgres store initialized")
    yield
    store.close()
    logger.info("API shut down")


app = FastAPI(
    title="Repo Copilot API",
    description="Agentic GitHub issue triager and PR drafter",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Tracing middleware ─────────────────────────────────────────────

@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    tracer = get_tracer()
    path = request.url.path
    method = request.method
    span_name = f"{method} {path}"

    attributes = {
        "http.method": method,
        "http.path": path,
        "http.query": str(request.url.query),
        "http.host": request.url.hostname,
    }

    with tracer.span(span_name, kind="http", attributes=attributes) as span:
        response = await call_next(request)
        span.set_attribute("http.status", response.status_code)
        return response


# ── Helpers ────────────────────────────────────────────────────────

def _paginate(items: list, total: int, page: int, per_page: int) -> PaginatedResponse:
    pages = max(1, (total + per_page - 1) // per_page)
    return PaginatedResponse(
        items=items, total=total, page=page, per_page=per_page, pages=pages,
    )


# ── Dashboard ──────────────────────────────────────────────────────

_DASHBOARD_HTML: str | None = None


def _load_dashboard() -> str:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
        if path.exists():
            _DASHBOARD_HTML = path.read_text(encoding="utf-8")
        else:
            _DASHBOARD_HTML = "<html><body><h1>Dashboard not found</h1></body></html>"
    return _DASHBOARD_HTML


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    return HTMLResponse(content=_load_dashboard())


# ── Health ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        uptime_seconds=time.time() - _start_time,
    )


# ── Webhook ───────────────────────────────────────────────────────

@app.post("/webhook/github", response_model=WebhookResponse, tags=["Webhook"])
async def github_webhook(
    request: Request,
    store: PostgresStore = Depends(get_store),
):
    payload_bytes = await request.body()
    await verify_github_signature(request, payload_bytes)

    import json
    payload = json.loads(payload_bytes)
    logger.info("Webhook received: action=%s event=%s",
                payload.get("action"), request.headers.get("X-GitHub-Event"))

    issue = parse_issue_from_webhook(payload)
    if issue is None:
        return WebhookResponse(
            status="ignored",
            message=f"Ignored event: {payload.get('action', 'unknown')}",
        )

    # Persist the issue
    issue_id = store.save_issue(issue)

    # Trigger agent (if orchestrator is configured)
    decision_id = None
    if orchestrator is not None:
        try:
            decision = orchestrator.run(issue)
            decision.issue_id = issue_id
            decision_id = store.save_decision(decision)
            logger.info("Agent decision saved: id=%d for issue #%d",
                        decision_id, issue.number)
        except Exception as e:
            logger.error("Agent execution failed: %s", e)
            # Save a fallback decision
            fallback = Decision(
                issue_id=issue_id,
                classification=Classification.UNCLEAR,
                action=DecisionAction.NO_ACTION,
                explanation=f"Agent execution failed: {e}",
            )
            decision_id = store.save_decision(fallback)

    return WebhookResponse(
        status="processed",
        decision_id=decision_id,
        message=f"Issue #{issue.number} processed",
    )


# ── Issues ────────────────────────────────────────────────────────

@app.get("/api/issues", response_model=PaginatedResponse, tags=["Issues"])
async def list_issues(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    repo: Optional[str] = None,
    classification: Optional[str] = None,
    store: PostgresStore = Depends(get_store),
):
    result = store.list_issues(
        page=page, per_page=per_page,
        repo=repo, classification=classification,
    )
    return _paginate(result.items, result.total, page, per_page)


@app.get("/api/issues/{issue_id}", response_model=IssueDetail, tags=["Issues"])
async def get_issue(
    issue_id: int,
    store: PostgresStore = Depends(get_store),
):
    issue = store.get_issue(issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")

    decisions = store.get_decisions_for_issue(issue_id)
    return IssueDetail(**issue, decisions=[DecisionOut(**d) for d in decisions])


# ── Decisions ─────────────────────────────────────────────────────

@app.get("/api/decisions/{decision_id}", response_model=DecisionDetail, tags=["Decisions"])
async def get_decision(
    decision_id: int,
    store: PostgresStore = Depends(get_store),
):
    decision = store.get_decision(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    issue = None
    if decision.get("issue_id"):
        issue = store.get_issue(decision["issue_id"])

    tool_calls = []
    if decision.get("session_id"):
        tool_calls = store.get_tool_calls_for_session(decision["session_id"])

    return DecisionDetail(
        **decision,
        issue=IssueOut(**issue) if issue else None,
        tool_calls=tool_calls,
    )


# ── Stats ─────────────────────────────────────────────────────────

@app.get("/api/stats", response_model=StatsResponse, tags=["Dashboard"])
async def get_stats(
    store: PostgresStore = Depends(get_store),
):
    return StatsResponse(**store.get_stats())


# ── Eval ──────────────────────────────────────────────────────────

@app.post("/api/eval/run", response_model=dict, tags=["Eval"])
async def trigger_eval(
    req: TriggerEvalRequest,
    store: PostgresStore = Depends(get_store),
):
    """Run the evaluation suite and persist results."""
    from evals.runner import EvalRunner
    from evals.report import ReportGenerator

    runner = EvalRunner(
        test_cases_dir="evals/test_cases",
        agent_fn=None,  # will fall back to mock
    )
    report = runner.run(results_path=req.results_path)

    # Persist eval run
    run_id = report.run_id
    store.save_eval_run(
        run_id=run_id,
        overall_score=report.overall_score,
        classification_accuracy=report.classification_accuracy,
        action_accuracy=report.action_accuracy,
        total_cases=report.total,
        failures=len(report.failures),
        summary={
            "categories": [
                {"name": c.name, "total": c.total,
                 "classification_correct": c.classification_correct,
                 "action_correct": c.action_correct}
                for c in report.categories
            ],
        },
    )

    for r in report.results:
        store.save_eval_result(
            run_id=run_id,
            case_id=r.case_id,
            category=r.category,
            classification_correct=r.classification_correct,
            action_correct=r.action_correct,
            expected=r.expected,
            actual=r.actual,
            errors=r.errors,
            judge_scores=r.judge_scores,
        )

    # Generate report files
    ReportGenerator().generate(report, formats=["json", "markdown"])

    return {
        "status": "completed",
        "run_id": run_id,
        "overall_score": report.overall_score,
        "classification_accuracy": report.classification_accuracy,
        "action_accuracy": report.action_accuracy,
        "total_cases": report.total,
        "failures": len(report.failures),
    }


@app.get("/api/eval/runs", response_model=list[EvalRunOut], tags=["Eval"])
async def list_eval_runs(
    limit: int = Query(20, ge=1, le=100),
    store: PostgresStore = Depends(get_store),
):
    return [EvalRunOut(**r) for r in store.list_eval_runs(limit=limit)]


@app.get("/api/eval/runs/{run_id}", response_model=EvalRunDetail, tags=["Eval"])
async def get_eval_run(
    run_id: str,
    store: PostgresStore = Depends(get_store),
):
    run = store.get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return EvalRunDetail(**run)
