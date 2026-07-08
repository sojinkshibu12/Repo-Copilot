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

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

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
_op_tools = None
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

    global orchestrator, _op_tools
    try:
        from core.llm import LLMClient
        from agent.orchestrator import Orchestrator
        from agent.tools.retrieval import RetrievalToolSet
        from agent.tools.operations import OperationToolSet
        from storage.vector_store import VectorStore

        llm = LLMClient()
        orch = Orchestrator(llm)
        ret_tools = RetrievalToolSet(VectorStore())
        orch.register_tool_set(ret_tools.get_tool_handlers())
        ops = OperationToolSet(
            repo_path=os.environ.get("REPO_LOCAL_PATH", "."),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            github_repo=os.environ.get("GITHUB_WATCHED_REPO", ""),
        )
        orch.register_tool_set(ops.get_tool_handlers())
        orch.register_tool("classify_issue", lambda **kw: json.dumps(kw))
        init_orchestrator(orch)
        orchestrator = orch
        _op_tools = ops
        logger.info("Orchestrator initialized: provider=%s model=%s",
                     llm.provider_name, llm.model)
    except Exception as e:
        logger.warning("Orchestrator init skipped: %s — issues will get placeholder decisions", e)

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


# ── Repo (Branches, Commits, PRs) ────────────────────────────────

_repo_instance = None


def _get_repo():
    global _repo_instance
    if _repo_instance is None:
        try:
            import git
            path = os.environ.get("REPO_LOCAL_PATH", ".")
            _repo_instance = git.Repo(path)
        except Exception as e:
            logger.warning("Cannot open git repo: %s", e)
            return None
    return _repo_instance


@app.get("/api/repo/branches", tags=["Repository"])
async def list_branches():
    """List all branches from GitHub API."""
    try:
        from core.github import GitHubClient
        gh = GitHubClient()
        pygh_repo = gh.get_repo()
        branches = []
        default = pygh_repo.default_branch
        for b in pygh_repo.get_branches():
            branches.append({
                "name": b.name,
                "is_head": b.name == default,
                "commit_sha": b.commit.sha[:8],
                "commit_message": b.commit.commit.message.strip().split("\n")[0],
                "commit_author": b.commit.commit.author.name or str(b.commit.commit.author),
                "commit_time": b.commit.commit.author.date.isoformat() + "Z" if b.commit.commit.author.date else None,
            })
        return sorted(branches, key=lambda x: (not x["is_head"], x["name"]))
    except Exception as e:
        logger.error("Failed to list branches: %s", e)
        return []


@app.get("/api/repo/commits", tags=["Repository"])
async def list_commits(
    branch: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """List commits from GitHub API with pagination."""
    try:
        from core.github import GitHubClient
        gh = GitHubClient()
        pygh_repo = gh.get_repo()
        sha = branch or pygh_repo.default_branch
        commits = list(pygh_repo.get_commits(sha=sha)[:page * per_page])
        start = (page - 1) * per_page
        items = commits[start:start + per_page]
        result = []
        for c in items:
            result.append({
                "sha": c.sha[:8],
                "full_sha": c.sha,
                "message": c.commit.message.strip().split("\n")[0],
                "body": "\n".join(c.commit.message.strip().split("\n")[1:]).strip(),
                "author": c.commit.author.name or str(c.commit.author),
                "email": c.commit.author.email or "",
                "time": c.commit.author.date.isoformat() + "Z" if c.commit.author.date else None,
                "files_changed": len(c.files) if c.files else 0,
            })
        return {"items": result, "total": len(commits), "page": page, "per_page": per_page}
    except Exception as e:
        logger.error("Failed to list commits: %s", e)
        return {"items": [], "total": 0, "page": page, "per_page": per_page}


@app.get("/api/repo/pulls", tags=["Repository"])
async def list_pulls(
    state: str = Query("open"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """List pull requests from GitHub."""
    try:
        from core.github import GitHubClient
        gh = GitHubClient()
        repo = gh.get_repo()
        pulls = repo.get_pulls(state=state, sort="updated", direction="desc")
        all_pulls = list(pulls[:page * per_page])
        start = (page - 1) * per_page
        items = all_pulls[start:start + per_page]
        result = []
        for pr in items:
            result.append({
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "draft": getattr(pr, "draft", False),
                "author": pr.user.login if pr.user else "unknown",
                "head": pr.head.ref,
                "base": pr.base.ref,
                "created_at": pr.created_at.isoformat() + "Z" if pr.created_at else None,
                "updated_at": pr.updated_at.isoformat() + "Z" if pr.updated_at else None,
                "url": pr.html_url,
            })
        return {"items": result, "total": len(all_pulls), "page": page, "per_page": per_page}
    except Exception as e:
        logger.error("Failed to list pulls: %s", e)
        return {"items": [], "total": 0, "page": page, "per_page": per_page}


@app.get("/api/repo/network", tags=["Repository"])
async def repo_network():
    """Return the full branch/commit DAG for a visual repo graph."""
    try:
        from core.github import GitHubClient
        gh = GitHubClient()
        pygh_repo = gh.get_repo()

        # Get all branches with their latest commit
        branches_raw = list(pygh_repo.get_branches())
        default = pygh_repo.default_branch

        # Collect the DAG: for each branch, trace back commits
        import collections
        dag = collections.defaultdict(list)
        branch_info = {}
        max_commits = 100

        for b in branches_raw:
            branch_info[b.name] = {
                "name": b.name,
                "is_default": b.name == default,
                "sha": b.commit.sha[:8],
                "full_sha": b.commit.sha,
                "message": b.commit.commit.message.strip().split("\n")[0],
                "author": b.commit.commit.author.name or str(b.commit.commit.author),
                "time": b.commit.commit.author.date.isoformat() + "Z" if b.commit.commit.author.date else None,
            }
            # Trace commits for this branch
            seen = set()
            for c in pygh_repo.get_commits(sha=b.name)[:max_commits]:
                if c.sha in seen:
                    break
                seen.add(c.sha)
                parents = [p.sha[:8] for p in c.parents[:2]]
                dag[c.sha[:8]] = {
                    "sha": c.sha[:8],
                    "full_sha": c.sha,
                    "message": c.commit.message.strip().split("\n")[0],
                    "author": c.commit.author.name or str(c.commit.author),
                    "time": c.commit.author.date.isoformat() + "Z" if c.commit.author.date else None,
                    "parents": parents,
                    "branches": [],
                }

        # Tag commits with branch names
        for bname, binfo in branch_info.items():
            if binfo["sha"] in dag:
                dag[binfo["sha"]]["branches"].append(bname)

        return {
            "branches": list(branch_info.values()),
            "commits": list(dag.values()),
        }
    except Exception as e:
        logger.error("Failed to build network: %s", e)
        return {"branches": [], "commits": []}


# ── Logs ──────────────────────────────────────────────────────────

_LOG_BUFFER: list[dict] = []
_LOG_BUFFER_MAX = 2000


class _LogHandler(logging.Handler):
    """Captures log records into an in-memory ring buffer for the dashboard."""
    def emit(self, record: logging.LogRecord):
        global _LOG_BUFFER
        entry = {
            "time": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "name": record.name,
            "badge": _infer_badge(record),
        }
        _LOG_BUFFER.append(entry)
        if len(_LOG_BUFFER) > _LOG_BUFFER_MAX:
            _LOG_BUFFER = _LOG_BUFFER[-_LOG_BUFFER_MAX:]


def _infer_badge(record: logging.LogRecord) -> str:
    msg = record.getMessage().lower()
    if any(w in msg for w in ("llm", "chat", "provider", "model")):
        return "llm"
    if any(w in msg for w in ("http", "get ", "post ", "webhook")):
        return "http"
    if any(w in msg for w in ("orchestrat", "iteration", "agent", "decision")):
        return "agent"
    if any(w in msg for w in ("tool", "classify", "register")):
        return "tool"
    return ""


# Install the custom handler at startup
_log_handler = _LogHandler()
_log_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_log_handler)


@app.get("/api/logs", tags=["System"])
async def get_logs(
    level: Optional[str] = None,
    limit: int = Query(200, ge=1, le=2000),
    since: Optional[str] = None,
):
    """Return recent log entries from the in-memory buffer.

    Optionally filter by level (info, warning, error, debug) and since
    (ISO timestamp)."""
    logs = list(_LOG_BUFFER)
    if level:
        logs = [l for l in logs if l["level"] == level.lower()]
    if since:
        logs = [l for l in logs if l["time"] >= since]
    return logs[-limit:]


# ── Health ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    prov = None
    model = None
    tools = 0
    if orchestrator is not None:
        prov = getattr(orchestrator.llm, "provider_name", None)
        model = getattr(orchestrator.llm, "model", None)
        tools = len(orchestrator._tool_handlers) if hasattr(orchestrator, "_tool_handlers") else 0
    return HealthResponse(
        uptime_seconds=time.time() - _start_time,
        llm_provider=prov,
        llm_model=model,
        tools_registered=tools,
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
            if _op_tools is not None:
                _op_tools.set_current_issue(issue.number)
            decision = orchestrator.run(issue)
            decision.issue_id = issue_id
            decision_id = store.save_decision(decision)
            logger.info("Agent decision saved: id=%d for issue #%d",
                        decision_id, issue.number)
        except Exception as e:
            logger.error("Agent execution failed: %s", e)
            fallback = Decision(
                issue_id=issue_id,
                classification=Classification.UNCLEAR,
                action=DecisionAction.NO_ACTION,
                explanation=f"Agent execution failed: {e}",
            )
            decision_id = store.save_decision(fallback)
    else:
        # No orchestrator — save a placeholder decision so dashboard shows data
        fallback = Decision(
            issue_id=issue_id,
            classification=Classification.UNCLEAR,
            action=DecisionAction.COMMENTED,
            explanation="No agent configured — LLM_PROVIDER=mock or set a real provider",
        )
        decision_id = store.save_decision(fallback)

    await _broadcast("issue", json.dumps({"issue_id": issue_id, "number": issue.number, "title": issue.title}))

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


@app.post("/api/issues/{issue_id}/reprocess", tags=["Issues"])
async def reprocess_issue(
    issue_id: int,
    store: PostgresStore = Depends(get_store),
):
    """Re-run the agent on an existing issue — replaces the latest decision."""
    data = store.get_issue(issue_id)
    if not data:
        raise HTTPException(status_code=404, detail="Issue not found")

    issue = Issue(
        id=data.get("gh_id", 0),
        repo=data["repo"],
        number=data["number"],
        title=data["title"],
        body=data.get("body", ""),
        author=data.get("author", "unknown"),
        labels=data.get("labels", []),
        url=data.get("url", ""),
    )

    decision_id = None
    if orchestrator is not None:
        try:
            if _op_tools is not None:
                _op_tools.set_current_issue(issue.number)
            decision = orchestrator.run(issue)
            decision.issue_id = issue_id
            decision_id = store.save_decision(decision)
            logger.info("Issue %d reprocessed: decision=%d", issue_id, decision_id)
        except Exception as e:
            logger.error("Reprocess failed for issue %d: %s", issue_id, e)
            fallback = Decision(
                issue_id=issue_id,
                classification=Classification.UNCLEAR,
                action=DecisionAction.NO_ACTION,
                explanation=f"Reprocess failed: {e}",
            )
            decision_id = store.save_decision(fallback)
    else:
        fallback = Decision(
            issue_id=issue_id,
            classification=Classification.UNCLEAR,
            action=DecisionAction.COMMENTED,
            explanation="No agent configured",
        )
        decision_id = store.save_decision(fallback)

    await _broadcast("issue", json.dumps({"issue_id": issue_id, "number": issue.number, "title": issue.title, "reprocessed": True}))

    return {"status": "ok", "decision_id": decision_id, "issue_id": issue_id}


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


# ── Event bus for SSE ──────────────────────────────────────────────

_subscribers: list[asyncio.Queue] = []
_sse_lock = asyncio.Lock()


async def _broadcast(event: str, data: str):
    async with _sse_lock:
        for q in _subscribers:
            await q.put(f"event: {event}\ndata: {data}\n\n")


async def _subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    async with _sse_lock:
        _subscribers.append(q)
    return q


async def _unsubscribe(q: asyncio.Queue):
    async with _sse_lock:
        if q in _subscribers:
            _subscribers.remove(q)


@app.get("/api/events", tags=["Dashboard"])
async def event_stream(request: Request):
    """Server-Sent Events endpoint — pushes real-time updates to the dashboard."""
    q = await _subscribe()
    async def generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30)
                    yield msg
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            await _unsubscribe(q)
    return StreamingResponse(generator(), media_type="text/event-stream")


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

    await _broadcast("eval", json.dumps({"run_id": run_id, "overall_score": report.overall_score}))

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
