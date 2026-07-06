import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from models.issue import Issue
from models.decision import Decision, Classification, DecisionAction

logger = logging.getLogger(__name__)


@dataclass
class PaginatedResult:
    items: list[dict]
    total: int
    page: int
    per_page: int
    pages: int


class PostgresStore:
    """Structured data store for issues, decisions, tool calls, and eval results.

    This is a relational store (PostgreSQL). It complements the vector store
    (Chroma) — the vector store holds codebase embeddings for semantic search,
    while Postgres holds operational data: issue metadata, agent decisions,
    tool call traces, and evaluation results.
    """

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.environ.get("DATABASE_URL", "")
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            import psycopg2
            self._conn = psycopg2.connect(self.database_url)
            self._conn.autocommit = False
            self._init_tables()
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Schema ─────────────────────────────────────────────────────────

    def _init_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS issues (
                    id SERIAL PRIMARY KEY,
                    gh_id BIGINT UNIQUE,
                    repo TEXT NOT NULL,
                    number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT DEFAULT '',
                    author TEXT DEFAULT 'unknown',
                    labels TEXT[] DEFAULT '{}',
                    url TEXT DEFAULT '',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id SERIAL PRIMARY KEY,
                    issue_id INTEGER NOT NULL REFERENCES issues(id) ON DELETE CASCADE,
                    classification TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    pr_url TEXT DEFAULT '',
                    explanation TEXT DEFAULT '',
                    session_id UUID,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id SERIAL PRIMARY KEY,
                    session_id UUID NOT NULL,
                    tool_name TEXT NOT NULL,
                    input_data JSONB DEFAULT '{}',
                    output_data JSONB DEFAULT '{}',
                    latency_ms INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id SERIAL PRIMARY KEY,
                    run_id UUID NOT NULL UNIQUE,
                    overall_score REAL,
                    classification_accuracy REAL,
                    action_accuracy REAL,
                    total_cases INTEGER,
                    failures INTEGER,
                    summary JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS eval_results (
                    id SERIAL PRIMARY KEY,
                    run_id UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
                    case_id TEXT NOT NULL,
                    category TEXT DEFAULT '',
                    classification_correct BOOLEAN,
                    action_correct BOOLEAN,
                    expected JSONB DEFAULT '{}',
                    actual JSONB DEFAULT '{}',
                    errors JSONB DEFAULT '[]',
                    judge_scores JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_issue_id ON decisions(issue_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id)
            """)
            self.conn.commit()

    # ── Issues ─────────────────────────────────────────────────────────

    def save_issue(self, issue: Issue) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO issues (gh_id, repo, number, title, body, author, labels, url)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (gh_id) DO UPDATE SET
                       title = EXCLUDED.title,
                       body = EXCLUDED.body,
                       labels = EXCLUDED.labels
                   RETURNING id""",
                (issue.id, issue.repo, issue.number, issue.title,
                 issue.body, issue.author, issue.labels, issue.url),
            )
            self.conn.commit()
            row_id = cur.fetchone()[0]
            logger.info("Saved issue #%d (pk=%d)", issue.number, row_id)
            return row_id

    def get_issue(self, issue_id: int) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, gh_id, repo, number, title, body, author, labels, url, created_at
                   FROM issues WHERE id = %s""",
                (issue_id,),
            )
            row = cur.fetchone()
            return self._row_to_dict(row, [
                "id", "gh_id", "repo", "number", "title", "body",
                "author", "labels", "url", "created_at",
            ]) if row else None

    def get_issue_by_number(self, repo: str, number: int) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, gh_id, repo, number, title, body, author, labels, url, created_at
                   FROM issues WHERE repo = %s AND number = %s""",
                (repo, number),
            )
            row = cur.fetchone()
            return self._row_to_dict(row, [
                "id", "gh_id", "repo", "number", "title", "body",
                "author", "labels", "url", "created_at",
            ]) if row else None

    def list_issues(
        self, page: int = 1, per_page: int = 20,
        repo: str | None = None, classification: str | None = None,
    ) -> PaginatedResult:
        conditions = []
        params: list[Any] = []
        if repo:
            conditions.append("repo = %s")
            params.append(repo)
        if classification:
            conditions.append(
                "id IN (SELECT issue_id FROM decisions WHERE classification = %s)"
            )
            params.append(classification)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM issues {where}", params)
            total = cur.fetchone()[0]

            offset = (page - 1) * per_page
            cur.execute(
                f"""SELECT i.id, i.gh_id, i.repo, i.number, i.title, i.body,
                           i.author, i.labels, i.url, i.created_at,
                           d.classification, d.action_taken, d.confidence,
                           d.pr_url, d.explanation, d.created_at AS decision_created_at
                    FROM issues i
                    LEFT JOIN LATERAL (
                        SELECT classification, action_taken, confidence, pr_url, explanation, created_at
                        FROM decisions WHERE issue_id = i.id
                        ORDER BY created_at DESC LIMIT 1
                    ) d ON TRUE
                    {where}
                    ORDER BY i.created_at DESC
                    LIMIT %s OFFSET %s""",
                params + [per_page, offset],
            )
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            items = [dict(zip(cols, r)) for r in rows]

        pages = max(1, (total + per_page - 1) // per_page)
        return PaginatedResult(
            items=items, total=total, page=page, per_page=per_page, pages=pages,
        )

    # ── Decisions ──────────────────────────────────────────────────────

    def save_decision(self, decision: Decision) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO decisions
                       (issue_id, classification, action_taken, confidence, pr_url, explanation, session_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    decision.issue_id,
                    decision.classification.value,
                    decision.action.value,
                    decision.confidence,
                    decision.pr_url or "",
                    decision.explanation,
                    str(uuid.uuid4()),
                ),
            )
            self.conn.commit()
            row_id = cur.fetchone()[0]
            logger.info("Saved decision #%d for issue #%d", row_id, decision.issue_id)
            return row_id

    def get_decision(self, decision_id: int) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, issue_id, classification, action_taken, confidence,
                          pr_url, explanation, session_id, created_at
                   FROM decisions WHERE id = %s""",
                (decision_id,),
            )
            row = cur.fetchone()
            return self._row_to_dict(row, [
                "id", "issue_id", "classification", "action_taken", "confidence",
                "pr_url", "explanation", "session_id", "created_at",
            ]) if row else None

    def get_decisions_for_issue(self, issue_id: int) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, issue_id, classification, action_taken, confidence,
                          pr_url, explanation, session_id, created_at
                   FROM decisions WHERE issue_id = %s
                   ORDER BY created_at DESC""",
                (issue_id,),
            )
            return [self._row_to_dict(r, [
                "id", "issue_id", "classification", "action_taken", "confidence",
                "pr_url", "explanation", "session_id", "created_at",
            ]) for r in cur.fetchall()]

    # ── Tool Calls ─────────────────────────────────────────────────────

    def log_tool_call(
        self, session_id: str, tool_name: str,
        input_data: dict, output_data: dict, latency_ms: int = 0,
    ):
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO tool_calls (session_id, tool_name, input_data, output_data, latency_ms)
                   VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)""",
                (session_id, tool_name,
                 json.dumps(input_data), json.dumps(output_data), latency_ms),
            )
            self.conn.commit()

    def get_tool_calls_for_session(self, session_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, session_id, tool_name, input_data, output_data, latency_ms, created_at
                   FROM tool_calls WHERE session_id = %s ORDER BY created_at""",
                (session_id,),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ── Eval Runs ──────────────────────────────────────────────────────

    def save_eval_run(
        self, run_id: str, overall_score: float,
        classification_accuracy: float, action_accuracy: float,
        total_cases: int, failures: int, summary: dict | None = None,
    ) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_runs
                       (run_id, overall_score, classification_accuracy, action_accuracy,
                        total_cases, failures, summary)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                   ON CONFLICT (run_id) DO UPDATE SET
                       overall_score = EXCLUDED.overall_score
                   RETURNING run_id""",
                (run_id, overall_score, classification_accuracy, action_accuracy,
                 total_cases, failures, json.dumps(summary or {})),
            )
            self.conn.commit()
            return cur.fetchone()[0]

    def save_eval_result(
        self, run_id: str, case_id: str, category: str,
        classification_correct: bool, action_correct: bool,
        expected: dict, actual: dict, errors: list[str],
        judge_scores: dict | None = None,
    ):
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_results
                       (run_id, case_id, category, classification_correct, action_correct,
                        expected, actual, errors, judge_scores)
                   VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)""",
                (run_id, case_id, category, classification_correct, action_correct,
                 json.dumps(expected), json.dumps(actual),
                 json.dumps(errors), json.dumps(judge_scores or {})),
            )
            self.conn.commit()

    def list_eval_runs(self, limit: int = 20) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, run_id, overall_score, classification_accuracy,
                          action_accuracy, total_cases, failures, summary, created_at
                   FROM eval_runs ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get_eval_run(self, run_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, run_id, overall_score, classification_accuracy,
                          action_accuracy, total_cases, failures, summary, created_at
                   FROM eval_runs WHERE run_id = %s""",
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            run = dict(zip(cols, row))

            cur.execute(
                """SELECT case_id, category, classification_correct, action_correct,
                          expected, actual, errors, judge_scores
                   FROM eval_results WHERE run_id = %s ORDER BY case_id""",
                (run_id,),
            )
            results_cols = [desc[0] for desc in cur.description]
            run["results"] = [dict(zip(results_cols, r)) for r in cur.fetchall()]
            return run

    # ── Dashboard / Stats ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        stats = {}
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM issues")
            stats["total_issues"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM decisions")
            stats["total_decisions"] = cur.fetchone()[0]

            cur.execute(
                """SELECT classification, COUNT(*) as cnt
                   FROM decisions GROUP BY classification ORDER BY cnt DESC"""
            )
            stats["classification_distribution"] = {
                row[0]: row[1] for row in cur.fetchall()
            }

            cur.execute(
                """SELECT action_taken, COUNT(*) as cnt
                   FROM decisions GROUP BY action_taken ORDER BY cnt DESC"""
            )
            stats["action_distribution"] = {
                row[0]: row[1] for row in cur.fetchall()
            }

            cur.execute(
                "SELECT COUNT(*) FROM decisions WHERE pr_url != '' AND pr_url IS NOT NULL"
            )
            stats["prs_opened"] = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM eval_runs"
            )
            stats["eval_runs"] = cur.fetchone()[0]

            cur.execute(
                "SELECT overall_score FROM eval_runs ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            stats["latest_eval_score"] = row[0] if row else None

        return stats

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row, keys: list[str]) -> dict:
        return {k: v for k, v in zip(keys, row)}
