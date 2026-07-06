# Repo Copilot — Architecture Documentation

> An agentic GitHub issue triager and PR drafter powered by LLM tool-calling.

---

## 1. System Overview

Repo Copilot watches GitHub repositories for incoming issues, autonomously triages them (bug, feature, duplicate, unclear), explores the codebase for well-scoped bugs, writes fixes, opens draft PRs, and logs every decision for evaluation.

```ascii
┌─────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│  GitHub      │────▶│  FastAPI         │────▶│  Agent Orchestrator  │
│  Webhook     │     │  (webhooks +     │     │  (tool-calling loop) │
│              │     │   dashboard API) │     │                      │
└─────────────┘     └─────────────────┘     └───────┬──────────────┘
                                                     │
                      ┌──────────────────────────────┼──────────────────────────┐
                      │               ┌──────────────▼──────────────┐          │
                      │               │       Tool Layer             │          │
                      │               │  ┌──────┬──────┬──────────┐ │          │
                      │               │  │ Code │  GH  │ Sandbox │ │          │
                      │               │  │Explor│ API │ Executor│ │          │
                      │               │  └──────┴──────┴──────────┘ │          │
                      │               └─────────────────────────────┘          │
                      │                                                       │
                      │  ┌──────────┐  ┌────────────┐  ┌──────────────────┐  │
                      │  │ Vector DB │  │  Postgres  │  │  Eval Harness    │  │
                      │  │ (Chroma)  │  │ (decisions │  │  (test suite +   │  │
                      │  │           │  │  + history)│  │   scoring)       │  │
                      │  └──────────┘  └────────────┘  └──────────────────┘  │
                      └───────────────────────────────────────────────────────┘
```

---

## 2. Layer-by-Layer Architecture

### 2.1 Model Layer — `core/llm.py`

**Choice:** Provider-agnostic LLM client supporting Anthropic, OpenAI, Google Gemini, OpenRouter, Ollama, and Mock.

- All agent-model interaction goes through a single `LLMClient` abstraction.
- Tools are declared as JSON schema in `core/tools.py` and passed via the `tools` parameter — no prompt-based function-call parsing.
- Supports token accounting (`usage` field from every response) for cost tracking.

```
LLMClient
├── __init__(model_name, api_key, max_tokens, temperature)
├── chat(messages, tools, system_prompt) → Response(tool_calls, content, usage)
└── _build_request(messages, tools, system)
```

**Why not hand-rolled prompt parsing:** Native tool-calling guarantees structured output, reduces prompt engineering surface, and is the pattern every production LLM app uses.

---

### 2.2 Orchestration — `agent/orchestrator.py`

**Version 1:** A pure Python loop. No LangGraph, no framework.

```
Orchestrator
├── run(issue: Issue) → Decision
│   1. Classify issue (bug / feature / duplicate / unclear)
│   2. If bug + well-scoped:
│      a. Search codebase (grep + vector)
│      b. Read relevant files
│      c. Draft fix
│      d. Run tests in sandbox
│      e. If tests pass → open draft PR
│      f. If tests fail → comment with diff
│   3. If ambiguous → comment asking clarifying questions
│   4. Log decision + score
└── _step(messages, available_tools) → ToolCall | str
```

**State machine (implicit):**

```
                 ┌─────────────────────────────┐
                 │       Issue Received          │
                 └─────────────┬───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Classify Issue     │
                    └──────┬──────┬───────┘
                           │      │
              ┌────────────┘      └────────────┐
              ▼                                 ▼
     ┌─────────────────┐             ┌───────────────────┐
     │  Bug + Scoped    │             │  Feature/Duplicate │
     │                  │             │  /Unclear          │
     └────────┬─────────┘             └────────┬──────────┘
              │                                │
     ┌────────▼────────┐              ┌────────▼─────────┐
     │ Explore Codebase │              │ Post Comment /   │
     │ Draft Fix        │              │ Apply Label      │
     │ Run Tests        │              └──────────────────┘
     │ Open Draft PR    │
     └─────────────────┘
```

**Version 2 (future):** Rebuild with LangGraph to add checkpointing, human-in-the-loop, and branching — showing framework fluency on top of fundamentals.

---

### 2.3 Tool Layer — `agent/tools/`

Each tool is a function with a JSON schema declaration.

| Tool | File | Schema | Description |
|------|------|--------|-------------|
| `read_file` | `tools/code.py` | `{path: str, offset?: int, limit?: int}` | Read file contents |
| `search_code` | `tools/code.py` | `{pattern: str, path?: str}` | Regex search across codebase |
| `grep` | `tools/code.py` | `{pattern: str, include?: str}` | Fast content search |
| `glob` | `tools/code.py` | `{pattern: str}` | File pattern matching |
| `semantic_search` | `tools/retrieval.py` | `{query: str, k: int}` | Vector search over codebase embeddings |
| `comment_on_issue` | `tools/github.py` | `{issue_number: int, body: str}` | Post a comment on the issue |
| `add_label` | `tools/github.py` | `{issue_number: int, label: str}` | Add a label to the issue |
| `create_branch` | `tools/git.py` | `{branch_name: str, base: str}` | Create and switch to a new branch |
| `commit_changes` | `tools/git.py` | `{message: str, files: list[str]}` | Stage and commit changes |
| `open_draft_pr` | `tools/github.py` | `{title: str, body: str, head: str, base: str}` | Open a draft pull request |
| `run_command` | `tools/sandbox.py` | `{command: str, timeout: int}` | Run a shell command in sandbox |
| `run_tests` | `tools/sandbox.py` | `{test_command: str}` | Run project tests in sandbox |

**Guardrails:**
- `run_command` is restricted to a Docker container or `nsjail` sandbox.
- Network access is blocked inside the sandbox.
- `commit_changes` only stages files within the repo.
- All GH API calls require explicit `issue_number` scope (cannot accidentally modify other issues).

---

### 2.4 Code & Repo Interaction — `core/github.py`, `core/git.py`

**GitHub API (PyGithub):**
```python
class GitHubClient:
    def get_issue(self, repo_full_name: str, number: int) -> Issue
    def comment(self, issue: Issue, body: str) -> Comment
    def add_label(self, issue: Issue, label: str) -> None
    def create_pr(self, repo, title, body, head, base, draft=True) -> PR
    def get_repo(self, full_name: str) -> Repository
```

**Local Git (GitPython + subprocess):**
```python
class GitClient:
    def clone(self, url: str, path: Path) -> None
    def checkout_branch(self, name: str) -> None
    def stage_files(self, paths: list[str]) -> None
    def commit(self, message: str) -> None
    def push(self, remote: str, branch: str) -> None
    def diff(self, base: str, head: str) -> str
```

**Security boundary:** Git operations run on a cloned fork (never the original repo). The agent only pushes to a fork, then opens a PR from fork → original upstream.

---

### 2.5 Storage Layer

#### Structured Data — PostgreSQL (`storage/postgres.py`)

| Table | Purpose |
|-------|---------|
| `issues` | Ingested issue metadata (GH ID, title, body, repo, timestamps) |
| `decisions` | Agent classification + action taken (bug/feature/duplicate/unclear) |
| `eval_results` | Per-test-case pass/fail/score from eval runs |
| `tool_calls` | Every tool invocation (for observability and replay) |

Schema for `decisions`:
```sql
CREATE TABLE decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id INTEGER NOT NULL REFERENCES issues(id),
    classification TEXT NOT NULL CHECK (classification IN ('bug', 'feature', 'duplicate', 'unclear')),
    confidence REAL,
    action_taken TEXT,
    pr_url TEXT,
    explanation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### Vector Data — Chroma (`storage/vector_store.py`)

- Stores chunked embeddings of the target repository's codebase.
- Supports automatic embedding via pluggable `EmbeddingEngine` or pre-computed vectors.
- Used for `semantic_search`: "find the function that handles authentication" → returns relevant file paths + line numbers.
- In-memory fallback (`_InMemoryCollection`) when Chroma is unavailable — supports keyword search and approximate vector search.
- Re-indexed on repo changes (optional webhook-triggered refresh).

```python
class VectorStore:
    def add_documents(self, documents: list[dict], embeddings: list[list[float]] | None = None) -> int
    def search(self, query: str, k: int = 5) -> list[dict]
    def search_by_file(self, filepath: str, k: int = 3) -> list[dict]
    def count(self) -> int
    def delete_collection(self) -> None
```

#### Embedding Engine — `core/embeddings.py`

Provider-agnostic embedding engine that supports three backends:

| Provider | Class | Model | Use Case |
|----------|-------|-------|----------|
| OpenAI | `_OpenAIEmbedder` | `text-embedding-3-small` (1536d) | Production — best quality |
| Local | `_LocalEmbedder` | `all-MiniLM-L6-v2` (384d) | Cost-conscious, offline |
| Mock | `_MockEmbedder` | `mock` (4d) | Tests, dev without API keys |

Configurable via `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` env vars.
```python
class EmbeddingEngine:
    def embed(self, text: str) -> EmbeddingResult
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]
```

#### Retrieval Tool — `agent/tools/retrieval.py`

Bridges the agent orchestrator to the vector store. Provides three tool handlers:

| Tool | Handler | Description |
|------|---------|-------------|
| `semantic_search(query, k)` | `RetrievalToolSet.semantic_search` | Find code by natural language query |
| `search_by_file(path, k)` | `RetrievalToolSet.search_by_file` | Retrieve all chunks for a given file |
| `index_status()` | `RetrievalToolSet.index_status` | Report index health to the agent |

Handlers are registered with the `Orchestrator` via `register_tool_set()`, making them available as LLM tool calls. The agent is instructed in the system prompt to use `semantic_search` **first** before falling back to regex-based `grep`.

---

### 2.6 Evaluation Harness — `evals/`

The evaluation suite is the **differentiator** of this project. It measures the agent's
classification accuracy, action correctness, and output quality against a curated
set of 28 test cases spanning 5 categories.

**Structure:**
```
evals/
├── test_cases/
│   ├── 001-bug-scoped.json        # 12 well-scoped bugs
│   ├── 013-bug-ambiguous.json     # 3 ambiguous bugs
│   ├── 016-feature.json           # 5 feature requests
│   ├── 021-duplicate.json         # 3 duplicate reports
│   ├── 024-unclear.json           # 5 unclear issues
│   └── ...                       28 cases total
├── scorer.py                      # Scoring engine with per-category breakdown
├── runner.py                      # Drives agent (or mock) through every case
├── judge_llm.py                   # LLM-as-judge for subjective outputs
└── report.py                      # Generates Markdown, JSON, and HTML reports
```

**Test case categories:**

| Category | Count | Description |
|----------|-------|-------------|
| `bug-scoped` | 12 | Well-defined bugs the agent should fix (typo, off-by-one, null check, etc.) |
| `bug-ambiguous` | 3 | Bugs lacking reproduction steps — agent should ask clarifying questions |
| `feature` | 5 | Feature requests — agent should comment, not open a PR |
| `duplicate` | 3 | Clearly marked duplicates — agent should reference original issue |
| `unclear` | 5 | Vague or nonsensical reports — agent should ask for more info |

Each test case:
```json
{
  "id": "001-bug-scoped",
  "category": "bug-scoped",
  "issue": {
    "title": "Typo in README: 'instalation' should be 'installation'",
    "body": "On line 42 of README.md, there's a typo: ...",
    "repo": "owner/repo"
  },
  "expected": {
    "classification": "bug",
    "action": "opened_pr",
    "requires_comment": false
  }
}
```

**Scoring:**

| Metric | Weight | Definition |
|--------|--------|------------|
| Classification Accuracy | 40% | % of cases where predicted class matches expected |
| Action Accuracy | 40% | % where the chosen action matches expected |
| PR Correctness | 20% | Average LLM-judge score on PR descriptions that were opened |
| Overall | 100% | Weighted composite |

**Per-category breakdown** — the report shows accuracy per category, revealing whether
the agent is weaker on ambiguous bugs vs. duplicates, for example.

**LLM-as-judge** (`judge_llm.py`):

Separate LLM calls rate the quality of:
- **PR descriptions** — clarity, safety, completeness, testing (1-5 scale)
- **Comments** — helpfulness, tone, accuracy (1-5 scale)
- **Classifications** — reasonableness, confidence calibration (1-5 scale)

Known limitations documented inline:
1. **Position bias** — judge favours content appearing earlier
2. **Self-enhancement bias** — same-model judge rates own outputs higher
3. **Verbosity bias** — longer outputs score higher regardless of quality
4. **Calibration drift** — scores are relative, not absolute
5. **Instruction sensitivity** — small prompt changes produce large score swings

**Runner** (`runner.py`):

Two modes:
- **Agent mode** — pass a callable `(title, body) → dict`; runner iterates all cases
- **Replay mode** — pass pre-recorded results JSON for offline scoring

Mock mode returns perfect answers (classification + action both 100%) for baseline validation.

**CI gate** (`.github/workflows/test.yml`):

The eval suite runs in CI on every push/PR with `--min-accuracy 0.8`, so accuracy below
80% fails the build. Reports are uploaded as build artifacts.

```
# Example eval output:
============================================================
  EVAL REPORT  ·  20260706T085912
============================================================
  Overall score:         100.0%
  Classification acc:    100.0%  (28/28)
  Action accuracy:       100.0%  (28/28)
  PR correctness:        100.0%
  Total cases:           28
  Failures:              0

  Per-category breakdown:
    bug-ambiguous          3 cases  cls: 100%  act: 100%
    bug-scoped            12 cases  cls: 100%  act: 100%
    duplicate              3 cases  cls: 100%  act: 100%
    feature                5 cases  cls: 100%  act: 100%
    unclear                5 cases  cls: 100%  act: 100%
============================================================
```

---

### 2.7 API Layer — `api/`

**FastAPI application:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/webhook/github` | POST | Receives `issues.opened` events from GitHub |
| `/api/issues` | GET | List processed issues with decisions |
| `/api/issues/{id}` | GET | Single issue detail + agent reasoning trace |
| `/api/decisions/{id}` | GET | Agent decision for a given issue |
| `/api/eval/run` | POST | Trigger a new eval run |
| `/api/eval/results` | GET | Eval scores over time (for dashboard chart) |
| `/api/health` | GET | Health check |

```python
@app.post("/webhook/github")
async def handle_webhook(event: GithubEvent):
    if event.action == "opened" and "issue" in event.payload:
        issue = parse_issue(event.payload)
        decision = await orchestrator.run(issue)
        return {"decision_id": decision.id, "status": "processed"}
```

---

### 2.8 Dashboard — `dashboard/`

A lightweight Next.js (or single React + Vite) page.

**Views:**
1. **Recent Activity** — Table of recent issues, classification, action taken, timestamp.
2. **Decision Detail** — Expandable trace of tool calls, model reasoning, final output.
3. **Eval Dashboard** — Line chart of accuracy over time (plotly or recharts).
4. **Failed Cases** — List of eval failures with expected vs. actual.

**API consumption:**
```typescript
// /api/issues returns:
interface IssueSummary {
  id: number;
  title: string;
  classification: "bug" | "feature" | "duplicate" | "unclear";
  action: "opened_pr" | "commented" | "labeled";
  pr_url?: string;
  confidence: number;
  processed_at: string;
}
```

---

### 2.9 Deployment & Observability — `scripts/`, `deploy/`

**Deployment targets:**
- FastAPI backend → Render / Fly.io / Railway (single cheap service)
- Dashboard → Vercel / GitHub Pages
- Postgres → Render managed DB / Supabase free tier
- Vector DB → Local persistence on the backend service (Chroma is file-based)

**CI/CD (`.github/workflows/`):**

| Workflow | Trigger | Action |
|----------|---------|--------|
| `test.yml` | PR to main | Run evals, require accuracy ≥ 80% |
| `deploy.yml` | Push to main | Deploy backend + dashboard |
| `index-repo.yml` | Manual / cron | Re-embed target codebase into vector store |

**Observability (structured JSON logging):**
```json
{
  "timestamp": "2026-07-06T12:00:00.000Z",
  "event": "tool_call",
  "session_id": "abc-123",
  "tool": "search_code",
  "input": {"pattern": "def authenticate", "path": "src/"},
  "output": {"matches": 3, "files": ["src/auth.py"]},
  "latency_ms": 234,
  "tokens_used": 152
}
```

Logs are written to `stdout` (for cloud log aggregation) and optionally to the `tool_calls` table for replay/debug.

---

## 3. Data Flow — End-to-End

```
1. GitHub sends `issues.opened` webhook → FastAPI `/webhook/github`
2. API parses issue, stores in `issues` table
3. Orchestrator.run(issue) begins:
   a. LLM classifies issue (tool: classify)
   b. If bug:
      - Agent calls semantic_search("...")  → vector search finds relevant files
      - Agent calls read_file(...)           → reads code from returned files
      - Agent calls grep(...), glob(...)     → narrows scope with regex
      - Agent calls run_tests(...)           → verifies no regressions
      - Agent calls create_branch(...)       → creates fix branch
      - Agent calls commit_changes(...)      → commits fix
      - Agent calls open_draft_pr(...)       → opens draft PR
   c. If unclear:
      - Agent calls comment_on_issue(...)    → asks clarifying question
      - Agent calls add_label(...)           → labels "needs-triage"
4. Decision is persisted to `decisions` table
5. Webhook returns 200 with decision_id
```

---

## 4. Safety & Guardrails

| Concern | Mitigation |
|---------|------------|
| Arbitrary code execution | All shell commands run in Docker container with no network, read-only repo mount |
| Accidental git push to main | Agent only pushes to feature branches; main branch protected via GH branch rules |
| Malicious issue body | System prompt instructs agent to ignore prompt injection attempts; tool calls are bounded by schema |
| API key leakage | All keys injected via env vars, never logged or exposed to the model |
| Runaway costs | Per-session token limit (`max_tokens` per response, session-level cap) |
| Duplicate PRs | Orchestrator checks for existing open PRs before creating a new one |
| Test flakiness | `run_tests` retries once before declaring failure |

---

## 5. Project Structure

```
repo-copilot/
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py        # Main agent loop
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── code.py            # read_file, search_code, grep, glob
│   │   ├── github.py          # comment_on_issue, add_label, open_draft_pr
│   │   ├── git.py             # create_branch, commit_changes
│   │   ├── retrieval.py       # semantic_search
│   │   └── sandbox.py         # run_command, run_tests
│   └── state.py               # Agent state / context accumulator
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app with routes
│   ├── webhook.py             # GitHub webhook handler
│   └── schemas.py             # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── llm.py                 # Multi-provider LLMClient (6 backends)
│   ├── embeddings.py          # Multi-provider EmbeddingEngine (3 backends)
│   ├── tools.py               # Tool schema definitions
│   ├── github.py              # GitHubClient (PyGithub)
│   └── git.py                 # GitClient (GitPython)
├── evals/
│   ├── __init__.py
│   ├── test_cases/            # 28 JSON test cases (5 categories)
│   ├── scorer.py              # Scoring engine with per-category breakdown
│   ├── runner.py              # Drives agent (or mock) through all cases
│   ├── judge_llm.py           # LLM-as-judge for subjective outputs
│   └── report.py              # Markdown / JSON / HTML report generator
├── models/
│   ├── __init__.py
│   ├── issue.py               # Issue dataclass
│   └── decision.py            # Decision dataclass
├── storage/
│   ├── __init__.py
│   ├── postgres.py            # Postgres repository
│   └── vector_store.py        # Chroma vector store
├── dashboard/
│   ├── pages/
│   ├── components/
│   └── package.json
├── scripts/
│   ├── seed_test_cases.py     # Generate initial test cases
│   ├── run_evals.py           # CLI for running evals
│   ├── index_codebase.py      # Build vector index for a repo
│   └── sandbox/
│       └── Dockerfile         # Sandbox container definition
├── docs/
│   └── REPO_OWNERSHIP.md      # How repo-copilot sees code ownership
├── .github/
│   └── workflows/
│       ├── test.yml           # CI — run evals
│       └── deploy.yml         # CD — deploy to production
├── tests/
│   ├── test_orchestrator.py
│   ├── test_tools.py
│   ├── test_llm.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retrieval_tool.py
│   └── conftest.py
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml         # Local dev: API + Postgres + Chroma
└── README.md
```

---

## 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure Python loop first, LangGraph later | Shows depth (I can explain every line) then breadth (I know the ecosystem) |
| Native tool-calling via API | Industry standard; no fragile prompt parsing |
| Chroma over pgvector | Simpler local dev; can migrate to pgvector later |
| Pluggable EmbeddingEngine (OpenAI/local/mock) | Cost-conscious local dev in CI, production-grade OpenAI/Azure in deployment |
| Vector store with in-memory fallback | Tests and offline dev work without Chroma installed |
| Eval harness before vector DB | Prevents building on shaky foundations — evals validate the agent works |
| Structured JSON logging | Cheap but production-ready; plug-and-play with any observability backend |
| Docker sandbox for code execution | Essential safety signal for security-conscious teams |
| Fork-based PR flow | Never touches protected branches; respects repo permissions |
| LLM-as-judge for subjective evals | Demonstrates awareness of this technique and its limitations |

---

## 7. Future Enhancements

- **Human-in-the-loop:** Pause before opening PRs for human approval via a Slack / Discord notification.
- **Multi-repo support:** Watch multiple repos with per-repo configuration.
- **Self-healing:** If a PR is closed without merging, the agent can re-analyze and update the PR.
- **Cost-aware routing:** Use cheaper/smaller models for classification, expensive models only for code generation.
- **Continuous learning:** Fine-tune classification decisions based on human feedback on past decisions.

---

## 8. Quick Start

```bash
# Clone
git clone https://github.com/yourusername/repo-copilot
cd repo-copilot

# Install
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run locally
docker-compose up -d          # Postgres + Chroma
python -m api.main            # FastAPI on :8000

# Index a codebase
python scripts/index_codebase.py --repo owner/repo

# Run evals
python scripts/run_evals.py

# Deploy
# Push to main → GitHub Actions deploys to Render/Fly.io
```
