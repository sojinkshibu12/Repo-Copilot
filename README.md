<div align="center">

  <!-- ── Logo / Title ────────────────────────────── -->
  <img src="https://img.shields.io/badge/status-beta-6366f1?style=for-the-badge" alt="status"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-22c55e?style=for-the-badge&logo=python" alt="python"/>
  <img src="https://img.shields.io/badge/tests-88%20passed-22c55e?style=for-the-badge" alt="tests"/>
  <img src="https://img.shields.io/badge/license-MIT-a855f7?style=for-the-badge" alt="license"/>

  <br/>

  # Repo Copilot

  **AI agent that triages GitHub issues, searches your codebase, drafts fixes, and opens PRs — automatically.**

  <br/>

  [Features](#-features) •
  [Quick Start](#-quick-start) •
  [Architecture](#-architecture) •
  [Dashboard](#-dashboard) •
  [Configuration](#-configuration) •
  [Deployment](#-deployment)

  <br/>

</div>

---

## Demo

| Step | What happens |
|------|-------------|
| 🐛 Issue opened | User files a bug report on GitHub |
| 🤖 Agent classifies | LLM determines: bug, feature, duplicate, or unclear |
| 🔍 Codebase search | Agent reads files, greps, does semantic search |
| ✏️ Drafts a fix | Agent writes code changes |
| ✅ Runs tests | Verifies the fix doesn't break anything |
| 🔀 Opens PR | Creates a draft PR with the changes |
| 📊 Dashboard | See everything in real-time |

---

## Features

### Core
- **Multi-LLM support** — Anthropic Claude, OpenAI GPT, Google Gemini, OpenRouter, Ollama, or mock
- **Tool-calling loop** — Pure Python orchestration (no LangGraph dependency)
- **12 tools** — classify, read_file, grep, glob, semantic_search, run_tests, comment, label, branch, commit, PR
- **Vector code search** — ChromaDB with 3 embedding backends (OpenAI, Google, local, mock)
- **Code execution sandbox** — Docker container with network disabled for safe testing

### Observability
- **Structured JSON tracing** — Every LLM call, tool execution, and HTTP request logged with model, tokens, latency, errors
- **Swappable emitter** — Today: JSON logs. Tomorrow: swap one class for OpenTelemetry

### Evaluation
- **28 test cases** across 5 categories (bug-scoped, bug-ambiguous, feature, duplicate, unclear)
- **CI gate at 80% accuracy** — weighted scoring (40% classification + 40% action + 20% PR correctness)
- **Reports** in Markdown, JSON, HTML formats

### API
- FastAPI backend with 10 endpoints
- GitHub webhook signature verification (HMAC-SHA256)
- Paginated, filterable issue listing
- Dashboard served at `/` — zero build step

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional, for sandbox + Postgres)

### 1. Clone and install

```bash
git clone https://github.com/sojinkshibu12/repo-copilot.git
cd repo-copilot
pip install fastapi uvicorn[standard] psycopg2-binary pydantic httpx python-dotenv
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — at minimum set one LLM provider key:

| Variable | Where to get it |
|----------|----------------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) — needs `repo` scope |

### 3. Run (offline, no keys needed)

```bash
LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock uvicorn api.main:app --reload --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** — dashboard is live.

### 4. Send a test issue

```bash
curl -X POST http://localhost:8000/webhook/github \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: issues" \
  -d '{
    "action": "opened",
    "issue": {
      "id": 1, "number": 1,
      "title": "Login button broken on mobile",
      "body": "Clicking login on iPhone Safari does nothing.",
      "user": {"login": "testuser"},
      "labels": [],
      "html_url": "https://github.com/sojinkshibu12/Cartly/issues/1"
    },
    "repository": {"full_name": "sojinkshibu12/Cartly"}
  }'
```

Refresh the dashboard — issue appears instantly.

### 5. Run tests

```bash
EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock python -m pytest tests/ -v
```

### 6. Run eval suite

```bash
python scripts/run_evals.py --mock
```

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   GitHub     │────▶│  FastAPI     │────▶│  Agent       │
│   Webhook    │     │  (api/)      │     │  (agent/)    │
└─────────────┘     └──────┬───────┘     └──────┬───────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐     ┌──────────────┐
                    │  Postgres    │     │  LLM Client  │
                    │  (issues,    │     │  (core/llm)  │
                    │   decisions, │     │  6 providers │
                    │   tool_calls,│     └──────┬───────┘
                    │   eval_runs) │            │
                    └──────────────┘     ┌──────▼───────┐
                                          │  Tools       │
                    ┌──────────────┐     │  (read, grep, │
                    │  Vector DB   │     │   search,     │
                    │  (Chroma +   │     │   commit, PR) │
                    │   embeddings)│     └──────────────┘
                    └──────────────┘
```

### Key components

| Layer | Directory | Purpose |
|-------|-----------|---------|
| **Core** | `core/` | LLM client (6 providers), embedding engine (3 backends), tool schemas, tracer |
| **Agent** | `agent/` | Pure Python orchestration loop, session management, decision builder |
| **Storage** | `storage/` | PostgreSQL store (5 tables), Chroma vector store with in-memory fallback |
| **API** | `api/` | FastAPI (10 endpoints), webhook handler, dependency injection, Pydantic schemas |
| **Eval** | `evals/` | 28 test cases, scorer with weighted metrics, LLM judge, report generator |
| **Dashboard** | `dashboard/` | Single-page HTML/JS served at `/` (Chart.js, auto-refresh) |

---

## Dashboard

The dashboard is served at `GET /` — open `http://localhost:8000` after starting the API.

- **Stats cards** — issues processed, decisions made, PRs opened, latest eval score
- **Recent issues** — table with classification, action, PR link, timestamp
- **Eval history** — Chart.js line chart with scores over time
- **Distribution bars** — classification and action breakdowns
- **Run Eval button** — triggers evaluation inline

---

## Configuration

All configuration via environment variables (see `.env.example`):

### LLM Provider

```env
LLM_PROVIDER=openai          # anthropic | openai | google | openrouter | ollama | mock
LLM_MODEL=gpt-4o             # or claude-sonnet-4, gemini-2.0-flash, etc.
```

### Embedding Provider

```env
EMBEDDING_PROVIDER=google    # openai | google | local | mock
```

### Tracing

```env
TRACE_EMITTER=json           # json | console | none
```

Every LLM call, tool execution, and HTTP request is logged as structured JSON:

```json
{
  "name": "llm.chat",
  "kind": "llm",
  "duration_ms": 1234.5,
  "attributes": {
    "model": "gpt-4o",
    "input_tokens": 450,
    "output_tokens": 120,
    "num_tool_calls": 2
  }
}
```

---

## Deployment

### Docker

```bash
docker compose up --build
```

Starts Postgres 16 + Chroma + API + sandbox container.

### Render

Push to GitHub, then use the included `render.yaml` Blueprint for one-click deploy.

### Fly.io

```bash
fly launch --image .
fly secrets set DATABASE_URL=... ANTHROPIC_API_KEY=...
fly deploy
```

---

## Project Structure

```
repo-copilot/
├── api/              # FastAPI app, routes, webhook, schemas, DI
├── agent/            # Orchestrator, session, tool handlers
│   └── tools/        # Retrieval tool set
├── core/             # LLM client, embeddings, tools, tracer
├── models/           # Issue, Decision dataclasses
├── storage/          # Postgres store, Chroma vector store
├── evals/            # Test cases, scorer, runner, judge, reports
├── scripts/          # CLI tools (run_evals, seed, index)
├── dashboard/        # Single-page HTML dashboard
├── tests/            # 88 tests across 8 test files
├── .github/          # CI workflow (pytest, coverage, eval gate)
├── Dockerfile        # Multi-stage production build
├── docker-compose.yml
├── render.yaml       # Render Blueprint
├── fly.toml          # Fly.io config
└── .env.example      # All config vars documented
```

---

## Testing

```bash
# All tests (offline, mock providers)
EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock python -m pytest tests/ -v

# With coverage
EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock python -m pytest tests/ --cov=. --cov-report=term

# Run eval suite (mock mode)
python scripts/run_evals.py --mock --report --format markdown
```

---

## License

MIT License — see [LICENSE](LICENSE).

*Want a different license? Just change the file.*

---

<div align="center">
  Built with Python, FastAPI, and ❤️
  <br/>
  <sub>Every line explainable in an interview — no black boxes.</sub>
</div>
