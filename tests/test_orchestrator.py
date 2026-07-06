import pytest
from agent.orchestrator import Orchestrator, AgentState
from core.llm import LLMClient
from models.issue import Issue


@pytest.fixture
def orchestrator():
    llm = LLMClient(model="mock")
    return Orchestrator(llm=llm)


def test_orchestrator_initializes(orchestrator):
    assert orchestrator is not None


def test_orchestrator_accepts_issue(orchestrator):
    issue = Issue(
        id=1,
        repo="owner/repo",
        number=42,
        title="Test issue",
        body="This is a test",
        author="test-user",
    )
    decision = orchestrator.run(issue)
    assert decision is not None
    assert decision.issue_id == 1


def test_orchestrator_with_classify_tool():
    llm = LLMClient(model="mock")
    orch = Orchestrator(llm=llm)

    # Register a tool handler that always returns a classification
    def mock_classify(**kwargs):
        return {"classification": "bug", "confidence": 0.9, "explanation": "Mock classification"}

    orch.register_tool("classify_issue", mock_classify)

    issue = Issue(id=2, repo="owner/repo", number=100, title="Bug", body="Something broke", author="user")
    decision = orch.run(issue)

    assert decision is not None
