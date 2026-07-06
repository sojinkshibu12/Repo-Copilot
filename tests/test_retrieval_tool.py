import pytest
import tempfile

from agent.tools.retrieval import RetrievalToolSet
from storage.vector_store import VectorStore
from core.embeddings import EmbeddingEngine


@pytest.fixture
def toolset():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(persist_dir=tmp)
        store.embedding_engine = EmbeddingEngine(provider="mock")
        store.add_documents([
            {"id": "1", "text": "def authenticate_user(username, password): return True",
             "metadata": {"file": "src/auth.py", "start_line": 1, "end_line": 3}},
            {"id": "2", "text": "class UserModel: pass",
             "metadata": {"file": "src/models/user.py", "start_line": 1, "end_line": 5}},
            {"id": "3", "text": "def handle_login(request): return authenticate_user(...)",
             "metadata": {"file": "src/routes/login.py", "start_line": 10, "end_line": 20}},
            {"id": "4", "text": "DATABASE_URL = postgres://localhost",
             "metadata": {"file": "src/config.py", "start_line": 5, "end_line": 5}},
        ])
        ts = RetrievalToolSet(vector_store=store)
        ts.index_root = "/tmp/test-repo"
        yield ts


class TestRetrievalToolSet:
    def test_semantic_search_returns_formatted_output(self, toolset):
        result = toolset.semantic_search("user authentication", k=2)
        assert isinstance(result, str)
        assert "auth" in result.lower() or "login" in result.lower()
        assert "Top 2" in result
        assert "distance=" in result

    def test_semantic_search_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(persist_dir=tmp)
            ts = RetrievalToolSet(vector_store=store)
            result = ts.semantic_search("anything")
            assert "empty" in result.lower()
            assert "index" in result.lower()

    def test_search_by_file(self, toolset):
        result = toolset.search_by_file("src/auth.py")
        assert "auth.py" in result
        assert "chunk" in result.lower()

    def test_search_by_file_missing(self, toolset):
        result = toolset.search_by_file("nonexistent.py")
        assert "not found" in result.lower() or "no" in result.lower()

    def test_index_status(self, toolset):
        result = toolset.index_status()
        assert "4" in result
        assert "index root" in result.lower()

    def test_get_tool_handlers(self, toolset):
        handlers = toolset.get_tool_handlers()
        assert "semantic_search" in handlers
        assert "search_by_file" in handlers
        assert "index_status" in handlers
        assert callable(handlers["semantic_search"])
        assert callable(handlers["search_by_file"])
        assert callable(handlers["index_status"])
