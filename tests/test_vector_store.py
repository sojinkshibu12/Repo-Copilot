import pytest
import tempfile
from pathlib import Path

from storage.vector_store import VectorStore
from core.embeddings import EmbeddingEngine


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(persist_dir=tmp)
        vs.embedding_engine = EmbeddingEngine(provider="mock")
        yield vs


class TestVectorStoreBasic:
    def test_empty_store_count(self, store):
        assert store.count() == 0

    def test_add_documents(self, store):
        docs = [
            {"id": "1", "text": "def hello(): return 'world'", "metadata": {"file": "greeting.py", "start_line": 1}},
            {"id": "2", "text": "def add(a, b): return a + b", "metadata": {"file": "math.py", "start_line": 10}},
        ]
        store.add_documents(docs)
        assert store.count() == 2

    def test_search_returns_results(self, store):
        store.add_documents([
            {"id": "1", "text": "def authenticate_user(): pass", "metadata": {"file": "auth.py"}},
            {"id": "2", "text": "def login(): pass", "metadata": {"file": "auth.py"}},
            {"id": "3", "text": "class Database: pass", "metadata": {"file": "db.py"}},
        ])
        results = store.search("login authentication", k=2)
        assert len(results) > 0
        assert results[0]["id"] in ("1", "2")

    def test_search_with_precomputed_embeddings(self, store):
        docs = [
            {"id": "a", "text": "function foo() {}", "metadata": {"file": "foo.js"}},
            {"id": "b", "text": "function bar() {}", "metadata": {"file": "bar.js"}},
        ]
        engine = store.embedding_engine
        embeddings = [r.embedding for r in engine.embed_batch([d["text"] for d in docs])]
        store.add_documents(docs, embeddings=embeddings)
        assert store.count() == 2

    def test_search_by_file(self, store):
        store.add_documents([
            {"id": "chunk1", "text": "import os", "metadata": {"file": "main.py", "start_line": 1}},
            {"id": "chunk2", "text": "def run(): pass", "metadata": {"file": "main.py", "start_line": 5}},
            {"id": "chunk3", "text": "import sys", "metadata": {"file": "utils.py", "start_line": 1}},
        ])
        results = store.search_by_file("main.py")
        assert len(results) == 2


class TestInMemoryFallback:
    def test_in_memory_keyword_search(self, store):
        docs = [
            {"id": "x", "text": "authentication handler middleware", "metadata": {"file": "middleware.py"}},
            {"id": "y", "text": "database connection pool", "metadata": {"file": "db.py"}},
            {"id": "z", "text": "login route handler", "metadata": {"file": "routes.py"}},
        ]
        store.add_documents(docs)
        results = store.search("database", k=2)
        assert any("db.py" in r["metadata"].get("file", "") for r in results)

    def test_delete_collection(self, store):
        store.add_documents([{"id": "1", "text": "test", "metadata": {}}])
        assert store.count() > 0
        store.delete_collection()
        assert store.count() == 0
