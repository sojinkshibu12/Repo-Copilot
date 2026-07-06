import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.embeddings import EmbeddingEngine, EmbeddingResult

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database for codebase embeddings using Chroma (local-first).

    Supports automatic embedding via an EmbeddingEngine when documents are added
    without pre-computed vectors.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_engine: "EmbeddingEngine | None" = None,
        collection_name: str = "codebase",
    ):
        self.persist_dir = persist_dir or os.environ.get(
            "CHROMA_PERSIST_DIR", "./data/chroma"
        )
        self.collection_name = collection_name
        self._embedding_engine = embedding_engine
        self._collection = None

    @property
    def embedding_engine(self) -> "EmbeddingEngine | None":
        if self._embedding_engine is None:
            try:
                from core.embeddings import EmbeddingEngine
                self._embedding_engine = EmbeddingEngine()
            except ImportError:
                pass
        return self._embedding_engine

    @embedding_engine.setter
    def embedding_engine(self, engine: "EmbeddingEngine"):
        self._embedding_engine = engine

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_dir)
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("Connected to Chroma collection '%s' at %s",
                            self.collection_name, self.persist_dir)
            except ImportError:
                logger.warning("chromadb not installed — using in-memory fallback")
                self._collection = _InMemoryCollection()
        return self._collection

    def add_documents(
        self,
        documents: list[dict],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """Add documents to the vector store.

        Args:
            documents: List of dicts with keys "id", "text", "metadata".
            embeddings: Optional pre-computed embeddings. If None, computed
                       via the configured EmbeddingEngine.

        Returns:
            Number of documents added.
        """
        ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]

        if embeddings is not None:
            self.collection.add(
                ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
            )
            logger.info("Added %d documents (pre-computed embeddings)", len(documents))
        elif self.embedding_engine is not None:
            results = self.embedding_engine.embed_batch(texts)
            computed = [r.embedding for r in results]
            self.collection.add(
                ids=ids, embeddings=computed, metadatas=metadatas, documents=texts
            )
            total_tokens = sum(r.tokens_used for r in results)
            logger.info(
                "Added %d documents (embedder=%s, tokens=%d)",
                len(documents), self.embedding_engine.model_name, total_tokens,
            )
        else:
            self.collection.add(ids=ids, metadatas=metadatas, documents=texts)
            logger.info("Added %d documents (no embeddings — relying on Chroma default)", len(documents))

        return len(documents)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Semantic search over stored documents.

        Passes the query text to the underlying store. Chroma handles its
        own embedding; the in-memory fallback uses keyword search.
        """
        results = self.collection.query(query_texts=[query], n_results=k)

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results.get("distances") else 0.0,
            })
        return formatted

    def search_by_file(self, filepath: str, k: int = 3) -> list[dict]:
        """Search for documents matching a specific file path."""
        try:
            results = self.collection.get(
                where={"file": filepath},
                limit=k,
            )
            formatted = []
            for i in range(len(results["ids"])):
                formatted.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    "distance": 0.0,
                })
            return formatted
        except Exception:
            return []

    def count(self) -> int:
        """Return the number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def delete_collection(self):
        if self._collection is not None:
            try:
                self._collection.delete()
            except Exception:
                pass
            self._collection = None
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            client.delete_collection(self.collection_name)
        except ImportError:
            pass
        logger.info("Deleted collection '%s'", self.collection_name)


class _InMemoryCollection:
    """Fallback when Chroma is not available. No actual vector search."""

    def __init__(self):
        self.documents: list[dict] = []
        self._id_map: dict[str, int] = {}

    def delete(self):
        self.documents.clear()
        self._id_map.clear()

    def add(self, ids, embeddings=None, metadatas=None, documents=None):
        for i, doc in enumerate(documents or []):
            self._id_map[ids[i]] = len(self.documents)
            self.documents.append({
                "id": ids[i],
                "text": doc,
                "embedding": embeddings[i] if embeddings else None,
                "metadata": metadatas[i] if metadatas else {},
            })

    def query(self, query_texts=None, query_embeddings=None, n_results=5):
        if query_embeddings:
            return self._vector_search(query_embeddings[0], n_results)
        return self._keyword_search(query_texts[0] if query_texts else "", n_results)

    def _keyword_search(self, query: str, n: int):
        q = query.lower()
        scored = []
        for doc in self.documents:
            score = sum(1 for word in q.split() if word in doc["text"].lower())
            scored.append((score, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:n]
        return {
            "ids": [[r[1]["id"] for r in top]],
            "documents": [[r[1]["text"] for r in top]],
            "metadatas": [[r[1]["metadata"] for r in top]],
            "distances": [[0.0 for _ in top]],
        }

    def _vector_search(self, query_vec: list[float], n: int):
        import math
        scored = []
        for doc in self.documents:
            if doc["embedding"] is None:
                continue
            sim = self._cosine_similarity(query_vec, doc["embedding"])
            scored.append((sim, doc))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:n]
        return {
            "ids": [[r[1]["id"] for r in top]],
            "documents": [[r[1]["text"] for r in top]],
            "metadatas": [[r[1]["metadata"] for r in top]],
            "distances": [[1.0 - r[0] for r in top]],
        }

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-10)

    def get(self, where: dict, limit: int):
        key, value = next(iter(where.items()))
        matched = [d for d in self.documents if d["metadata"].get(key) == value]
        matched = matched[:limit]
        return {
            "ids": [d["id"] for d in matched],
            "documents": [d["text"] for d in matched],
            "metadatas": [d["metadata"] for d in matched],
        }

    def count(self):
        return len(self.documents)
