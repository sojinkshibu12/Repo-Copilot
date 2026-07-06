import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.tracer import get_tracer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    embedding: list[float]
    model: str
    dimension: int
    tokens_used: int = 0


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


_SUPPORTED_EMBEDDERS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "openai-large": "text-embedding-3-large",
    "openai-ada": "text-embedding-ada-002",
    "google": "text-embedding-004",
    "local": "all-MiniLM-L6-v2",
    "mock": "mock",
}


class EmbeddingEngine:
    """Unified embedding engine — switch provider via EMBEDDING_PROVIDER env var."""

    def __init__(self, provider: str = "", model: str = "", api_key: str | None = None):
        self.provider = provider or os.environ.get("EMBEDDING_PROVIDER", "local")
        self._model_name = model or os.environ.get("EMBEDDING_MODEL", "")
        self.api_key = api_key
        self._embedder: BaseEmbedder = self._build_embedder()

        logger.info("EmbeddingEngine: provider=%s model=%s dim=%d",
                     self.provider, self.model_name, self._embedder.dimension)

    def _build_embedder(self) -> BaseEmbedder:
        factory = {
            "openai": _OpenAIEmbedder,
            "google": _GoogleEmbedder,
            "local": _LocalEmbedder,
            "mock": _MockEmbedder,
        }
        cls = factory.get(self.provider)
        if cls is None:
            logger.warning("Unknown embedding provider '%s', falling back to mock", self.provider)
            return _MockEmbedder()
        try:
            return cls(model=self._model_name, api_key=self.api_key)
        except TypeError:
            return cls(model=self._model_name)

    def embed(self, text: str) -> EmbeddingResult:
        tracer = get_tracer()
        with tracer.span("embed.single", kind="embed", attributes={
            "provider": self.provider,
            "model": self.model_name,
            "text_length": len(text),
        }) as span:
            result = self._embedder.embed(text)
            span.set_attribute("dimension", result.dimension)
            span.set_attribute("tokens_used", result.tokens_used)
            return result

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        tracer = get_tracer()
        with tracer.span("embed.batch", kind="embed", attributes={
            "provider": self.provider,
            "model": self.model_name,
            "batch_size": len(texts),
            "total_length": sum(len(t) for t in texts),
        }) as span:
            results = self._embedder.embed_batch(texts)
            span.set_attribute("num_results", len(results))
            span.set_attribute("dimension", results[0].dimension if results else 0)
            return results

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        return self._model_name or (self._embedder.model_name if hasattr(self, "_embedder") else "unknown")


# ---------------------------------------------------------------------------
# OpenAI Embedder
# ---------------------------------------------------------------------------

class _OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "", api_key: str | None = None):
        self.model_name = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    @property
    def _client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @_client.setter
    def _client(self, value):
        self._client = value

    @property
    def dimension(self) -> int:
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dims.get(self.model_name, 1536)

    def embed(self, text: str) -> EmbeddingResult:
        response = self._client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        data = response.data[0]
        return EmbeddingResult(
            embedding=data.embedding,
            model=self.model_name,
            dimension=len(data.embedding),
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        results = []
        for data in response.data:
            results.append(EmbeddingResult(
                embedding=data.embedding,
                model=self.model_name,
                dimension=len(data.embedding),
                tokens_used=(response.usage.total_tokens // len(texts)) if response.usage else 0,
            ))
        return results


# ---------------------------------------------------------------------------
# Google Gemini Embedder (free tier: 60 req/min)
# model: text-embedding-004 (768-dim)
# ---------------------------------------------------------------------------

class _GoogleEmbedder(BaseEmbedder):
    def __init__(self, model: str = "", api_key: str | None = None):
        self.model_name = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai

    @property
    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> EmbeddingResult:
        self._ensure_client()
        result = self._client.embed_content(
            model=f"models/{self.model_name}",
            content=text,
        )
        vec = result["embedding"]
        return EmbeddingResult(
            embedding=vec,
            model=self.model_name,
            dimension=len(vec),
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        self._ensure_client()
        result = self._client.embed_content(
            model=f"models/{self.model_name}",
            content=texts,
        )
        embeddings = result["embedding"]
        return [
            EmbeddingResult(embedding=vec, model=self.model_name, dimension=len(vec))
            for vec in embeddings
        ]


# ---------------------------------------------------------------------------
# Local Embedder (sentence-transformers)
# ---------------------------------------------------------------------------

class _LocalEmbedder(BaseEmbedder):
    def __init__(self, model: str = "", api_key: str | None = None):
        self.model_name = model or os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded local embedding model: %s (dim=%d)",
                            self.model_name, self._model.get_sentence_embedding_dimension())
            except ImportError:
                logger.warning("sentence-transformers not installed, falling back to mock")
                return _MockEmbedder()._model
        return self._model

    @property
    def dimension(self) -> int:
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            return 384

    def embed(self, text: str) -> EmbeddingResult:
        vec = self.model.encode(text).tolist()
        return EmbeddingResult(
            embedding=vec,
            model=self.model_name,
            dimension=len(vec),
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        vectors = self.model.encode(texts).tolist()
        return [
            EmbeddingResult(embedding=vec, model=self.model_name, dimension=len(vec))
            for vec in vectors
        ]


# ---------------------------------------------------------------------------
# Mock Embedder (deterministic, no network)
# ---------------------------------------------------------------------------

class _MockEmbedder(BaseEmbedder):
    def __init__(self, model: str = "", api_key: str | None = None):
        self.model_name = model or "mock"

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, text: str) -> EmbeddingResult:
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, 8, 2)]
        return EmbeddingResult(embedding=vec, model=self.model_name, dimension=len(vec))

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed(t) for t in texts]


def list_supported_embedders() -> list[dict]:
    return [
        {"alias": alias, "model": model}
        for alias, model in sorted(_SUPPORTED_EMBEDDERS.items())
    ]
