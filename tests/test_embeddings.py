import pytest
from core.embeddings import EmbeddingEngine, list_supported_embedders


class TestMockEmbedder:
    def test_mock_embed_returns_vector(self):
        engine = EmbeddingEngine(provider="mock")
        result = engine.embed("hello world")
        assert len(result.embedding) == 4
        assert all(isinstance(v, float) for v in result.embedding)
        assert result.model == "mock"

    def test_mock_embed_is_deterministic(self):
        engine = EmbeddingEngine(provider="mock")
        r1 = engine.embed("same text")
        r2 = engine.embed("same text")
        assert r1.embedding == r2.embedding

    def test_mock_different_inputs_different_vectors(self):
        engine = EmbeddingEngine(provider="mock")
        r1 = engine.embed("cat")
        r2 = engine.embed("dog")
        assert r1.embedding != r2.embedding

    def test_mock_embed_batch(self):
        engine = EmbeddingEngine(provider="mock")
        results = engine.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(r.embedding) == 4 for r in results)

    def test_mock_dimension(self):
        engine = EmbeddingEngine(provider="mock")
        assert engine.dimension == 4

    def test_mock_model_name(self):
        engine = EmbeddingEngine(provider="mock")
        assert "mock" in engine.model_name


class TestEmbeddingEngineFactory:
    def test_default_provider_falls_back_to_mock(self):
        engine = EmbeddingEngine(provider="nonexistent")
        assert "mock" in engine.model_name

    def test_google_embedder_initializes(self):
        engine = EmbeddingEngine(provider="google")
        assert engine.dimension == 768
        assert "text-embedding-004" in engine.model_name

    def test_list_supported(self):
        embedders = list_supported_embedders()
        assert len(embedders) >= 4
        names = [e["alias"] for e in embedders]
        assert "openai" in names
        assert "google" in names
        assert "local" in names
        assert "mock" in names


class TestEmbeddingResult:
    def test_result_dataclass(self):
        from core.embeddings import EmbeddingResult
        r = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimension=3,
            tokens_used=42,
        )
        assert r.embedding == [0.1, 0.2, 0.3]
        assert r.dimension == 3
        assert r.tokens_used == 42
