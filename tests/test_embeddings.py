"""Embedding backend tests, including the remote paths via a stub transport."""

from __future__ import annotations

import json
import math
import urllib.error
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from memcore.embeddings import (
    CachingEmbedder,
    Embedder,
    GeminiEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
    cosine_matrix,
    l2_normalize,
)

# --------------------------------------------------------------- hashing


def test_vectors_are_deterministic_normalised_and_non_negative() -> None:
    embedder = HashingEmbedder(dimensions=64)
    first = embedder.embed("stable input")
    assert first == embedder.embed("stable input")
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(value**2 for value in first)), 1.0, rel_tol=1e-9)
    # log1p weighting must not produce negative components, which would
    # make cosine scores negative and break min_score filtering.
    assert all(value >= 0.0 for value in first)
    assert embedder.embed("") == [0.0] * 64


def test_batch_matches_single() -> None:
    embedder = HashingEmbedder(dimensions=32)
    texts = ["alpha beta", "gamma delta"]
    assert embedder.embed_batch(texts) == [embedder.embed(text) for text in texts]
    assert embedder.embed_batch([]) == []


def test_character_ngrams_bridge_morphology() -> None:
    """"database" and "databases" must not look unrelated."""
    words_only = HashingEmbedder(256, char_ngrams=None)
    with_chars = HashingEmbedder(256, char_ngrams=(3, 5))

    def similarity(embedder: Embedder, a: str, b: str) -> float:
        return float(np.dot(embedder.embed(a), embedder.embed(b)))

    assert similarity(words_only, "database", "databases") == pytest.approx(0.0, abs=1e-9)
    assert similarity(with_chars, "database", "databases") > 0.5


def test_retrieval_quality_regression_guard() -> None:
    """Pins the default configuration's recall on a small labelled set.

    The local embedder is lexical, so perfect recall is not the bar; this
    guards against a change that quietly makes retrieval *worse*.
    """
    docs = [
        "we use DuckDB for the analytics warehouse",
        "the office coffee machine is broken again",
        "my budget for this project is five thousand dollars",
        "we deploy the FastAPI service on a single virtual machine",
        "the deadline for the migration is March third",
        "I prefer writing raw SQL over using an ORM layer",
        "Redis caches the hottest aggregate queries",
        "our team is based in Berlin and works European hours",
    ]
    queries = [
        ("DuckDB analytics warehouse", 0),
        ("which database do we run", 0),
        ("what is the budget", 2),
        ("where is the API deployed", 3),
        ("when is the migration due", 4),
        ("do we use an ORM", 5),
        ("caching layer", 6),
        ("which city is the team in", 7),
    ]
    embedder = HashingEmbedder(512)
    matrix = np.array(embedder.embed_batch(docs))
    hits = sum(
        int(np.argmax(cosine_matrix(embedder.embed(query), matrix)) == gold)
        for query, gold in queries
    )
    assert hits / len(queries) >= 0.5


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dimensions=0)
    with pytest.raises(ValueError):
        HashingEmbedder(char_ngrams=(0, 3))
    with pytest.raises(ValueError):
        HashingEmbedder(char_ngrams=(5, 3))


# --------------------------------------------------------------- caching


def test_cache_avoids_recomputation() -> None:
    class Counting:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def dimensions(self) -> int:
            return 4

        def embed(self, text: str) -> list[float]:
            self.calls += 1
            return [1.0, 0.0, 0.0, 0.0]

        def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            return [self.embed(text) for text in texts]

    inner = Counting()
    cache = CachingEmbedder(inner, max_entries=8)
    for _ in range(5):
        cache.embed("same text")

    assert inner.calls == 1
    assert cache.stats() == {"hits": 4, "misses": 1, "entries": 1}
    assert cache.dimensions == 4


def test_cache_evicts_least_recently_used() -> None:
    cache = CachingEmbedder(HashingEmbedder(8), max_entries=2)
    cache.embed("a")
    cache.embed("b")
    cache.embed("a")  # refreshes "a"
    cache.embed("c")  # evicts "b"
    assert cache.stats()["entries"] == 2
    before = cache.stats()["misses"]
    cache.embed("b")
    assert cache.stats()["misses"] == before + 1  # "b" was indeed evicted


def test_cache_batch_deduplicates() -> None:
    cache = CachingEmbedder(HashingEmbedder(8))
    vectors = cache.embed_batch(["x", "y", "x"])
    assert len(vectors) == 3
    assert vectors[0] == vectors[2]
    assert cache.stats()["entries"] == 2


def test_cached_vectors_cannot_be_mutated_by_callers() -> None:
    cache = CachingEmbedder(HashingEmbedder(8))
    first = cache.embed("text")
    first[0] = 999.0
    assert cache.embed("text")[0] != 999.0


# ---------------------------------------------------------------- gemini


class _StubTransport:
    """Records requests and replays canned responses (no network)."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
        self.requests.append(
            {"url": url, "body": json.loads(payload.decode()), "timeout": timeout}
        )
        reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply).encode()


def _embedding(values: list[float]) -> dict[str, Any]:
    return {"embedding": {"values": values}}


def test_gemini_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbedder()


def test_gemini_sends_the_documented_request_and_normalises_the_reply() -> None:
    transport = _StubTransport([_embedding([3.0, 4.0])])
    embedder = GeminiEmbedder(api_key="k", dimensions=2, transport=transport)

    vector = embedder.embed("hello")

    request = transport.requests[0]
    assert "gemini-embedding-001:embedContent" in request["url"]
    assert request["url"].endswith("key=k")
    assert request["body"]["content"]["parts"][0]["text"] == "hello"
    assert request["body"]["outputDimensionality"] == 2
    # 3-4-5 triangle: the reply must come back unit length.
    assert vector == pytest.approx([0.6, 0.8])


def test_gemini_batches_into_one_request() -> None:
    transport = _StubTransport([{"embeddings": [{"values": [1.0, 0.0]}, {"values": [0.0, 2.0]}]}])
    embedder = GeminiEmbedder(api_key="k", dimensions=2, transport=transport, batch_size=8)

    vectors = embedder.embed_batch(["one", "two"])

    assert len(transport.requests) == 1  # batched, not one call per text
    assert "batchEmbedContents" in transport.requests[0]["url"]
    assert len(transport.requests[0]["body"]["requests"]) == 2
    assert vectors == [pytest.approx([1.0, 0.0]), pytest.approx([0.0, 1.0])]


def test_gemini_respects_batch_size() -> None:
    page = {"embeddings": [{"values": [1.0, 0.0]}]}
    transport = _StubTransport([page, page, page])
    embedder = GeminiEmbedder(api_key="k", dimensions=2, transport=transport, batch_size=1)
    assert len(embedder.embed_batch(["a", "b", "c"])) == 3
    assert len(transport.requests) == 3


def test_gemini_retries_transient_failures_then_succeeds() -> None:
    slept: list[float] = []
    transport = _StubTransport(
        [
            urllib.error.HTTPError("u", 503, "unavailable", {}, None),  # type: ignore[arg-type]
            urllib.error.URLError("connection reset"),
            _embedding([1.0, 0.0]),
        ]
    )
    embedder = GeminiEmbedder(
        api_key="k",
        dimensions=2,
        transport=transport,
        max_attempts=3,
        backoff_base=0.5,
        sleep_fn=slept.append,
    )

    assert embedder.embed("hi") == pytest.approx([1.0, 0.0])
    assert len(transport.requests) == 3
    assert slept == [0.5, 1.0]  # exponential backoff


def test_gemini_does_not_retry_client_errors() -> None:
    transport = _StubTransport(
        [urllib.error.HTTPError("u", 400, "bad request", {}, None)]  # type: ignore[arg-type]
    )
    embedder = GeminiEmbedder(
        api_key="k", transport=transport, max_attempts=4, sleep_fn=lambda _: None
    )
    with pytest.raises(RuntimeError, match="failed"):
        embedder.embed("hi")
    assert len(transport.requests) == 1  # 400 is not retryable


def test_gemini_gives_up_after_max_attempts() -> None:
    transport = _StubTransport([urllib.error.URLError("down")] * 3)
    embedder = GeminiEmbedder(
        api_key="k", transport=transport, max_attempts=3, sleep_fn=lambda _: None
    )
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        embedder.embed("hi")


def test_gemini_rejects_malformed_responses() -> None:
    embedder = GeminiEmbedder(
        api_key="k", transport=_StubTransport([{"nope": True}]), sleep_fn=lambda _: None
    )
    with pytest.raises(RuntimeError, match="unexpected gemini response"):
        embedder.embed("hi")


def test_gemini_rejects_short_batch_responses() -> None:
    embedder = GeminiEmbedder(
        api_key="k",
        transport=_StubTransport([{"embeddings": [{"values": [1.0]}]}]),
        sleep_fn=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="unexpected gemini batch response"):
        embedder.embed_batch(["a", "b"])


def test_gemini_configuration_is_validated() -> None:
    with pytest.raises(ValueError):
        GeminiEmbedder(api_key="k", max_attempts=0)
    with pytest.raises(ValueError):
        GeminiEmbedder(api_key="k", batch_size=0)


# ------------------------------------------------- sentence-transformers


class _StubModel:
    """Mimics the SentenceTransformer surface the adapter relies on."""

    def __init__(self, dimensions: int = 3) -> None:
        self._dimensions = dimensions
        self.encoded: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimensions

    def encode(self, texts: list[str], normalize_embeddings: bool = False) -> Any:
        self.encoded.append(list(texts))
        return np.array([[float(len(text)), 1.0, 0.0] for text in texts])


def test_sentence_transformer_adapter_normalises_and_batches() -> None:
    model = _StubModel()
    embedder = SentenceTransformerEmbedder(model=model)

    assert embedder.dimensions == 3
    vectors = embedder.embed_batch(["ab", "cde"])
    assert model.encoded == [["ab", "cde"]]
    for vector in vectors:
        assert math.isclose(math.sqrt(sum(value**2 for value in vector)), 1.0, rel_tol=1e-9)
    assert embedder.embed_batch([]) == []
    assert len(embedder.embed("single")) == 3


# ----------------------------------------------------------- build_embedder


def test_build_embedder_returns_a_cached_local_backend() -> None:
    embedder = build_embedder("local")
    assert isinstance(embedder, CachingEmbedder)
    assert isinstance(embedder.inner, HashingEmbedder)
    assert isinstance(build_embedder("local", cache=False), HashingEmbedder)


def test_build_embedder_degrades_instead_of_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing API key must cost recall quality, not availability."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    embedder = build_embedder("gemini", cache=False)
    assert isinstance(embedder, HashingEmbedder)


def test_build_embedder_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown embedder"):
        build_embedder("no-such-backend")


def test_l2_normalize_leaves_zero_vectors_alone() -> None:
    zero = np.zeros(3, dtype=np.float64)
    assert np.array_equal(l2_normalize(zero), zero)
    assert cosine_matrix([0.0, 0.0], np.array([[1.0, 0.0]])).tolist() == [0.0]
    assert cosine_matrix([1.0], np.zeros((0, 0))).size == 0
