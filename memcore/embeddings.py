"""Embedding backends for episodic memory.

The tiers care only about the :class:`Embedder` protocol, so the backend is
a deployment decision rather than an architectural one.  Four are provided,
in ascending order of retrieval quality and cost:

===========================  ==========  =============================
Backend                      Cost        Semantic?
===========================  ==========  =============================
:class:`HashingEmbedder`     free        no — lexical overlap only
:class:`SentenceTransformerEmbedder`  free (local model)  yes
:class:`GeminiEmbedder`      API         yes
:class:`CachingEmbedder`     wrapper     inherits the wrapped backend
===========================  ==========  =============================

The honest caveat, stated up front because it is easy to miss: the default
``HashingEmbedder`` matches *words*, not *meaning*.  A paraphrased query
("what datastore powers reporting?") will not retrieve "we use DuckDB".
It is deterministic and dependency-free, which makes it the right default
for tests and demos and the wrong default for production recall.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from hashlib import blake2b
from itertools import pairwise
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CachingEmbedder",
    "Embedder",
    "GeminiEmbedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "Transport",
    "build_embedder",
    "l2_normalize",
]

logger = logging.getLogger("memcore.embeddings")

#: A pluggable HTTP transport: ``(url, payload, headers, timeout) -> body``.
#: Injecting this is how the remote backends are tested without a network.
Transport = Callable[[str, bytes, dict[str, str], float], bytes]


def l2_normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale ``vector`` to unit length; a zero vector is returned unchanged."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into a fixed-width unit vector."""

    @property
    def dimensions(self) -> int:
        """Width of the produced vectors."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single passage."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed several passages, ideally in one round trip."""
        ...


class HashingEmbedder:
    """Deterministic, offline embedder built on the hashing trick.

    Features are word unigrams, word bigrams, and (optionally) character
    n-grams, hashed into a fixed number of buckets with sublinear term
    frequency weighting and L2 normalisation.

    Character n-grams buy robustness to morphology — ``database`` and
    ``databases`` overlap, and so do ``datastore`` and ``database`` through
    their shared ``data`` stem — but no amount of hashing produces true
    synonymy.  For that, use a learned backend.
    """

    def __init__(
        self,
        dimensions: int = 256,
        char_ngrams: tuple[int, int] | None = (3, 5),
        sublinear_tf: bool = True,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if char_ngrams is not None:
            low, high = char_ngrams
            if low <= 0 or high < low:
                raise ValueError("char_ngrams must be a (low, high) pair with 0 < low <= high")
        self._dimensions = dimensions
        self._char_ngrams = char_ngrams
        self._sublinear_tf = sublinear_tf

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
        return cleaned.split()

    def _bucket(self, feature: str) -> int:
        digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimensions

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = self._tokenize(text)
        features: list[tuple[str, float]] = [(word, 1.0) for word in words]
        features.extend((f"{left}_{right}", 0.5) for left, right in pairwise(words))

        if self._char_ngrams is not None:
            low, high = self._char_ngrams
            for word in words:
                padded = f"^{word}$"
                for size in range(low, high + 1):
                    if len(padded) < size:
                        break
                    for start in range(len(padded) - size + 1):
                        features.append((f"#{padded[start:start + size]}", 0.3))
        return features

    def embed(self, text: str) -> list[float]:
        counts = np.zeros(self._dimensions, dtype=np.float64)
        for feature, weight in self._features(text):
            counts[self._bucket(feature)] += weight

        if self._sublinear_tf:
            # log1p damps repeated terms so a word said ten times does not
            # dominate ten distinct words. log1p rather than 1+log(tf)
            # because fractional feature weights (bigrams at 0.5, character
            # n-grams at 0.3) would otherwise go negative.
            counts = np.log1p(counts)
        return [float(value) for value in l2_normalize(counts)]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class SentenceTransformerEmbedder:
    """Local learned embeddings via ``sentence-transformers`` (optional extra).

    This is the recommended zero-cost path to *real* semantic recall:
    the model runs locally, nothing leaves the machine, and paraphrases
    retrieve correctly.  Install with ``pip install 'mem-core[embeddings]'``.

    The library is imported lazily so the dependency stays optional; a
    missing install raises :class:`RuntimeError` with the fix in the message.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Any | None = None,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on extra
                raise RuntimeError(
                    "sentence-transformers is not installed; "
                    "run `pip install 'mem-core[embeddings]'` or choose another embedder"
                ) from exc
            model = SentenceTransformer(model_name)
        self._model = model
        self._model_name = model_name
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        raw = self._model.encode(list(texts), normalize_embeddings=True)
        matrix = np.asarray(raw, dtype=np.float64).reshape(len(texts), -1)
        return [[float(value) for value in l2_normalize(row)] for row in matrix]


def _default_transport(
    url: str, payload: bytes, headers: dict[str, str], timeout: float
) -> bytes:
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body: bytes = response.read()
    return body


_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})


class GeminiEmbedder:
    """Remote embeddings via Google's ``gemini-embedding-001``.

    Hardened for unattended use: bounded retries with exponential backoff on
    transient failures, true batching through ``batchEmbedContents``, and an
    injectable :data:`Transport` so the request/response handling is testable
    without a network.

    Note: the request and response shapes are exercised against a faithful
    stub in the test suite.  Verify against the live API before relying on
    this path in production.
    """

    BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001"

    def __init__(
        self,
        api_key: str | None = None,
        dimensions: int = 768,
        timeout: float = 20.0,
        max_attempts: int = 3,
        backoff_base: float = 0.5,
        batch_size: int = 64,
        transport: Transport | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; use HashingEmbedder or "
                "SentenceTransformerEmbedder for offline runs"
            )
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._api_key = key
        self._dimensions = dimensions
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._batch_size = batch_size
        self._transport: Transport = transport if transport is not None else _default_transport
        self._sleep: Callable[[float], None] = sleep_fn if sleep_fn is not None else time.sleep

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST with bounded exponential backoff on transient failures."""
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._transport(url, payload, headers, self._timeout)
                parsed: dict[str, Any] = json.loads(raw.decode("utf-8"))
                return parsed
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < self._max_attempts:
                delay = self._backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "gemini embedding attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt,
                    self._max_attempts,
                    last_error,
                    delay,
                )
                self._sleep(delay)

        raise RuntimeError(
            f"gemini embedding request failed after {self._max_attempts} attempt(s): {last_error}"
        )

    def _vector(self, values: Any) -> list[float]:
        if not isinstance(values, list):
            raise RuntimeError(f"unexpected gemini response shape: {values!r}")
        vector = np.asarray([float(value) for value in values], dtype=np.float64)
        return [float(value) for value in l2_normalize(vector)]

    def embed(self, text: str) -> list[float]:
        body = self._post(
            f"{self.BASE}:embedContent?key={self._api_key}",
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": self._dimensions,
            },
        )
        return self._vector(body.get("embedding", {}).get("values"))

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            window = texts[start : start + self._batch_size]
            body = self._post(
                f"{self.BASE}:batchEmbedContents?key={self._api_key}",
                {
                    "requests": [
                        {
                            "model": "models/gemini-embedding-001",
                            "content": {"parts": [{"text": text}]},
                            "outputDimensionality": self._dimensions,
                        }
                        for text in window
                    ]
                },
            )
            embeddings = body.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(window):
                raise RuntimeError(f"unexpected gemini batch response shape: {body!r}")
            vectors.extend(self._vector(item.get("values")) for item in embeddings)
        return vectors


class CachingEmbedder:
    """LRU cache in front of any embedder.

    Re-embedding is the dominant cost of a remote backend and conversation
    text repeats constantly (the same turn is embedded once per chunk it
    lands in), so this is close to free throughput.
    """

    def __init__(self, inner: Embedder, max_entries: int = 4096) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._inner = inner
        self._max_entries = max_entries
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def inner(self) -> Embedder:
        """The wrapped backend."""
        return self._inner

    def _remember(self, text: str, vector: list[float]) -> None:
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(text)
            return list(cached)
        self.misses += 1
        vector = self._inner.embed(text)
        self._remember(text, vector)
        return list(vector)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            for text, vector in zip(missing, self._inner.embed_batch(missing), strict=True):
                self._remember(text, vector)
        results: list[list[float]] = []
        for text in texts:
            cached = self._cache.get(text)
            if cached is None:  # pragma: no cover - defensive
                cached = self._inner.embed(text)
                self._remember(text, cached)
                self.misses += 1
            else:
                self.hits += 1
                self._cache.move_to_end(text)
            results.append(list(cached))
        return results

    def stats(self) -> dict[str, int]:
        """Cache counters, useful for a metrics endpoint."""
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._cache)}


def build_embedder(name: str, cache: bool = True, **kwargs: Any) -> Embedder:
    """Construct a backend by name, optionally wrapped in an LRU cache.

    Falls back to the offline :class:`HashingEmbedder` (with a warning) when
    a learned backend is requested but unavailable, so a missing API key
    degrades the *quality* of recall rather than taking the agent down.
    """
    embedder: Embedder
    if name in {"local", "hashing"}:
        embedder = HashingEmbedder(**kwargs)
    elif name in {"sentence-transformers", "st", "minilm"}:
        try:
            embedder = SentenceTransformerEmbedder(**kwargs)
        except RuntimeError as exc:
            logger.warning("%s; falling back to HashingEmbedder", exc)
            embedder = HashingEmbedder()
    elif name == "gemini":
        try:
            embedder = GeminiEmbedder(**kwargs)
        except RuntimeError as exc:
            logger.warning("%s; falling back to HashingEmbedder", exc)
            embedder = HashingEmbedder()
    else:
        raise ValueError(
            f"unknown embedder {name!r}; expected one of: local, sentence-transformers, gemini"
        )
    return CachingEmbedder(embedder) if cache else embedder


def cosine_matrix(
    query: Sequence[float] | NDArray[np.float64], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Cosine similarity of ``query`` against every row of ``matrix``.

    Rows are assumed L2-normalised (every backend here normalises), so this
    is a single mat-vec product instead of a Python loop.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    vector = np.asarray(query, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    similarities: NDArray[np.float64] = matrix @ (vector / norm)
    return np.clip(similarities, -1.0, 1.0)


def iter_batches(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Yield ``items`` in slices of at most ``size``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def approximate_memory_bytes(count: int, dimensions: int) -> int:
    """Bytes needed to hold ``count`` float64 vectors of ``dimensions`` width."""
    return count * dimensions * 8 + math.ceil(count * 64)
