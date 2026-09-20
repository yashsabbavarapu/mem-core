"""Tier 2 — Episodic memory: semantic recall with exponential time decay.

Chunks of past conversation are embedded once and retrieved by similarity
to the incoming query, *penalised by age*:

    score = cosine_similarity(q, c) * exp(-lambda_decay * delta_hours)

The decay term stops a highly similar but stale episode from outranking a
slightly less similar but current one.  See the README for the derivation
of ``lambda_decay`` from a chosen half-life.

Production characteristics:

* **Durable.** Pass ``path=`` and episodes (including their vectors)
  persist in SQLite across restarts; the in-memory default stays available
  for tests.
* **Vectorised.** Scoring is one mat-vec product over a contiguous
  ``(N, d)`` matrix, not a Python loop over chunks.
* **Bounded.** ``max_chunks`` compacts the store, evicting the
  lowest-value episodes rather than growing without limit.
* **Thread-safe.** All mutating and scoring paths hold a re-entrant lock.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from memcore.embeddings import (
    CachingEmbedder,
    Embedder,
    GeminiEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
    cosine_matrix,
)
from memcore.models import EpisodicChunk, Turn, count_tokens
from memcore.summarize import compress_turns, summarize_to_tokens

# Embedder names are re-exported so ``from memcore.episodic import
# HashingEmbedder`` keeps working after the split into ``memcore.embeddings``.
__all__ = [
    "CachingEmbedder",
    "Embedder",
    "EpisodicMemory",
    "GeminiEmbedder",
    "HashingEmbedder",
    "ScoredChunk",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "cosine_similarity",
    "decay_factor",
    "half_life_to_lambda",
]

logger = logging.getLogger("memcore.episodic")

SECONDS_PER_HOUR: Final[float] = 3600.0

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS episodes (
    id         TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    timestamp  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes (timestamp DESC);
"""


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 for degenerate input)."""
    if len(left) != len(right) or not left:
        return 0.0
    a: NDArray[np.float64] = np.asarray(left, dtype=np.float64)
    b: NDArray[np.float64] = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def decay_factor(delta_hours: float, lambda_decay: float) -> float:
    """Exponential recency weight ``exp(-lambda * delta_hours)``, clamped to (0, 1]."""
    if delta_hours <= 0.0 or lambda_decay <= 0.0:
        return 1.0
    return float(math.exp(-lambda_decay * delta_hours))


def half_life_to_lambda(half_life_hours: float) -> float:
    """Convert a half-life in hours into the decay constant ``lambda``.

    ``exp(-lambda * t_half) = 0.5``  =>  ``lambda = ln(2) / t_half``.
    """
    if half_life_hours <= 0.0:
        raise ValueError("half_life_hours must be positive")
    return math.log(2.0) / half_life_hours


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieval result: the chunk plus its decomposed ranking terms."""

    chunk: EpisodicChunk
    similarity: float
    decay: float
    score: float


class EpisodicMemory:
    """Vector store of past episodes ranked by similarity x recency.

    Args:
        embedder: Vectoriser; defaults to a cached
            :class:`~memcore.embeddings.HashingEmbedder`.
        lambda_decay: Decay constant in inverse hours.  The default
            corresponds to a 24-hour half-life.
        min_score: Results scoring at or below this are dropped.
        path: SQLite file for durable storage.  ``None`` keeps everything
            in process memory (lost on restart).
        max_chunks: Optional cap; the store compacts to this size by
            evicting the least valuable episodes.
        summarize_on_ingest: Compress drained turns into a summary rather
            than storing the raw transcript.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        lambda_decay: float = half_life_to_lambda(24.0),
        min_score: float = 0.0,
        path: str | Path | None = None,
        max_chunks: int | None = None,
        summarize_on_ingest: bool = True,
        summary_token_limit: int | None = 120,
    ) -> None:
        if lambda_decay < 0.0:
            raise ValueError("lambda_decay must be non-negative")
        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be positive when provided")

        self.embedder: Embedder = (
            embedder if embedder is not None else CachingEmbedder(HashingEmbedder())
        )
        self.lambda_decay = lambda_decay
        self.min_score = min_score
        self.max_chunks = max_chunks
        self.summarize_on_ingest = summarize_on_ingest
        self.summary_token_limit = summary_token_limit

        self._lock = threading.RLock()
        self._chunks: list[EpisodicChunk] = []
        self._matrix: NDArray[np.float64] | None = None
        self._conn: sqlite3.Connection | None = None
        self.path = str(path) if path is not None else None

        if self.path is not None:
            self._open_store(self.path)

    # ------------------------------------------------------------ persistence

    def _open_store(self, path: str) -> None:
        """Open (and create) the SQLite store, then load existing episodes."""
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.executescript(_SCHEMA)
        connection.commit()
        self._conn = connection
        self._load_from_store()

    def _load_from_store(self) -> None:
        if self._conn is None:
            return
        rows = self._conn.execute(
            "SELECT id, content, embedding, dimensions, timestamp FROM episodes "
            "ORDER BY timestamp"
        ).fetchall()
        self._chunks = [
            EpisodicChunk(
                id=str(row["id"]),
                content=str(row["content"]),
                embedding=[
                    float(value)
                    for value in np.frombuffer(row["embedding"], dtype=np.float32)
                ],
                timestamp=float(row["timestamp"]),
            )
            for row in rows
        ]
        self._matrix = None
        if self._chunks:
            logger.info("loaded %d episodes from %s", len(self._chunks), self.path)

    def _persist(self, chunk: EpisodicChunk) -> None:
        if self._conn is None:
            return
        # float32 halves the on-disk footprint; the precision loss is far
        # below the noise floor of any embedding model's cosine scores.
        blob = np.asarray(chunk.embedding, dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO episodes (id, content, embedding, dimensions, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk.id, chunk.content, blob, len(chunk.embedding), chunk.timestamp),
        )
        self._conn.commit()

    def _forget(self, chunk_ids: Sequence[str]) -> None:
        if self._conn is None or not chunk_ids:
            return
        self._conn.executemany(
            "DELETE FROM episodes WHERE id = ?", [(chunk_id,) for chunk_id in chunk_ids]
        )
        self._conn.commit()

    # ---------------------------------------------------------------- writes

    def add(
        self,
        content: str,
        timestamp: float | None = None,
        chunk_id: str | None = None,
    ) -> EpisodicChunk:
        """Embed and store ``content`` as a single episode."""
        chunk = EpisodicChunk(
            content=content,
            embedding=self.embedder.embed(content),
            timestamp=time.time() if timestamp is None else timestamp,
        )
        if chunk_id is not None:
            chunk.id = chunk_id
        return self.add_chunk(chunk)

    def add_chunk(self, chunk: EpisodicChunk) -> EpisodicChunk:
        """Store a pre-built chunk, embedding it if necessary."""
        if not chunk.embedding:
            chunk.embedding = self.embedder.embed(chunk.content)
        with self._lock:
            self._chunks.append(chunk)
            self._matrix = None
            self._persist(chunk)
            self._compact()
        return chunk

    def ingest_turns(
        self,
        turns: Iterable[Turn],
        max_turns_per_chunk: int = 4,
    ) -> list[EpisodicChunk]:
        """Absorb turns drained from working memory into episodes.

        Consecutive turns are grouped so retrieval returns a coherent
        exchange rather than an orphaned half of a question.  When
        ``summarize_on_ingest`` is set, each group is compressed: filler
        turns are dropped and the remainder capped at
        ``summary_token_limit``, which keeps Tier 2 dense instead of
        letting it accumulate verbatim chat.
        """
        batch = list(turns)
        if not batch:
            return []
        if max_turns_per_chunk <= 0:
            raise ValueError("max_turns_per_chunk must be positive")

        created: list[EpisodicChunk] = []
        for start in range(0, len(batch), max_turns_per_chunk):
            group = batch[start : start + max_turns_per_chunk]
            rendered = [turn.render() for turn in group]
            content = (
                compress_turns(rendered, self.summary_token_limit)
                if self.summarize_on_ingest
                else " | ".join(rendered)
            )
            if content.strip():
                created.append(self.add(content, timestamp=group[-1].timestamp))
        return created

    def _compact(self) -> None:
        """Evict the least valuable episodes once ``max_chunks`` is exceeded.

        Value is age-weighted: the oldest episodes go first, which matches
        the decay curve already applied at retrieval time.  Called with the
        lock held.
        """
        if self.max_chunks is None or len(self._chunks) <= self.max_chunks:
            return
        surplus = len(self._chunks) - self.max_chunks
        ordered = sorted(self._chunks, key=lambda chunk: chunk.timestamp)
        evicted = ordered[:surplus]
        evicted_ids = {chunk.id for chunk in evicted}
        self._chunks = [chunk for chunk in self._chunks if chunk.id not in evicted_ids]
        self._matrix = None
        self._forget([chunk.id for chunk in evicted])
        logger.info("compacted episodic memory: evicted %d episode(s)", surplus)

    # ---------------------------------------------------------------- reads

    @property
    def chunks(self) -> list[EpisodicChunk]:
        """All stored episodes, insertion ordered."""
        with self._lock:
            return list(self._chunks)

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    def _embedding_matrix(self) -> NDArray[np.float64]:
        """Contiguous ``(N, d)`` matrix of stored vectors, cached until write.

        Building this once per write instead of per query is what turns a
        linear Python loop into a single BLAS call.
        """
        if self._matrix is None:
            if not self._chunks:
                self._matrix = np.zeros((0, 0), dtype=np.float64)
            else:
                width = max(len(chunk.embedding) for chunk in self._chunks)
                matrix = np.zeros((len(self._chunks), width), dtype=np.float64)
                for row, chunk in enumerate(self._chunks):
                    vector = np.asarray(chunk.embedding, dtype=np.float64)
                    matrix[row, : vector.shape[0]] = vector
                self._matrix = matrix
        return self._matrix

    def _compute(
        self, query: str, now: float | None
    ) -> tuple[list[EpisodicChunk], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Score every episode with vector math, materialising no objects.

        Returns the chunks alongside parallel ``(similarity, decay, score)``
        arrays.  Keeping this object-free is what makes large stores
        practical: building a Pydantic model per chunk costs far more than
        the mat-vec product itself.
        """
        with self._lock:
            if not self._chunks:
                empty = np.zeros(0, dtype=np.float64)
                return [], empty, empty, empty
            reference = time.time() if now is None else now
            matrix = self._embedding_matrix()
            chunks = list(self._chunks)
            timestamps = np.array([chunk.timestamp for chunk in chunks], dtype=np.float64)

        query_vector = self.embedder.embed(query)
        width = matrix.shape[1]
        padded = np.zeros(width, dtype=np.float64)
        source = np.asarray(query_vector, dtype=np.float64)[:width]
        padded[: source.shape[0]] = source

        similarities = cosine_matrix(padded, matrix)
        ages = np.maximum(0.0, (reference - timestamps) / SECONDS_PER_HOUR)
        decays = (
            np.exp(-self.lambda_decay * ages)
            if self.lambda_decay > 0.0
            else np.ones_like(ages)
        )
        return chunks, similarities, decays, similarities * decays

    @staticmethod
    def _rank_order(
        scores: NDArray[np.float64], timestamps: NDArray[np.float64]
    ) -> NDArray[np.intp]:
        """Indices ordered by score desc, ties broken by recency desc."""
        order: NDArray[np.intp] = np.lexsort((-timestamps, -scores))
        return order

    def _materialize(
        self,
        chunks: Sequence[EpisodicChunk],
        similarities: NDArray[np.float64],
        decays: NDArray[np.float64],
        scores: NDArray[np.float64],
        indices: Sequence[int],
    ) -> list[ScoredChunk]:
        """Build result objects for the selected indices only.

        Stored chunks are never mutated: each result carries its own copy
        with ``decay_score`` populated, so concurrent readers cannot
        observe each other's rankings.
        """
        return [
            ScoredChunk(
                chunk=chunks[index].model_copy(update={"decay_score": float(decays[index])}),
                similarity=float(similarities[index]),
                decay=float(decays[index]),
                score=float(scores[index]),
            )
            for index in indices
        ]

    def score_all(self, query: str, now: float | None = None) -> list[ScoredChunk]:
        """Score every episode against ``query``, best first.

        ``score = cosine_similarity * exp(-lambda_decay * delta_hours)``.

        This materialises one object per stored episode; prefer
        :meth:`search` on large stores, which only builds the top ``k``.
        """
        chunks, similarities, decays, scores = self._compute(query, now)
        if not chunks:
            return []
        timestamps = np.array([chunk.timestamp for chunk in chunks], dtype=np.float64)
        order = self._rank_order(scores, timestamps)
        return self._materialize(chunks, similarities, decays, scores, [int(i) for i in order])

    def search(
        self,
        query: str,
        top_k: int = 5,
        now: float | None = None,
    ) -> list[ScoredChunk]:
        """Return the ``top_k`` best-scoring episodes above ``min_score``.

        Only the winners are turned into objects, so cost is dominated by
        the vectorised scoring rather than by the size of the store.
        """
        if top_k <= 0:
            return []
        chunks, similarities, decays, scores = self._compute(query, now)
        if not chunks:
            return []
        timestamps = np.array([chunk.timestamp for chunk in chunks], dtype=np.float64)
        order = self._rank_order(scores, timestamps)

        selected: list[int] = []
        for index in order:
            if scores[index] <= self.min_score:
                break  # ordered by score, so everything after is below too
            selected.append(int(index))
            if len(selected) == top_k:
                break
        return self._materialize(chunks, similarities, decays, scores, selected)

    def retrieve(
        self,
        query: str,
        token_limit: int,
        top_k: int = 5,
        now: float | None = None,
    ) -> list[ScoredChunk]:
        """Best episodes that jointly fit inside ``token_limit`` tokens.

        Greedy by rank: a chunk that does not fit is skipped (not
        truncated) and the next-best candidate is tried, so the block stays
        coherent.
        """
        if token_limit <= 0:
            return []
        selected: list[ScoredChunk] = []
        used = 0
        for item in self.search(query, top_k=top_k, now=now):
            cost = count_tokens(item.chunk.render()) + (1 if selected else 0)
            if used + cost <= token_limit:
                selected.append(item)
                used += cost
        return selected

    def render(
        self,
        query: str,
        token_limit: int,
        top_k: int = 5,
        now: float | None = None,
    ) -> str:
        """Render the retrieved episodes as a prompt block."""
        if token_limit <= 0:
            return ""
        selected = self.retrieve(query, token_limit, top_k=top_k, now=now)
        if not selected:
            # Nothing fits whole. Compress the best hit by information density
            # rather than cutting its head off: a hard truncation keeps
            # whatever happens to come first, which is often the pleasantry
            # ("Sounds good, thanks!") and not the fact.
            best = self.search(query, top_k=1, now=now)
            if not best:
                return ""
            return summarize_to_tokens(best[0].chunk.render(), token_limit)
        return "\n".join(item.chunk.render() for item in selected)

    def stats(self) -> dict[str, Any]:
        """Operational counters for logging or a metrics endpoint."""
        with self._lock:
            width = self._chunks[0].embedding.__len__() if self._chunks else 0
            return {
                "episodes": len(self._chunks),
                "dimensions": width,
                "durable": self._conn is not None,
                "path": self.path,
                "lambda_decay": self.lambda_decay,
                "max_chunks": self.max_chunks,
                "matrix_cached": self._matrix is not None,
            }

    # ------------------------------------------------------- JSON interchange

    def save(self, path: str | Path) -> None:
        """Write all episodes to a JSON file (portable interchange format)."""
        with self._lock:
            payload = [chunk.model_dump() for chunk in self._chunks]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> int:
        """Load episodes from a JSON file, replacing the current set."""
        file = Path(path)
        if not file.exists():
            return 0
        raw: list[dict[str, Any]] = json.loads(file.read_text(encoding="utf-8"))
        with self._lock:
            self._chunks = [EpisodicChunk.model_validate(item) for item in raw]
            self._matrix = None
            if self._conn is not None:
                self._conn.execute("DELETE FROM episodes")
                for chunk in self._chunks:
                    self._persist(chunk)
            return len(self._chunks)

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Close the durable store, if one is open."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> EpisodicMemory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
