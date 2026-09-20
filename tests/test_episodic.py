"""Tier 2 tests: cosine ranking, exponential decay, and budgeted retrieval."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from memcore.episodic import (
    EpisodicMemory,
    HashingEmbedder,
    cosine_similarity,
    decay_factor,
    half_life_to_lambda,
)
from memcore.models import Turn, count_tokens

NOW = 1_700_000_000.0
HOUR = 3600.0
TEXT = "the user wants vector retrieval with exponential temporal decay"


def _memory(half_life_hours: float = 24.0) -> EpisodicMemory:
    return EpisodicMemory(
        embedder=HashingEmbedder(dimensions=128),
        lambda_decay=half_life_to_lambda(half_life_hours),
    )


def test_identical_content_ranks_newer_above_older() -> None:
    """The headline property: equal similarity, unequal age -> recency wins."""
    memory = _memory()
    old = memory.add(TEXT, timestamp=NOW - 72 * HOUR)
    new = memory.add(TEXT, timestamp=NOW - 1 * HOUR)

    ranked = memory.search(TEXT, top_k=2, now=NOW)
    assert [item.chunk.id for item in ranked] == [new.id, old.id]
    # Similarity is identical, so the entire gap comes from decay.
    assert math.isclose(ranked[0].similarity, ranked[1].similarity, abs_tol=1e-9)
    assert ranked[0].decay > ranked[1].decay
    assert ranked[0].score > ranked[1].score


def test_decay_follows_the_documented_formula() -> None:
    lambda_decay = half_life_to_lambda(24.0)
    memory = EpisodicMemory(embedder=HashingEmbedder(128), lambda_decay=lambda_decay)
    memory.add(TEXT, timestamp=NOW - 12 * HOUR)

    (scored,) = memory.search(TEXT, top_k=1, now=NOW)
    expected_decay = math.exp(-lambda_decay * 12.0)
    assert math.isclose(scored.decay, expected_decay, rel_tol=1e-9)
    assert math.isclose(scored.score, scored.similarity * expected_decay, rel_tol=1e-9)


def test_half_life_halves_the_weight() -> None:
    lambda_decay = half_life_to_lambda(6.0)
    assert math.isclose(decay_factor(6.0, lambda_decay), 0.5, rel_tol=1e-9)
    assert math.isclose(decay_factor(12.0, lambda_decay), 0.25, rel_tol=1e-9)
    assert decay_factor(0.0, lambda_decay) == 1.0
    assert decay_factor(-5.0, lambda_decay) == 1.0  # clock skew is clamped
    with pytest.raises(ValueError):
        half_life_to_lambda(0.0)


def test_strong_recent_match_beats_weak_recent_match() -> None:
    memory = _memory()
    relevant = memory.add("duckdb columnar analytics engine", timestamp=NOW - HOUR)
    memory.add("the weather in berlin is cold today", timestamp=NOW - HOUR)

    ranked = memory.search("duckdb columnar analytics", top_k=2, now=NOW)
    assert ranked[0].chunk.id == relevant.id


def test_zero_decay_reduces_ranking_to_pure_similarity() -> None:
    memory = EpisodicMemory(embedder=HashingEmbedder(128), lambda_decay=0.0)
    ancient = memory.add(TEXT, timestamp=NOW - 10_000 * HOUR)
    (scored,) = memory.search(TEXT, top_k=1, now=NOW)
    assert scored.chunk.id == ancient.id
    assert scored.decay == 1.0
    assert math.isclose(scored.score, scored.similarity, rel_tol=1e-12)


def test_cosine_similarity_edge_cases() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0, rel_tol=1e-9)
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0  # mismatched widths
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dimensions=64)
    first = embedder.embed("stable input")
    assert first == embedder.embed("stable input")
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(value**2 for value in first)), 1.0, rel_tol=1e-9)
    assert embedder.embed("") == [0.0] * 64


def test_ingest_turns_groups_drained_turns_into_chunks() -> None:
    memory = _memory()
    turns = [
        Turn(role="user", content=f"message {index}", timestamp=NOW - index)
        for index in range(5)
    ]
    chunks = memory.ingest_turns(turns, max_turns_per_chunk=2)

    assert len(chunks) == 3  # 2 + 2 + 1
    assert "user: message 0" in chunks[0].content
    assert all(chunk.embedding for chunk in chunks)
    assert memory.ingest_turns([]) == []


def test_retrieve_never_exceeds_its_token_limit() -> None:
    memory = _memory()
    for index in range(10):
        memory.add(f"{TEXT} variant {index}", timestamp=NOW - index * HOUR)

    for limit in (0, 5, 20, 60, 400):
        selected = memory.retrieve(TEXT, token_limit=limit, top_k=10, now=NOW)
        rendered = "\n".join(item.chunk.render() for item in selected)
        assert count_tokens(rendered) <= limit
        assert count_tokens(memory.render(TEXT, token_limit=limit, top_k=10, now=NOW)) <= limit


def test_render_of_empty_memory_is_empty() -> None:
    memory = _memory()
    assert memory.render(TEXT, token_limit=100) == ""
    assert memory.search(TEXT, top_k=3) == []
    assert len(memory) == 0


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "episodes.json"
    memory = _memory()
    memory.add(TEXT, timestamp=NOW)
    memory.save(path)

    restored = _memory()
    assert restored.load(path) == 1
    assert restored.chunks[0].content == TEXT
    assert restored.load(tmp_path / "missing.json") == 0


# ------------------------------------------------------- production hardening


def test_episodes_survive_a_restart(tmp_path: Path) -> None:
    """Regression: Tier 2 used to evaporate on process exit."""
    path = tmp_path / "episodes.sqlite"
    with EpisodicMemory(path=path, embedder=HashingEmbedder(128)) as first:
        first.add("we chose DuckDB over Postgres for the warehouse", timestamp=NOW)
        first.add("the deadline is March 3rd", timestamp=NOW)

    with EpisodicMemory(path=path, embedder=HashingEmbedder(128)) as second:
        assert len(second) == 2
        hit = second.search("which warehouse database", top_k=1, now=NOW)[0]
        assert "DuckDB" in hit.chunk.content
        # Vectors were persisted, not recomputed from scratch.
        assert len(hit.chunk.embedding) == 128
        assert hit.similarity > 0.0


def test_durable_and_in_memory_stores_rank_identically(tmp_path: Path) -> None:
    contents = [(f"episode {index} about retrieval", NOW - index * HOUR) for index in range(8)]
    volatile = _memory()
    durable = EpisodicMemory(
        path=tmp_path / "e.sqlite",
        embedder=HashingEmbedder(dimensions=128),
        lambda_decay=half_life_to_lambda(24.0),
    )
    for content, when in contents:
        volatile.add(content, timestamp=when)
        durable.add(content, timestamp=when)

    left = [item.chunk.content for item in volatile.search("retrieval", top_k=5, now=NOW)]
    right = [item.chunk.content for item in durable.search("retrieval", top_k=5, now=NOW)]
    assert left == right
    durable.close()


def test_compaction_bounds_the_store(tmp_path: Path) -> None:
    memory = EpisodicMemory(
        path=tmp_path / "bounded.sqlite", embedder=HashingEmbedder(64), max_chunks=50
    )
    for index in range(200):
        memory.add(f"episode {index}", timestamp=NOW - (200 - index) * 60)

    assert len(memory) == 50
    assert memory.chunks[-1].content == "episode 199"  # newest retained
    memory.close()

    # Compaction is durable, not just in-process.
    with EpisodicMemory(path=tmp_path / "bounded.sqlite", embedder=HashingEmbedder(64)) as reopened:
        assert len(reopened) == 50


def test_compaction_requires_a_positive_bound() -> None:
    with pytest.raises(ValueError):
        EpisodicMemory(max_chunks=0)


def test_search_matches_score_all_ordering() -> None:
    """The top-k fast path must agree with the exhaustive ranking."""
    memory = _memory()
    for index in range(40):
        memory.add(f"{TEXT} variant {index % 7}", timestamp=NOW - index * HOUR)

    exhaustive = [item.chunk.id for item in memory.score_all(TEXT, now=NOW)][:5]
    fast_path = [item.chunk.id for item in memory.search(TEXT, top_k=5, now=NOW)]
    assert exhaustive == fast_path


def test_scoring_does_not_mutate_stored_chunks() -> None:
    """Concurrent readers must not see each other's decay values."""
    memory = _memory()
    chunk = memory.add(TEXT, timestamp=NOW - 48 * HOUR)
    assert chunk.decay_score == 1.0

    result = memory.search(TEXT, top_k=1, now=NOW)[0]
    assert result.chunk.decay_score < 1.0  # the copy carries the ranking
    assert memory.chunks[0].decay_score == 1.0  # the stored chunk is untouched
    assert memory.chunks[0].id == chunk.id


def test_concurrent_writes_and_searches_are_safe() -> None:
    import threading

    memory = _memory()
    errors: list[str] = []

    def worker(index: int) -> None:
        try:
            for round_index in range(25):
                memory.add(f"thread {index} episode {round_index}", timestamp=NOW - round_index)
                memory.search("episode", top_k=3, now=NOW)
        except Exception as exc:  # pragma: no cover - only on regression
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(memory) == 150


def test_ingest_summarises_drained_turns_by_default() -> None:
    memory = _memory()
    turns = [
        Turn(role="user", content="My budget is $5000", timestamp=NOW),
        Turn(role="assistant", content="Sounds good, thanks!", timestamp=NOW),
        Turn(role="user", content="We deploy FastAPI in Frankfurt", timestamp=NOW),
    ]
    (chunk,) = memory.ingest_turns(turns, max_turns_per_chunk=4)
    assert "$5000" in chunk.content
    assert "Sounds good" not in chunk.content  # filler compressed away


def test_summarisation_can_be_disabled() -> None:
    memory = EpisodicMemory(embedder=HashingEmbedder(64), summarize_on_ingest=False)
    turns = [Turn(role="assistant", content="Sounds good, thanks!", timestamp=NOW)]
    (chunk,) = memory.ingest_turns(turns)
    assert "Sounds good" in chunk.content


def test_stats_describe_the_store() -> None:
    memory = _memory()
    memory.add(TEXT, timestamp=NOW)
    stats = memory.stats()
    assert stats["episodes"] == 1
    assert stats["dimensions"] == 128
    assert stats["durable"] is False


def test_json_round_trip_also_repopulates_a_durable_store(tmp_path: Path) -> None:
    source = _memory()
    source.add(TEXT, timestamp=NOW)
    source.save(tmp_path / "episodes.json")

    path = tmp_path / "restored.sqlite"
    with EpisodicMemory(path=path, embedder=HashingEmbedder(128)) as restored:
        assert restored.load(tmp_path / "episodes.json") == 1
    with EpisodicMemory(path=path, embedder=HashingEmbedder(128)) as reopened:
        assert len(reopened) == 1  # the JSON import was persisted
