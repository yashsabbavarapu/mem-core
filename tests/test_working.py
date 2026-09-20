"""Tier 1 tests: FIFO eviction, token accounting, and drain semantics."""

from __future__ import annotations

import pytest

from memcore.models import Role, Turn, count_tokens
from memcore.working import WorkingBuffer


def _turn(index: int, content: str = "hello there", role: Role = "user") -> Turn:
    return Turn(role=role, content=f"{content} {index}", timestamp=1_000.0 + index)


def test_token_count_is_derived_from_content() -> None:
    turn = Turn(role="user", content="the quick brown fox jumps over the lazy dog")
    assert turn.token_count == count_tokens(turn.content)
    assert turn.token_count > 0


def test_empty_content_costs_zero_tokens() -> None:
    assert Turn(role="user", content="").token_count == 0


def test_explicit_token_count_is_respected() -> None:
    assert Turn(role="user", content="anything", token_count=42).token_count == 42


def test_buffer_evicts_oldest_turns_first() -> None:
    buffer = WorkingBuffer(max_turns=3)
    evicted_all: list[Turn] = []
    for index in range(5):
        evicted_all.extend(buffer.add(_turn(index)))

    assert len(buffer) == 3
    assert [turn.content for turn in buffer] == ["hello there 2", "hello there 3", "hello there 4"]
    # Nothing is lost: evicted turns are handed back for Tier 2 ingestion.
    assert [turn.content for turn in evicted_all] == ["hello there 0", "hello there 1"]


def test_add_turn_returns_new_turn_and_evictions() -> None:
    buffer = WorkingBuffer(max_turns=1)
    first, evicted = buffer.add_turn("user", "first")
    assert evicted == []
    second, evicted = buffer.add_turn("assistant", "second")
    assert second.role == "assistant"
    assert [turn.content for turn in evicted] == ["first"]
    assert buffer.turns == [second]
    assert first not in buffer.turns


def test_token_cap_evicts_until_buffer_fits() -> None:
    long_text = "word " * 40  # ~50 tokens each
    buffer = WorkingBuffer(max_turns=100, max_tokens=120)
    for index in range(5):
        buffer.add(_turn(index, content=long_text))

    assert buffer.total_tokens <= 120
    assert len(buffer) < 5
    # The survivors are the newest ones.
    assert buffer.turns[-1].content.endswith("4")


def test_single_oversized_turn_is_never_silently_dropped() -> None:
    buffer = WorkingBuffer(max_turns=10, max_tokens=5)
    buffer.add(Turn(role="user", content="word " * 200))
    assert len(buffer) == 1
    assert buffer.total_tokens > 5  # kept; truncation is the compiler's job


def test_total_tokens_matches_sum_of_turns() -> None:
    buffer = WorkingBuffer(max_turns=10)
    buffer.extend(_turn(index) for index in range(4))
    assert buffer.total_tokens == sum(turn.token_count for turn in buffer)


def test_recent_returns_newest_k_in_chronological_order() -> None:
    buffer = WorkingBuffer(max_turns=10)
    buffer.extend(_turn(index) for index in range(6))
    recent = buffer.recent(2)
    assert [turn.content for turn in recent] == ["hello there 4", "hello there 5"]
    assert buffer.recent(0) == []
    assert len(buffer.recent(99)) == 6


def test_drain_keeps_newest_and_returns_the_rest() -> None:
    buffer = WorkingBuffer(max_turns=10)
    buffer.extend(_turn(index) for index in range(5))

    drained = buffer.drain(keep=2)
    assert [turn.content for turn in drained] == [
        "hello there 0",
        "hello there 1",
        "hello there 2",
    ]
    assert len(buffer) == 2
    assert buffer.clear() and len(buffer) == 0


def test_render_respects_token_limit_and_prefers_recent_turns() -> None:
    buffer = WorkingBuffer(max_turns=10)
    buffer.extend(_turn(index, content="a fairly wordy statement about memory") for index in range(6))

    rendered = buffer.render(token_limit=30)
    assert count_tokens(rendered) <= 30
    assert rendered.endswith("memory 5")  # newest turn is always retained
    assert "memory 0" not in rendered  # oldest dropped first


def test_render_truncates_when_even_one_turn_overflows() -> None:
    buffer = WorkingBuffer(max_turns=2)
    buffer.add(Turn(role="user", content="word " * 100))
    rendered = buffer.render(token_limit=10)
    assert 0 < count_tokens(rendered) <= 10
    assert buffer.render(token_limit=0) == ""


def test_invalid_capacities_are_rejected() -> None:
    with pytest.raises(ValueError):
        WorkingBuffer(max_turns=0)
    with pytest.raises(ValueError):
        WorkingBuffer(max_turns=4, max_tokens=0)
