"""Tests for deterministic extractive compression."""

from __future__ import annotations

import pytest

from memcore.models import count_tokens
from memcore.summarize import (
    compress_turns,
    density_score,
    sentence_split,
    summarize_to_tokens,
    trim_lines_to_tokens,
)

MIXED = (
    "Sounds good, thanks! | My budget is $5000 and the deadline is March 3rd. | "
    "Got it, no problem. | We deploy FastAPI on a single VM in Frankfurt."
)

FACTS = "\n".join(
    [
        "## DURABLE FACTS (entity memory)",
        "user.name = Alex",
        "user.budget = 5000",
        "user.constraint = must not use Kubernetes",
        'user.tech_stack = ["FastAPI", "DuckDB"]',
    ]
)


# ------------------------------------------------------------------ density


def test_filler_scores_below_informative_text() -> None:
    assert density_score("Sounds good, thanks!") < density_score(
        "My budget is $5000 and the deadline is March 3rd."
    )
    assert density_score("Got it, no problem.") < 0.0


def test_numbers_and_proper_nouns_raise_density() -> None:
    assert density_score("we deployed it") < density_score("we deployed 3 VMs in Frankfurt")
    assert density_score("") == 0.0


def test_sentence_split_strips_turn_separators() -> None:
    units = sentence_split(MIXED)
    assert not any(unit.startswith("|") for unit in units)
    assert "Sounds good, thanks!" in units
    assert sentence_split("   ") == []


# ---------------------------------------------------------------- summarize


@pytest.mark.parametrize("limit", [0, 1, 4, 10, 18, 30, 60, 500])
def test_summary_never_exceeds_the_limit(limit: int) -> None:
    assert count_tokens(summarize_to_tokens(MIXED, limit)) <= limit


def test_filler_is_dropped_before_facts() -> None:
    summary = summarize_to_tokens(MIXED, 30)
    assert "$5000" in summary
    assert "Sounds good" not in summary
    assert "no problem" not in summary


def test_a_truncated_fact_beats_an_intact_pleasantry() -> None:
    """At a budget too small for any whole unit, keep the densest one."""
    summary = summarize_to_tokens(MIXED, 10)
    assert "budget" in summary
    assert "Sounds good" not in summary


def test_surviving_units_keep_their_original_order() -> None:
    summary = summarize_to_tokens(MIXED, 30)
    assert summary.index("$5000") < summary.index("Frankfurt")


def test_text_within_budget_is_returned_untouched() -> None:
    assert summarize_to_tokens(MIXED, 10_000) == MIXED


def test_header_is_preserved_when_requested() -> None:
    block = "## RECALLED EPISODES\n" + MIXED
    summary = summarize_to_tokens(block, 30, keep_first_line=True)
    assert summary.startswith("## RECALLED EPISODES")
    assert count_tokens(summary) <= 30


def test_summarization_is_deterministic() -> None:
    assert summarize_to_tokens(MIXED, 25) == summarize_to_tokens(MIXED, 25)


# --------------------------------------------------------------- trim lines


@pytest.mark.parametrize("limit", [0, 3, 6, 9, 14, 20, 30, 200])
def test_trim_never_exceeds_the_limit(limit: int) -> None:
    assert count_tokens(trim_lines_to_tokens(FACTS, limit)) <= limit


def test_facts_are_atomic_never_partial() -> None:
    """The core safety property: no fabricated half-facts."""
    for limit in range(6, 40):
        out = trim_lines_to_tokens(FACTS, limit)
        for line in out.split("\n")[1:]:  # skip the header
            assert line in FACTS.split("\n"), f"partial fact at limit={limit}: {line!r}"


def test_header_survives_and_last_lines_go_first() -> None:
    out = trim_lines_to_tokens(FACTS, 20)
    assert out.startswith("## DURABLE FACTS")
    assert "user.name = Alex" in out
    assert "user.tech_stack" not in out  # dropped from the end


def test_drop_from_start_preserves_the_newest_lines() -> None:
    turns = "## RECENT TURNS\nuser: oldest\nassistant: middle\nuser: newest"
    out = trim_lines_to_tokens(turns, 12, drop_from="start")
    assert "newest" in out
    assert "oldest" not in out


def test_trim_falls_back_when_even_the_header_overflows() -> None:
    out = trim_lines_to_tokens(FACTS, 4)
    assert count_tokens(out) <= 4
    assert out  # something is emitted rather than nothing


# ------------------------------------------------------------ compress_turns


def test_compress_turns_drops_filler_turns() -> None:
    rendered = [
        "user: My budget is $5000",
        "assistant: Sounds good, thanks!",
        "user: We deploy FastAPI in Frankfurt",
    ]
    compressed = compress_turns(rendered)
    assert "$5000" in compressed
    assert "Sounds good" not in compressed


def test_compress_turns_keeps_everything_when_all_is_filler() -> None:
    rendered = ["assistant: Sounds good, thanks!", "assistant: Got it."]
    assert compress_turns(rendered)


def test_compress_turns_respects_a_token_cap() -> None:
    rendered = [f"user: statement number {index} about the migration" for index in range(20)]
    assert count_tokens(compress_turns(rendered, max_tokens=40)) <= 40
