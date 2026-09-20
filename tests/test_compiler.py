"""Compiler tests: the budget guarantee, allocation, and priority ordering."""

from __future__ import annotations

from pathlib import Path

import pytest

from memcore.compiler import SECTION_HEADERS, ContextCompiler
from memcore.episodic import EpisodicMemory, HashingEmbedder, half_life_to_lambda
from memcore.models import ContextBudget, Role, Turn, count_tokens
from memcore.semantic import SemanticMemory
from memcore.working import WorkingBuffer

NOW = 1_700_000_000.0
HOUR = 3600.0

_CONVERSATION: tuple[tuple[Role, str], ...] = (
    ("user", "My name is Alex and I am a backend engineer in Berlin."),
    ("assistant", "Good to meet you. What is the system you are building?"),
    ("user", "We use FastAPI and DuckDB for an internal analytics service."),
    ("assistant", "That pairing works well for read heavy analytical workloads."),
    ("user", "My budget is $5000 and we can't use Kubernetes at all."),
    ("assistant", "Then a single virtual machine deployment is the pragmatic choice."),
    ("user", "The deadline is March 3rd and I prefer SQL over ORMs."),
    ("assistant", "DuckDB's SQL surface can stay the primary query abstraction."),
    ("user", "We also use Redis to cache the hottest aggregate queries."),
    ("assistant", "Invalidate those cached aggregates on write to avoid staleness."),
)


def _populated() -> tuple[WorkingBuffer, EpisodicMemory, SemanticMemory]:
    working = WorkingBuffer(max_turns=4)
    episodic = EpisodicMemory(
        embedder=HashingEmbedder(dimensions=128),
        lambda_decay=half_life_to_lambda(24.0),
    )
    semantic = SemanticMemory(":memory:")

    for index, (role, content) in enumerate(_CONVERSATION):
        timestamp = NOW - (len(_CONVERSATION) - index) * HOUR
        turn = Turn(role=role, content=content, timestamp=timestamp)
        episodic.ingest_turns(working.add(turn))
        if role == "user":
            semantic.ingest_text(content, timestamp=timestamp)
    return working, episodic, semantic


@pytest.mark.parametrize("budget", [0, 1, 5, 17, 40, 80, 150, 300, 500, 1_000, 1_500, 4_000])
def test_compiled_prompt_never_exceeds_the_budget(budget: int) -> None:
    """The contract: measured tokens <= budget, at every budget size."""
    working, episodic, semantic = _populated()
    compiler = ContextCompiler(ContextBudget(total_token_budget=budget))

    context = compiler.compile(
        "What stack and budget are we working with?", working, episodic, semantic, now=NOW
    )
    measured = count_tokens(context.to_prompt())

    assert measured <= budget
    assert context.total_tokens == measured
    semantic.close()


@pytest.mark.parametrize(
    "ratios",
    [(0.4, 0.4, 0.2), (0.8, 0.1, 0.1), (0.1, 0.1, 0.8), (0.34, 0.33, 0.33)],
)
def test_budget_holds_for_any_ratio_split(ratios: tuple[float, float, float]) -> None:
    working, episodic, semantic = _populated()
    budget = ContextBudget(
        total_token_budget=220,
        working_ratio=ratios[0],
        episodic_ratio=ratios[1],
        entity_ratio=ratios[2],
    )
    context = ContextCompiler(budget).compile("budget?", working, episodic, semantic, now=NOW)
    assert count_tokens(context.to_prompt()) <= 220
    semantic.close()


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        ContextBudget(working_ratio=0.5, episodic_ratio=0.5, entity_ratio=0.5)


def test_allocation_is_exact_and_deterministic() -> None:
    budget = ContextBudget(total_token_budget=1_000)
    allocation = budget.allocate(997)
    assert sum(allocation.values()) == 997  # no tokens lost to rounding
    assert allocation == budget.allocate(997)
    assert sum(budget.allocate(0).values()) == 0


def test_all_three_tiers_are_represented_at_a_generous_budget() -> None:
    working, episodic, semantic = _populated()
    context = ContextCompiler(ContextBudget(total_token_budget=1_500)).compile(
        "what is my stack?", working, episodic, semantic, now=NOW
    )

    assert SECTION_HEADERS["entity"] in context.entity_block
    assert SECTION_HEADERS["episodic"] in context.episodic_block
    assert SECTION_HEADERS["working"] in context.working_block
    assert "user.tech_stack" in context.entity_block
    # The newest turn must always be present verbatim.
    assert _CONVERSATION[-1][1] in context.working_block
    semantic.close()


def test_scarce_budget_drops_episodic_before_entity_facts() -> None:
    working, episodic, semantic = _populated()
    context = ContextCompiler(ContextBudget(total_token_budget=90)).compile(
        "what is my budget?", working, episodic, semantic, now=NOW
    )

    assert count_tokens(context.to_prompt()) <= 90
    assert context.entity_block  # durable constraints are the last to go
    assert count_tokens(context.episodic_block) <= count_tokens(context.entity_block)
    semantic.close()


def test_system_instructions_and_query_are_always_present() -> None:
    working, episodic, semantic = _populated()
    context = ContextCompiler(ContextBudget(total_token_budget=400)).compile(
        "why did we rule out Kubernetes?", working, episodic, semantic, now=NOW
    )
    assert "why did we rule out Kubernetes?" in context.system_instructions
    assert SECTION_HEADERS["system"] in context.system_instructions
    semantic.close()


def test_tiny_budget_truncates_the_mandatory_block_instead_of_overflowing() -> None:
    working, episodic, semantic = _populated()
    context = ContextCompiler(ContextBudget(total_token_budget=6)).compile(
        "hello", working, episodic, semantic, now=NOW
    )
    assert count_tokens(context.to_prompt()) <= 6
    assert context.entity_block == ""
    assert context.episodic_block == ""
    assert context.working_block == ""
    semantic.close()


def test_zero_budget_yields_an_empty_prompt() -> None:
    working, episodic, semantic = _populated()
    context = ContextCompiler(ContextBudget(total_token_budget=0)).compile(
        "hello", working, episodic, semantic, now=NOW
    )
    assert context.to_prompt() == ""
    assert context.total_tokens == 0
    semantic.close()


def test_compiling_with_empty_memories_is_safe() -> None:
    context = ContextCompiler(ContextBudget(total_token_budget=200)).compile("anything")
    assert context.entity_block == ""
    assert context.episodic_block == ""
    assert context.working_block == ""
    assert count_tokens(context.to_prompt()) <= 200


def test_unused_slices_are_donated_to_other_tiers() -> None:
    """An empty entity store must hand its slice to the tiers that can use it."""
    working, episodic, semantic = _populated()
    budget = ContextBudget(total_token_budget=150)

    with_facts = ContextCompiler(budget)
    with_facts.compile("what stack?", working, episodic, semantic, now=NOW)
    before = with_facts.last_trace

    semantic.clear()
    without_facts = ContextCompiler(budget)
    context = without_facts.compile("what stack?", working, episodic, semantic, now=NOW)
    after = without_facts.last_trace

    assert before.used["entity"] > 0
    assert after.used["entity"] == 0
    assert after.refilled  # the freed slice was spent, not wasted
    assert (
        after.used["episodic"] + after.used["working"]
        > before.used["episodic"] + before.used["working"]
    )
    assert count_tokens(context.to_prompt()) <= 150
    semantic.close()


def test_trace_accounting_matches_the_compiled_blocks() -> None:
    working, episodic, semantic = _populated()
    compiler = ContextCompiler(ContextBudget(total_token_budget=500))
    context = compiler.compile("stack and budget?", working, episodic, semantic, now=NOW)

    trace = compiler.last_trace
    assert trace.budget == 500
    assert trace.reserved == count_tokens(context.system_instructions)
    assert trace.total_tokens == context.total_tokens <= 500
    assert "final total" in trace.render()
    semantic.close()


def test_compile_prompt_matches_compile() -> None:
    working, episodic, semantic = _populated()
    compiler = ContextCompiler(ContextBudget(total_token_budget=300))
    prompt = compiler.compile_prompt("stack?", working, episodic, semantic, now=NOW)
    assert prompt == compiler.compile("stack?", working, episodic, semantic, now=NOW).to_prompt()
    semantic.close()


def test_budget_holds_under_adversarially_long_content() -> None:
    working = WorkingBuffer(max_turns=50)
    episodic = EpisodicMemory(embedder=HashingEmbedder(dimensions=64))
    semantic = SemanticMemory(":memory:")
    blob = "supercalifragilistic " * 300
    for index in range(20):
        working.add(Turn(role="user", content=f"{blob} {index}", timestamp=NOW - index))
        episodic.add(f"{blob} episode {index}", timestamp=NOW - index * HOUR)
    semantic.ingest_text(f"My name is {blob}", timestamp=NOW)

    for limit in (25, 100, 512, 1_500):
        context = ContextCompiler(ContextBudget(total_token_budget=limit)).compile(
            blob, working, episodic, semantic, now=NOW
        )
        assert count_tokens(context.to_prompt()) <= limit
    semantic.close()


# ------------------------------------------------------- structure-aware trims


def test_entity_facts_are_never_rendered_partially() -> None:
    """A truncated fact is a fabricated constraint; the compiler must not make one."""
    working, episodic, semantic = _populated()
    whole_lines = {fact.render() for fact in semantic.all_facts()}

    for budget in range(40, 260, 5):
        context = ContextCompiler(ContextBudget(total_token_budget=budget)).compile(
            "what is my budget?", working, episodic, semantic, now=NOW
        )
        lines = context.entity_block.split("\n")[1:]  # skip the section header
        for line in lines:
            if line:
                assert line in whole_lines, f"partial fact at budget={budget}: {line!r}"
        assert count_tokens(context.to_prompt()) <= budget
    semantic.close()


def test_working_block_keeps_the_newest_turn_under_pressure() -> None:
    working, episodic, semantic = _populated()
    newest = _CONVERSATION[-1][1]

    for budget in (120, 160, 200, 260, 320):
        context = ContextCompiler(ContextBudget(total_token_budget=budget)).compile(
            "what did we just say?", working, episodic, semantic, now=NOW
        )
        if context.working_block:
            assert newest[:40] in context.working_block
    semantic.close()


def test_episodic_block_is_summarised_before_it_is_cut() -> None:
    """Low-density recall should be dropped ahead of dense recall."""
    working = WorkingBuffer(max_turns=1)
    episodic = EpisodicMemory(
        embedder=HashingEmbedder(dimensions=128), summarize_on_ingest=False
    )
    semantic = SemanticMemory(":memory:")
    episodic.add(
        "user: Sounds good, thanks! The budget is $5000 and the deadline is March 3rd. "
        "Got it, no problem at all.",
        timestamp=NOW,
    )
    working.add(Turn(role="user", content="remind me", timestamp=NOW))

    context = ContextCompiler(ContextBudget(total_token_budget=90)).compile(
        "budget and deadline?", working, episodic, semantic, now=NOW
    )
    assert count_tokens(context.to_prompt()) <= 90
    if context.episodic_block:
        assert "$5000" in context.episodic_block or "March" in context.episodic_block
    semantic.close()


def test_trace_reports_what_was_reclaimed() -> None:
    working, episodic, semantic = _populated()
    compiler = ContextCompiler(ContextBudget(total_token_budget=110))
    compiler.compile("what is my budget?", working, episodic, semantic, now=NOW)

    trace = compiler.last_trace
    assert trace.total_tokens <= 110
    assert set(trace.trimmed).issubset({"entity", "episodic", "working"})
    semantic.close()


def test_budget_holds_with_a_durable_backing_store(tmp_path: Path) -> None:
    """The guarantee must not depend on where the tiers keep their data."""
    working = WorkingBuffer(max_turns=4)
    episodic = EpisodicMemory(
        path=tmp_path / "episodes.sqlite", embedder=HashingEmbedder(dimensions=128)
    )
    semantic = SemanticMemory(tmp_path / "facts.sqlite")
    for index, (role, content) in enumerate(_CONVERSATION):
        timestamp = NOW - (len(_CONVERSATION) - index) * HOUR
        episodic.ingest_turns(working.add(Turn(role=role, content=content, timestamp=timestamp)))
        if role == "user":
            semantic.ingest_text(content, timestamp=timestamp)

    for budget in (0, 12, 60, 150, 400, 1_200):
        context = ContextCompiler(ContextBudget(total_token_budget=budget)).compile(
            "stack and budget?", working, episodic, semantic, now=NOW
        )
        assert count_tokens(context.to_prompt()) <= budget
    episodic.close()
    semantic.close()
