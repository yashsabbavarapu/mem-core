"""Tier 3 tests: deterministic extraction, SQLite upserts, conflict policy."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from memcore.models import EntityFact, count_tokens
from memcore.semantic import RegexExtractor, SemanticMemory, extract_facts

NOW = 1_700_000_000.0


@pytest.fixture()
def store() -> Iterator[SemanticMemory]:
    memory = SemanticMemory(":memory:")
    yield memory
    memory.close()


def _fact(attribute: str, value: str, confidence: float = 0.9, at: float = NOW) -> EntityFact:
    return EntityFact(
        entity="user",
        attribute=attribute,
        value=value,
        confidence=confidence,
        updated_at=at,
    )


# ------------------------------------------------------------------ extraction


@pytest.mark.parametrize(
    ("text", "attribute", "value"),
    [
        ("My name is Alex.", "name", "Alex"),
        ("My name is Alex Rivera.", "name", "Alex Rivera"),
        ("Hi! My name is Alex and I am a backend engineer.", "name", "Alex"),
        ("We use Python and DuckDB.", "tech_stack", '["Python", "DuckDB"]'),
        ("Our tech stack is Rust, Postgres and Redis.", "tech_stack", '["Rust", "Postgres", "Redis"]'),
        ("My budget is $10k.", "budget", "10000"),
        ("The budget is $5,000 for this quarter.", "budget", "5000"),
        ("Our budget is 2m.", "budget", "2000000"),
        ("The deadline is March 3rd, so we need something simple.", "deadline", "March 3rd"),
        ("I live in Berlin, so reviews land late.", "location", "Berlin"),
        ("I prefer SQL over ORMs for the query layer.", "preference", "SQL over ORMs"),
        ("We can't use Kubernetes.", "constraint", "must not use Kubernetes"),
        ("I am a backend engineer at Acme.", "role", "backend engineer"),
    ],
)
def test_extraction_rules(text: str, attribute: str, value: str) -> None:
    facts = {fact.attribute: fact.value for fact in extract_facts(text)}
    assert facts.get(attribute) == value


def test_extraction_is_deterministic() -> None:
    text = "My name is Alex, we use FastAPI and DuckDB, my budget is $5000."
    first = extract_facts(text, timestamp=NOW)
    second = extract_facts(text, timestamp=NOW)
    assert [fact.model_dump() for fact in first] == [fact.model_dump() for fact in second]


def test_extraction_ignores_unremarkable_text() -> None:
    assert extract_facts("Sounds good, thanks for the help!") == []


def test_extraction_does_not_absorb_neighbouring_clauses() -> None:
    facts = {f.attribute: f.value for f in extract_facts("We use FastAPI, my budget is $900")}
    assert json.loads(facts["tech_stack"]) == ["FastAPI"]
    assert facts["budget"] == "900"


# ---------------------------------------------------------------------- upsert


def test_upsert_inserts_then_reads_back(store: SemanticMemory) -> None:
    assert store.upsert(_fact("name", "Alex")) is True
    stored = store.get("user", "name")
    assert stored is not None
    assert (stored.entity, stored.attribute, stored.value) == ("user", "name", "Alex")
    assert len(store) == 1
    assert store.get("user", "missing") is None


def test_primary_key_is_entity_plus_attribute(store: SemanticMemory) -> None:
    store.upsert(_fact("name", "Alex"))
    store.upsert(EntityFact(entity="company", attribute="name", value="Acme"))
    store.upsert(_fact("budget", "5000"))
    assert len(store) == 3
    assert len(store.all_facts(entity="user")) == 2


def test_equal_confidence_overwrites_on_recency(store: SemanticMemory) -> None:
    store.upsert(_fact("budget", "5000", confidence=0.9, at=NOW))
    assert store.upsert(_fact("budget", "8000", confidence=0.9, at=NOW + 10)) is True

    stored = store.get("user", "budget")
    assert stored is not None
    assert stored.value == "8000"
    assert stored.updated_at == NOW + 10
    assert len(store) == 1  # updated in place, not duplicated


def test_higher_confidence_wins(store: SemanticMemory) -> None:
    store.upsert(_fact("location", "Berlin", confidence=0.6))
    assert store.upsert(_fact("location", "Munich", confidence=0.95)) is True
    stored = store.get("user", "location")
    assert stored is not None and stored.value == "Munich"


def test_lower_confidence_never_clobbers(store: SemanticMemory) -> None:
    store.upsert(_fact("location", "Berlin", confidence=0.9))
    assert store.upsert(_fact("location", "Paris", confidence=0.3)) is False
    stored = store.get("user", "location")
    assert stored is not None and stored.value == "Berlin"
    assert stored.confidence == 0.9


def test_reaffirmation_refreshes_timestamp_without_reporting_a_change(
    store: SemanticMemory,
) -> None:
    store.upsert(_fact("name", "Alex", at=NOW))
    assert store.upsert(_fact("name", "Alex", at=NOW + 60)) is False
    stored = store.get("user", "name")
    assert stored is not None and stored.updated_at == NOW + 60


def test_list_attributes_merge_when_requested(store: SemanticMemory) -> None:
    store.upsert(_fact("tech_stack", '["Python"]', confidence=0.85), merge_lists=True)
    store.upsert(_fact("tech_stack", '["DuckDB", "python"]', confidence=0.85), merge_lists=True)
    stored = store.get("user", "tech_stack")
    assert stored is not None
    assert json.loads(stored.value) == ["Python", "DuckDB"]  # case-insensitive union


def test_list_attributes_overwrite_by_default(store: SemanticMemory) -> None:
    store.upsert(_fact("tech_stack", '["Python"]'))
    store.upsert(_fact("tech_stack", '["Rust"]'))
    stored = store.get("user", "tech_stack")
    assert stored is not None and json.loads(stored.value) == ["Rust"]


def test_ingest_text_persists_extracted_facts(store: SemanticMemory) -> None:
    applied = store.ingest_text("My name is Alex and my budget is $10k", timestamp=NOW)
    assert {fact.attribute for fact in applied} == {"name", "budget"}

    store.ingest_text("We use FastAPI", timestamp=NOW + 1)
    store.ingest_text("We also use DuckDB", timestamp=NOW + 2)
    stored = store.get("user", "tech_stack")
    assert stored is not None
    assert json.loads(stored.value) == ["FastAPI", "DuckDB"]


def test_upsert_many_counts_only_effective_writes(store: SemanticMemory) -> None:
    facts = [_fact("name", "Alex"), _fact("budget", "1000"), _fact("name", "Alex")]
    assert store.upsert_many(facts) == 2


def test_delete_and_clear(store: SemanticMemory) -> None:
    store.upsert(_fact("name", "Alex"))
    assert store.delete("user", "name") is True
    assert store.delete("user", "name") is False
    store.upsert(_fact("budget", "10"))
    store.clear()
    assert len(store) == 0


def test_render_prioritises_confidence_and_respects_budget(store: SemanticMemory) -> None:
    store.upsert(_fact("name", "Alexander Hamilton Rivera", confidence=0.99))
    store.upsert(_fact("preference", "an extremely long winded preference statement", confidence=0.2))

    rendered = store.render(token_limit=12)
    assert count_tokens(rendered) <= 12
    assert "name" in rendered  # highest confidence survives the squeeze
    assert store.render(token_limit=0) == ""
    assert count_tokens(store.render(token_limit=3)) <= 3


def test_facts_survive_reopening_the_database(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    with SemanticMemory(db_path) as first:
        first.ingest_text("My name is Alex", timestamp=NOW)

    with SemanticMemory(db_path) as second:
        stored = second.get("user", "name")
        assert stored is not None and stored.value == "Alex"


# ------------------------------------------------------- production hardening


def test_store_is_safe_to_share_across_threads(tmp_path: Path) -> None:
    """Regression: SQLite objects used off the creating thread used to raise."""
    import threading

    memory = SemanticMemory(tmp_path / "threads.sqlite")
    errors: list[str] = []
    barrier = threading.Barrier(8)

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=5)
            for round_index in range(20):
                memory.ingest_text(f"My name is Worker{index}", timestamp=NOW + round_index)
                memory.upsert(_fact(f"attr{index}", str(round_index), at=NOW + round_index))
                memory.all_facts()
                len(memory)
        except Exception as exc:  # pragma: no cover - only on regression
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(memory) == 9  # 8 per-thread attributes + the shared name
    memory.close()


def test_schema_version_is_recorded_and_future_versions_are_refused(
    tmp_path: Path,
) -> None:
    import sqlite3

    from memcore.semantic import SCHEMA_VERSION

    path = tmp_path / "versioned.sqlite"
    with SemanticMemory(path) as memory:
        assert memory.schema_version == SCHEMA_VERSION

    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="newer mem-core"):
        SemanticMemory(path)


@pytest.mark.parametrize(
    "text",
    [
        "My api_key = sk-abcdef0123456789abcdef",
        "my password: hunter2000000",
        "the card is 4111 1111 1111 1111",
        "token=ghp_abcdefghijklmnopqrstuvwxyz0123",
    ],
)
def test_credentials_are_never_persisted(store: SemanticMemory, text: str) -> None:
    store.ingest_text(text)
    assert all("sk-" not in fact.value and "ghp_" not in fact.value for fact in store.all_facts())
    assert len(store) == 0


def test_secret_shaped_values_are_refused_on_direct_upsert(store: SemanticMemory) -> None:
    assert store.upsert(_fact("api_key", "sk-abcdef0123456789abcdef")) is False
    assert len(store) == 0


def test_retraction_subtracts_from_the_stack(store: SemanticMemory) -> None:
    store.ingest_text("We use Python, DuckDB and Redis", timestamp=NOW)
    assert json.loads(store.get("user", "tech_stack").value) == ["Python", "DuckDB", "Redis"]  # type: ignore[union-attr]

    applied = store.ingest_text("We stopped using Redis", timestamp=NOW + 1)
    assert applied  # the retraction changed the store
    assert json.loads(store.get("user", "tech_stack").value) == ["Python", "DuckDB"]  # type: ignore[union-attr]
    assert store.get("user", "tech_stack_removed") is None  # not stored as a fact


def test_retraction_of_an_unknown_tool_is_a_no_op(store: SemanticMemory) -> None:
    store.ingest_text("We use Python", timestamp=NOW)
    store.ingest_text("We stopped using Cobol", timestamp=NOW + 1)
    assert json.loads(store.get("user", "tech_stack").value) == ["Python"]  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("text", "attribute", "value"),
    [
        ("We ended up going with Postgres.", "tech_stack", '["Postgres"]'),
        ("We switched to Rust.", "tech_stack", '["Rust"]'),
        ("We standardised on Terraform.", "tech_stack", '["Terraform"]'),
        ("My name's Sam.", "name", "Sam"),
        ("Our team is based in Lisbon.", "location", "Lisbon"),
        ("I'm in CET.", "timezone", "CET"),
        ("I am on UTC+2.", "timezone", "UTC+2"),
    ],
)
def test_additional_phrasings(text: str, attribute: str, value: str) -> None:
    facts = {fact.attribute: fact.value for fact in extract_facts(text)}
    assert facts.get(attribute) == value


def test_a_timezone_is_not_mistaken_for_a_place() -> None:
    facts = {fact.attribute: fact.value for fact in extract_facts("We are in PST")}
    assert facts.get("timezone") == "PST"
    assert "location" not in facts


def test_an_acronym_is_not_mistaken_for_a_timezone() -> None:
    assert extract_facts("I'm in NYC") == []
    assert extract_facts("we are in trouble") == []


def test_render_never_emits_a_partial_fact(store: SemanticMemory) -> None:
    store.upsert(_fact("budget", "5000", confidence=0.9))
    store.upsert(_fact("name", "Alexander", confidence=0.95))
    whole_lines = {fact.render() for fact in store.all_facts()}
    for limit in range(1, 30):
        rendered = store.render(limit)
        assert count_tokens(rendered) <= limit
        for line in rendered.split("\n"):
            if line:
                assert line in whole_lines, f"partial fact at limit={limit}: {line!r}"


def test_custom_extractors_compose(store: SemanticMemory) -> None:
    class Sentiment:
        def extract(
            self, text: str, entity: str = "user", timestamp: float | None = None
        ) -> list[EntityFact]:
            if "love" not in text:
                return []
            return [
                EntityFact(
                    entity=entity, attribute="mood", value="positive", confidence=0.6,
                    updated_at=timestamp or NOW,
                )
            ]

    memory = SemanticMemory(":memory:", extractors=[RegexExtractor(), Sentiment()])
    memory.ingest_text("My name is Alex and I love this stack", timestamp=NOW)
    assert memory.get("user", "name") is not None
    stored = memory.get("user", "mood")
    assert stored is not None and stored.value == "positive"
    memory.close()


def test_stats_reports_store_shape(store: SemanticMemory) -> None:
    store.ingest_text("My name is Alex and my budget is $10k")
    stats = store.stats()
    assert stats["facts"] == 2
    assert stats["entities"] == 1
    assert stats["extractors"] == ["RegexExtractor"]
