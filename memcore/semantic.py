"""Tier 3 — Semantic entity memory: durable facts in embedded SQLite.

Where Tier 1 forgets fast and Tier 2 fades with time, Tier 3 does not
decay.  It holds the small set of assertions that must survive forever —
names, stacks, budgets, hard constraints — as ``entity.attribute = value``
rows keyed on ``(entity, attribute)``.

Production characteristics:

* **Thread-safe.** WAL journaling, a busy timeout, and a re-entrant lock
  around every statement, so an async or threaded agent server can share
  one store.
* **Versioned.** The schema carries a ``user_version`` and migrates
  forward on open, so an existing database is never silently misread.
* **Pluggable.** Extraction is a protocol: the built-in deterministic
  regex rules can be layered under a model-based extractor without
  touching the store.
* **Secret-averse.** Values that look like credentials are refused rather
  than persisted to disk.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, runtime_checkable

from memcore.models import EntityFact, count_tokens

__all__ = [
    "DEFAULT_DB_PATH",
    "LIST_ATTRIBUTES",
    "SCHEMA_VERSION",
    "Extractor",
    "RegexExtractor",
    "SemanticMemory",
    "extract_facts",
    "looks_like_secret",
]

logger = logging.getLogger("memcore.semantic")

DEFAULT_DB_PATH: Final[str] = "memory.sqlite"

#: Bumped whenever the table layout changes; migrations run on open.
SCHEMA_VERSION: Final[int] = 1

#: Attributes whose value is a JSON list and which accumulate rather than
#: replace when ``merge_lists`` is enabled.
LIST_ATTRIBUTES: Final[frozenset[str]] = frozenset({"tech_stack"})

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS entities (
    entity     TEXT NOT NULL,
    attribute  TEXT NOT NULL,
    value      TEXT NOT NULL,
    confidence REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (entity, attribute)
);
CREATE INDEX IF NOT EXISTS idx_entities_updated_at ON entities (updated_at DESC);
"""

# --------------------------------------------------------------------------
# Credential guard
# --------------------------------------------------------------------------

_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b(?:sk|pk|api|key|token|secret|passwd|password)[-_ ]?[:=]", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # payment card shaped
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),  # long opaque blob
)


def looks_like_secret(text: str) -> bool:
    """Heuristically detect credentials that must never be persisted.

    Memory is written to disk and replayed into prompts, so a false
    positive (refusing to store something innocuous) is far cheaper than a
    false negative (persisting an API key).
    """
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


# --------------------------------------------------------------------------
# Deterministic extraction rules
# --------------------------------------------------------------------------

_SPLIT_LIST = re.compile(r"\s*(?:,|/|\band\b|\bplus\b|&)\s*", re.IGNORECASE)
_STOP_TAIL = re.compile(
    r"\b(?:for|to|because|since|when|while|but|so)\b.*$"
    # A new clause ("..., my budget is 5000") ends the list.
    r"|\b(?:my|our|the|i|we)\s+[\w\s]{0,20}?\b(?:is|are|was|costs?)\b.*$"
    # ...as does a conjoined subject ("Berlin and I'm in CET").
    r"|\band\s+(?:i|we)\s*(?:'m|'re|am|are)\b.*$",
    re.IGNORECASE,
)

#: Words that mark a captured fragment as a clause, not a tool name.
_CLAUSE_VERBS: Final[frozenset[str]] = frozenset(
    {"is", "are", "was", "were", "be", "have", "has", "need", "needs", "want", "wants"}
)

_NAME_RE = re.compile(
    r"(?i:\b(?:my name is|my name's|i am called|call me)\s+)"
    # Trigger matching is case-insensitive; the name itself is not, so a
    # following lowercase word ("Alex and I...") is not absorbed as a surname.
    r"(?P<value>[A-Za-z][\w'\-]*(?:\s+[A-Z][\w'\-]*)?)"
)
_STACK_RE = re.compile(
    r"\b(?:we|i)\s+(?:also\s+|currently\s+|mostly\s+|now\s+|already\s+)?"
    r"(?:use|are using|'re using|work with|build with|run on|run)\s+(?P<value>[^.!?;]+)",
    re.IGNORECASE,
)
_STACK_IS_RE = re.compile(
    r"\b(?:our|my|the)\s+(?:tech\s+)?stack\s+is\s+(?P<value>[^.!?;]+)",
    re.IGNORECASE,
)
_STACK_SWITCH_RE = re.compile(
    r"\b(?:we|i)\s+(?:switched to|moved to|migrated to|ended up (?:going )?with|"
    r"settled on|standardi[sz]ed on)\s+(?P<value>[^.!?;]+)",
    re.IGNORECASE,
)
_STACK_DROP_RE = re.compile(
    r"\b(?:we|i)\s+(?:no longer use|stopped using|dropped|removed|are not using|"
    r"aren't using|don't use)\s+(?P<value>[^.!?;]+)",
    re.IGNORECASE,
)
_BUDGET_RE = re.compile(
    r"\bbudget\b[^.$\d]{0,20}\$?\s*(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>k|m|000)?",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"\b(?:deadline|due date)\s*(?:is|:)?\s*(?P<value>[^.!?;]+)"
    r"|\b(?:we|i)\s+(?:need to |have to |must )?ship(?:\s+it)?\s+by\s+(?P<value2>[^.!?;]+)",
    re.IGNORECASE,
)
_LOCATION_RE = re.compile(
    r"\bi(?:'m| am)?\s+(?:live\s+in|based\s+in|located\s+in)\s+(?P<value>[^.!?;]+)",
    re.IGNORECASE,
)
_TEAM_LOCATION_RE = re.compile(
    # Trigger is case-insensitive; the place name must still be capitalised,
    # which is what keeps "we are in trouble" from becoming a location.
    r"(?i:\b(?:our team|the team|we)\s+(?:is|are)\s+(?:based\s+in|in)\s+)"
    r"(?P<value>[A-Z][^.!?;,]*)"
)
_ROLE_RE = re.compile(
    r"\bi(?:'m| am)\s+an?\s+(?P<value>[a-z][\w\s\-]{2,40}?)(?=\s+(?:at|for|in)\b|[.,!?;]|$)",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\bi\s+(?:prefer|really like|'d rather use|would rather use)\s+(?P<value>[^.!?;]+)",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(?:we|i)\s+(?P<negation>can't|cannot|can not|must not|won't|will not|don't want to)"
    r"\s+(?P<value>[^.!?;]+)",
    re.IGNORECASE,
)
_TIMEZONE_RE = re.compile(
    r"(?i:\b(?:i(?:'m| am)?|we(?:'re| are)?)\s+(?:in|on)\s+)"
    # An explicit allowlist, not a generic acronym match: "I'm in NYC"
    # is a place, not a timezone.
    r"(?P<value>UTC[+-]\d{1,2}|GMT[+-]\d{1,2}|CES?T|[CEMP][SD]T|BST|IST|JST|AES?T|AEDT|UTC|GMT)\b"
)


def _clean(text: str) -> str:
    return " ".join(text.split()).strip(" .,;:-")


def _clean_clause(text: str) -> str:
    """Clean a free-text value and drop any trailing subordinate clause."""
    return _clean(_STOP_TAIL.sub("", text))


def _looks_like_technology(item: str) -> bool:
    """Reject list members that are clearly prose rather than a tool name.

    Tool names are short and symbol-free; anything long, monetary, or
    verb-bearing is a run-on clause the regex over-captured.
    """
    if not item or len(item) > 40:
        return False
    if "$" in item or "%" in item:
        return False
    words = item.split()
    if len(words) > 3:
        return False
    return not any(word.lower() in _CLAUSE_VERBS for word in words)


def _as_list(raw: str) -> list[str]:
    trimmed = _STOP_TAIL.sub("", raw)
    parts = [_clean(part) for part in _SPLIT_LIST.split(trimmed)]
    return [part for part in parts if _looks_like_technology(part)]


def _as_list_value(raw: str) -> str:
    """Normalise ``"Python, DuckDB and FastAPI"`` into a JSON list string."""
    return json.dumps(_as_list(raw))


def _as_budget_value(amount: str, unit: str | None) -> str:
    """Normalise ``$10k`` / ``5,000`` into a plain integer-ish string."""
    number = float(amount.replace(",", ""))
    suffix = (unit or "").lower()
    if suffix == "k":
        number *= 1_000
    elif suffix == "m":
        number *= 1_000_000
    elif suffix == "000":
        number *= 1_000
    return str(int(number)) if number.is_integer() else str(number)


@runtime_checkable
class Extractor(Protocol):
    """Turns an utterance into candidate facts.

    Implement this to layer a model-based extractor over (or under) the
    deterministic rules: the store composes extractors in order and the
    normal confidence-gated conflict policy arbitrates between them.
    """

    def extract(
        self, text: str, entity: str = "user", timestamp: float | None = None
    ) -> list[EntityFact]:
        """Return the facts asserted by ``text``."""
        ...


def extract_facts(
    text: str,
    entity: str = "user",
    timestamp: float | None = None,
) -> list[EntityFact]:
    """Parse ``text`` into durable entity facts using deterministic rules.

    Confidence reflects how explicit the phrasing is: a direct declaration
    ("my budget is $10k") scores higher than a loose inference ("I'm a
    backend engineer"), which matters because conflict resolution is
    confidence-gated.

    Values that look like credentials are dropped, never stored.
    """
    when = time.time() if timestamp is None else timestamp
    facts: list[EntityFact] = []

    def emit(attribute: str, value: str, confidence: float) -> None:
        if not value or value in {"[]", '""'}:
            return
        if looks_like_secret(value):
            logger.warning("refusing to store secret-shaped value for %s.%s", entity, attribute)
            return
        facts.append(
            EntityFact(
                entity=entity,
                attribute=attribute,
                value=value,
                confidence=confidence,
                updated_at=when,
            )
        )

    name = _NAME_RE.search(text)
    if name:
        emit("name", _clean(name.group("value")), 0.95)

    stack = _STACK_IS_RE.search(text) or _STACK_SWITCH_RE.search(text) or _STACK_RE.search(text)
    if stack:
        emit("tech_stack", _as_list_value(stack.group("value")), 0.85)

    dropped = _STACK_DROP_RE.search(text)
    if dropped:
        # Retractions are facts too: recorded so the store can subtract
        # them from the accumulated stack.
        emit("tech_stack_removed", _as_list_value(dropped.group("value")), 0.85)

    budget = _BUDGET_RE.search(text)
    if budget:
        emit("budget", _as_budget_value(budget.group("value"), budget.group("unit")), 0.9)

    deadline = _DEADLINE_RE.search(text)
    if deadline:
        raw = deadline.group("value") or deadline.group("value2") or ""
        emit("deadline", _clean_clause(raw), 0.8)

    # Timezone is resolved first so "we are in PST" is not also recorded as
    # a place called PST.
    timezone = _TIMEZONE_RE.search(text)
    timezone_value = _clean(timezone.group("value")) if timezone else None
    if timezone_value:
        emit("timezone", timezone_value, 0.8)

    location = _LOCATION_RE.search(text)
    if location:
        place = _clean_clause(location.group("value"))
        if place != timezone_value:
            emit("location", place, 0.85)
    else:
        team = _TEAM_LOCATION_RE.search(text)
        if team:
            place = _clean_clause(team.group("value"))
            if place != timezone_value:
                emit("location", place, 0.75)

    role = _ROLE_RE.search(text)
    if role and not name:
        emit("role", _clean(role.group("value")), 0.7)

    preference = _PREFERENCE_RE.search(text)
    if preference:
        emit("preference", _clean_clause(preference.group("value")), 0.75)

    constraint = _CONSTRAINT_RE.search(text)
    if constraint:
        # Normalise every negated form to "must not X" so the stored fact
        # cannot be read as permission.
        emit("constraint", f"must not {_clean(constraint.group('value'))}", 0.9)

    return facts


class RegexExtractor:
    """The default :class:`Extractor`: deterministic rules, no model call."""

    def extract(
        self, text: str, entity: str = "user", timestamp: float | None = None
    ) -> list[EntityFact]:
        return extract_facts(text, entity=entity, timestamp=timestamp)


class SemanticMemory:
    """SQLite-backed store of :class:`EntityFact` rows.

    Args:
        db_path: Path to the SQLite file, or ``":memory:"`` for an
            ephemeral store.
        extractors: Extraction pipeline; defaults to
            :class:`RegexExtractor`.  Later extractors can override earlier
            ones only by asserting higher confidence.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        extractors: Sequence[Extractor] | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.extractors: list[Extractor] = (
            list(extractors) if extractors is not None else [RegexExtractor()]
        )
        self._lock = threading.RLock()
        # check_same_thread=False plus the lock lets one store be shared by
        # a threaded or async server; SQLite itself is serialised by WAL.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Bring an existing database up to :data:`SCHEMA_VERSION`."""
        row = self._conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row is not None else 0
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.db_path} was written by a newer mem-core "
                f"(schema v{version} > v{SCHEMA_VERSION}); upgrade the package"
            )
        if version < SCHEMA_VERSION:
            # v0 -> v1 is the initial layout, already applied by _SCHEMA.
            # Future migrations append here, each guarded by its version.
            logger.info("migrating %s: schema v%d -> v%d", self.db_path, version, SCHEMA_VERSION)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @property
    def schema_version(self) -> int:
        """The on-disk schema version."""
        with self._lock:
            row = self._conn.execute("PRAGMA user_version").fetchone()
            return int(row[0]) if row is not None else 0

    # ---------------------------------------------------------------- writes

    def upsert(self, fact: EntityFact, merge_lists: bool = False) -> bool:
        """Insert or update a fact; return ``True`` if the store changed.

        Conflict policy: a new value wins when its confidence is greater
        than *or equal to* the stored value's, so a later restatement of an
        equally-trusted fact overwrites the earlier one (recency breaks
        ties).  A lower-confidence assertion never clobbers a
        higher-confidence one.
        """
        if looks_like_secret(fact.value):
            logger.warning(
                "refusing to store secret-shaped value for %s.%s", fact.entity, fact.attribute
            )
            return False

        with self._lock:
            existing = self.get(fact.entity, fact.attribute)
            value = fact.value
            if existing is not None:
                if fact.confidence < existing.confidence:
                    return False
                if merge_lists and fact.attribute in LIST_ATTRIBUTES:
                    value = _merge_list_values(existing.value, fact.value)
                if value == existing.value and fact.confidence == existing.confidence:
                    # Still refresh the timestamp: the fact was reaffirmed.
                    self._conn.execute(
                        "UPDATE entities SET updated_at = ? WHERE entity = ? AND attribute = ?",
                        (fact.updated_at, fact.entity, fact.attribute),
                    )
                    self._conn.commit()
                    return False

            self._conn.execute(
                """
                INSERT INTO entities (entity, attribute, value, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(entity, attribute) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (fact.entity, fact.attribute, value, fact.confidence, fact.updated_at),
            )
            self._conn.commit()
            return True

    def upsert_many(self, facts: Iterable[EntityFact], merge_lists: bool = False) -> int:
        """Upsert a batch of facts; return how many changed the store."""
        with self._lock:
            return sum(1 for fact in facts if self.upsert(fact, merge_lists=merge_lists))

    def ingest_text(
        self,
        text: str,
        entity: str = "user",
        timestamp: float | None = None,
    ) -> list[EntityFact]:
        """Extract facts from ``text`` and persist the ones that win conflicts.

        Retractions ("we stopped using Redis") subtract from list-valued
        attributes instead of being stored as facts of their own.
        """
        candidates: list[EntityFact] = []
        for extractor in self.extractors:
            candidates.extend(extractor.extract(text, entity=entity, timestamp=timestamp))

        applied: list[EntityFact] = []
        with self._lock:
            for fact in candidates:
                if fact.attribute == "tech_stack_removed":
                    removed = self._subtract_from_list(
                        fact.entity, "tech_stack", fact.value, fact.updated_at
                    )
                    if removed is not None:
                        applied.append(removed)
                    continue
                if self.upsert(fact, merge_lists=True):
                    stored = self.get(fact.entity, fact.attribute)
                    applied.append(stored if stored is not None else fact)
        return applied

    def _subtract_from_list(
        self, entity: str, attribute: str, removals_json: str, updated_at: float
    ) -> EntityFact | None:
        """Remove members from a list-valued attribute (retraction handling)."""
        existing = self.get(entity, attribute)
        if existing is None:
            return None
        try:
            current = [str(item) for item in json.loads(existing.value)]
            removals = {str(item).casefold() for item in json.loads(removals_json)}
        except json.JSONDecodeError:
            return None
        remaining = [item for item in current if item.casefold() not in removals]
        if remaining == current:
            return None

        self._conn.execute(
            "UPDATE entities SET value = ?, updated_at = ? WHERE entity = ? AND attribute = ?",
            (json.dumps(remaining), updated_at, entity, attribute),
        )
        self._conn.commit()
        logger.info("retraction: %s.%s -> %s", entity, attribute, remaining)
        return self.get(entity, attribute)

    def delete(self, entity: str, attribute: str) -> bool:
        """Remove a single fact; return whether a row was deleted."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM entities WHERE entity = ? AND attribute = ?",
                (entity, attribute),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Drop every stored fact."""
        with self._lock:
            self._conn.execute("DELETE FROM entities")
            self._conn.commit()

    # ---------------------------------------------------------------- reads

    def get(self, entity: str, attribute: str) -> EntityFact | None:
        """Fetch one fact, or ``None`` when it is not stored."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE entity = ? AND attribute = ?",
                (entity, attribute),
            ).fetchone()
        return _row_to_fact(row) if row is not None else None

    def all_facts(self, entity: str | None = None) -> list[EntityFact]:
        """All facts, most recently updated first."""
        with self._lock:
            if entity is None:
                rows = self._conn.execute(
                    "SELECT * FROM entities ORDER BY updated_at DESC, entity, attribute"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM entities WHERE entity = ? "
                    "ORDER BY updated_at DESC, attribute",
                    (entity,),
                ).fetchall()
        return [_row_to_fact(row) for row in rows]

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()
            return int(row["n"])

    def __iter__(self) -> Iterator[EntityFact]:
        return iter(self.all_facts())

    def render(self, token_limit: int, entity: str | None = None) -> str:
        """Render facts as prompt lines, highest-confidence first.

        Facts are **atomic**: a line is emitted whole or not at all.  A
        truncated ``user.budget = 500`` would be a fabrication, so when the
        slice cannot hold a fact the fact is dropped instead.
        """
        if token_limit <= 0:
            return ""
        facts = sorted(
            self.all_facts(entity),
            key=lambda fact: (fact.confidence, fact.updated_at),
            reverse=True,
        )
        lines: list[str] = []
        used = 0
        for fact in facts:
            line = fact.render()
            cost = count_tokens(line) + (1 if lines else 0)
            if used + cost <= token_limit:
                lines.append(line)
                used += cost
        return "\n".join(lines)

    def stats(self) -> dict[str, object]:
        """Operational counters for logging or a metrics endpoint."""
        with self._lock:
            entities = self._conn.execute(
                "SELECT COUNT(DISTINCT entity) AS n FROM entities"
            ).fetchone()
            return {
                "facts": len(self),
                "entities": int(entities["n"]),
                "path": self.db_path,
                "schema_version": self.schema_version,
                "extractors": [type(item).__name__ for item in self.extractors],
            }

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SemanticMemory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _row_to_fact(row: sqlite3.Row) -> EntityFact:
    return EntityFact(
        entity=str(row["entity"]),
        attribute=str(row["attribute"]),
        value=str(row["value"]),
        confidence=float(row["confidence"]),
        updated_at=float(row["updated_at"]),
    )


def _merge_list_values(existing: str, incoming: str) -> str:
    """Union two JSON list values, preserving first-seen order."""
    try:
        old = json.loads(existing)
        new = json.loads(incoming)
    except json.JSONDecodeError:
        return incoming
    if not isinstance(old, list) or not isinstance(new, list):
        return incoming
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*old, *new]:
        text = str(item)
        if text.casefold() not in seen:
            seen.add(text.casefold())
            merged.append(text)
    return json.dumps(merged)
