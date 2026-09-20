"""Core data models for the ``mem-core`` hierarchical memory engine.

Every tier of the memory hierarchy speaks in terms of the models defined
here, so this module also hosts the single token-accounting primitive
(:func:`count_tokens`) that the budget compiler relies on.  Keeping one
counter in one place is what makes the budget guarantee auditable: if the
compiler and the tiers disagreed about what a token is, the guarantee would
be meaningless.
"""

from __future__ import annotations

import math
import time
import uuid
from functools import lru_cache
from typing import Any, Final, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "CHARS_PER_TOKEN",
    "CompiledContext",
    "ContextBudget",
    "EntityFact",
    "EpisodicChunk",
    "Role",
    "Turn",
    "count_tokens",
    "truncate_to_tokens",
]

Role = Literal["user", "assistant", "system"]

#: Fallback heuristic used when ``tiktoken`` is unavailable: English prose
#: averages roughly four characters per BPE token.
CHARS_PER_TOKEN: Final[int] = 4


@lru_cache(maxsize=1)
def _encoder() -> Any | None:
    """Return a cached ``tiktoken`` encoder, or ``None`` if unavailable.

    ``tiktoken`` is an optional extra.  The engine must stay usable with a
    zero-dependency install, so a missing encoder silently degrades to the
    character heuristic rather than raising.
    """
    try:  # pragma: no cover - depends on optional extra being installed
        import tiktoken
    except Exception:
        return None
    try:  # pragma: no cover - network/cache failure at first use
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Count the tokens in ``text``.

    Uses ``tiktoken``'s ``cl100k_base`` encoding when installed and falls
    back to ``ceil(len(text) / 4)`` otherwise.  The fallback is deliberately
    a ceiling: over-estimating tokens can only make the compiler pack *less*
    context, never overflow the budget.
    """
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return math.ceil(len(text) / CHARS_PER_TOKEN)
    tokens: list[int] = enc.encode(text)
    return len(tokens)


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "…") -> str:
    """Shrink ``text`` until it costs at most ``max_tokens`` tokens.

    Implemented as a binary search over character prefixes: it is tokenizer
    agnostic (works for both the real BPE and the heuristic) and always
    terminates with a *verified* measurement rather than an estimate.
    """
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    suffix = suffix if count_tokens(suffix) <= max_tokens else ""
    lo, hi, best = 0, len(text), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


class Turn(BaseModel):
    """One verbatim interaction in the conversation (Tier 1 unit)."""

    role: Role
    content: str
    timestamp: float = Field(default_factory=time.time)
    # Defaulted for constructor ergonomics; the "before" validator below
    # fills in the real count whenever the caller omits it.
    token_count: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _default_token_count(cls, data: Any) -> Any:
        """Derive ``token_count`` from ``content`` when not supplied."""
        if isinstance(data, dict) and "token_count" not in data:
            data = {**data, "token_count": count_tokens(str(data.get("content", "")))}
        return data

    def render(self) -> str:
        """Render the turn the way it is packed into a prompt."""
        return f"{self.role}: {self.content}"


class EntityFact(BaseModel):
    """A durable ``entity.attribute = value`` assertion (Tier 3 unit)."""

    entity: str
    attribute: str
    value: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    updated_at: float = Field(default_factory=time.time)

    @property
    def key(self) -> tuple[str, str]:
        """Primary key of the fact within the semantic store."""
        return (self.entity, self.attribute)

    def render(self) -> str:
        """Render the fact as a compact prompt line."""
        return f"{self.entity}.{self.attribute} = {self.value}"


class EpisodicChunk(BaseModel):
    """An embedded passage of past conversation (Tier 2 unit)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    embedding: list[float] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
    decay_score: float = 1.0

    def render(self) -> str:
        """Render the chunk as a prompt line."""
        return f"- {self.content}"


class ContextBudget(BaseModel):
    """Hard token budget and the per-tier split of that budget."""

    total_token_budget: int = Field(default=1500, ge=0)
    working_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    episodic_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    entity_ratio: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> ContextBudget:
        total = self.working_ratio + self.episodic_ratio + self.entity_ratio
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"tier ratios must sum to 1.0, got {total:.6f} "
                f"(working={self.working_ratio}, episodic={self.episodic_ratio}, "
                f"entity={self.entity_ratio})"
            )
        return self

    def allocate(self, available: int) -> dict[str, int]:
        """Split ``available`` tokens across the three tiers.

        Uses largest-remainder rounding so the slices sum to exactly
        ``available`` and no token is lost to floor division.
        """
        available = max(0, available)
        raw = {
            "entity": available * self.entity_ratio,
            "episodic": available * self.episodic_ratio,
            "working": available * self.working_ratio,
        }
        floors = {tier: int(value) for tier, value in raw.items()}
        remainder = available - sum(floors.values())
        # Hand leftover tokens to the tiers with the largest fractional part;
        # ties break on tier priority (entity first) for determinism.
        order = sorted(
            raw,
            key=lambda tier: (-(raw[tier] - floors[tier]), _TIER_PRIORITY[tier]),
        )
        for tier in order[:remainder]:
            floors[tier] += 1
        return floors


#: Packing priority when budget is scarce: durable constraints first,
#: conversational immediacy second, recalled history last.
_TIER_PRIORITY: Final[dict[str, int]] = {"entity": 0, "working": 1, "episodic": 2}


class CompiledContext(BaseModel):
    """The final, budget-checked prompt produced by the compiler."""

    system_instructions: str = ""
    entity_block: str = ""
    episodic_block: str = ""
    working_block: str = ""
    total_tokens: int = Field(default=0, ge=0)

    @field_validator("total_tokens")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        return value

    def to_prompt(self) -> str:
        """Concatenate the non-empty blocks into the prompt string."""
        blocks = [
            self.system_instructions,
            self.entity_block,
            self.episodic_block,
            self.working_block,
        ]
        return "\n\n".join(block for block in blocks if block.strip())
