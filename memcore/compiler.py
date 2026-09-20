"""The Context Budget Compiler: deterministic packing under a hard limit.

The compiler is the component that makes the hierarchy safe to deploy.  It
takes a query plus the three tiers and emits a prompt whose measured token
count is *guaranteed* to satisfy ``total_tokens <= total_token_budget`` —
not estimated, measured, with a final verify-and-trim pass that cannot be
skipped.

Packing proceeds in four phases:

1. **Reserve.**  System instructions and the current query are mandatory;
   they are charged against the budget before anything else is considered.
2. **Allocate.**  What remains is split by the :class:`ContextBudget`
   ratios (largest-remainder rounding, so no token is lost).
3. **Refill.**  A tier that cannot spend its slice (e.g. an empty entity
   store) donates the remainder to the other tiers, in priority order
   ``entity -> working -> episodic``.
4. **Enforce.**  The assembled prompt is measured.  Any overflow — from
   separators or tokenizer non-additivity — is reclaimed from the
   lowest-priority block first, until the measurement passes.

Reclaiming is *structure-aware*, which matters more than it sounds:

* Entity facts are **atomic**.  A fact is dropped whole rather than cut,
  because a truncated ``user.budget = 5000`` reads as ``user.budget = 500``
  — a fabricated constraint is worse than a missing one.
* Recent turns are trimmed from the **oldest** end, preserving the live
  exchange.
* Episodic recall is **summarised** before it is cut: low-density
  sentences ("Sounds good, thanks!") are discarded ahead of dense ones
  ("the budget is $5,000").

Only when all three fail does the compiler fall back to hard character
truncation, and the final measured check runs regardless, so the budget
contract holds unconditionally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from memcore.episodic import EpisodicMemory
from memcore.models import (
    CompiledContext,
    ContextBudget,
    count_tokens,
    truncate_to_tokens,
)
from memcore.semantic import SemanticMemory
from memcore.summarize import summarize_to_tokens, trim_lines_to_tokens
from memcore.working import WorkingBuffer

__all__ = [
    "DEFAULT_SYSTEM_INSTRUCTIONS",
    "SECTION_HEADERS",
    "CompilationTrace",
    "ContextCompiler",
]

logger = logging.getLogger("memcore.compiler")

DEFAULT_SYSTEM_INSTRUCTIONS: Final[str] = (
    "You are an assistant with hierarchical memory. Treat DURABLE FACTS as "
    "binding constraints, RECALLED EPISODES as background, and RECENT TURNS "
    "as the live conversation."
)

SECTION_HEADERS: Final[dict[str, str]] = {
    "system": "## SYSTEM",
    "query": "## CURRENT QUERY",
    "entity": "## DURABLE FACTS (entity memory)",
    "episodic": "## RECALLED EPISODES (similarity x recency)",
    "working": "## RECENT TURNS (verbatim)",
}

#: Lowest priority is trimmed first when the measured prompt overflows.
_TRIM_ORDER: Final[tuple[str, ...]] = ("episodic", "working", "entity")

_BLOCK_SEPARATOR: Final[str] = "\n\n"


@dataclass
class CompilationTrace:
    """Diagnostics for one compile call — what each tier was given and used."""

    budget: int = 0
    reserved: int = 0
    available: int = 0
    allocated: dict[str, int] = field(default_factory=dict)
    used: dict[str, int] = field(default_factory=dict)
    refilled: dict[str, int] = field(default_factory=dict)
    trimmed: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0

    def render(self) -> str:
        """Human-readable one-block summary for the CLI."""
        lines = [
            f"budget            : {self.budget}",
            f"reserved (system) : {self.reserved}",
            f"available to tiers: {self.available}",
        ]
        for tier in ("entity", "episodic", "working"):
            allocated = self.allocated.get(tier, 0)
            refilled = self.refilled.get(tier, 0)
            used = self.used.get(tier, 0)
            trimmed = self.trimmed.get(tier, 0)
            lines.append(
                f"  {tier:<9}: allocated={allocated:<5} refill=+{refilled:<4} "
                f"used={used:<5} trimmed={trimmed}"
            )
        lines.append(f"final total       : {self.total_tokens} / {self.budget}")
        return "\n".join(lines)


class ContextCompiler:
    """Packs the three memory tiers into a strictly budget-bounded prompt."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
        top_k_episodes: int = 5,
    ) -> None:
        self.budget = budget if budget is not None else ContextBudget()
        self.system_instructions = system_instructions
        self.top_k_episodes = top_k_episodes
        self.last_trace = CompilationTrace()

    # ------------------------------------------------------------------ api

    def compile(
        self,
        query: str,
        working: WorkingBuffer | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        now: float | None = None,
    ) -> CompiledContext:
        """Compile a prompt that provably fits the configured budget."""
        total_budget = self.budget.total_token_budget
        trace = CompilationTrace(budget=total_budget)

        system_block = self._render_system(query)
        reserved = count_tokens(system_block)
        if reserved >= total_budget:
            # Degenerate budget: the mandatory header alone fills (or
            # overflows) it. Truncate and emit nothing else.
            system_block = truncate_to_tokens(system_block, total_budget)
            trace.reserved = count_tokens(system_block)
            trace.total_tokens = trace.reserved
            self.last_trace = trace
            return CompiledContext(
                system_instructions=system_block, total_tokens=trace.reserved
            )

        trace.reserved = reserved
        # Charge for the "\n\n" separators that will join the tier blocks.
        separator_cost = count_tokens(_BLOCK_SEPARATOR) * 3
        available = max(0, total_budget - reserved - separator_cost)
        trace.available = available

        allocation = self.budget.allocate(available)
        trace.allocated = dict(allocation)

        blocks = {
            tier: self._render_tier(tier, allocation[tier], query, working, episodic, semantic, now)
            for tier in ("entity", "episodic", "working")
        }
        used = {tier: count_tokens(text) for tier, text in blocks.items()}

        # Phase 3 — refill: spend whatever the tiers left on the table.
        leftover = available - sum(used.values())
        for tier in ("entity", "working", "episodic"):
            if leftover <= 0:
                break
            extra = allocation[tier] + leftover
            candidate = self._render_tier(
                tier, extra, query, working, episodic, semantic, now
            )
            candidate_cost = count_tokens(candidate)
            if candidate_cost > used[tier]:
                trace.refilled[tier] = candidate_cost - used[tier]
                leftover -= candidate_cost - used[tier]
                blocks[tier] = candidate
                used[tier] = candidate_cost

        trace.used = dict(used)

        # Phase 4 — enforce: measure the real prompt and trim if needed.
        blocks, total_tokens, trimmed = self._enforce_budget(
            system_block, blocks, total_budget
        )
        trace.trimmed = trimmed
        trace.total_tokens = total_tokens
        self.last_trace = trace

        return CompiledContext(
            system_instructions=system_block,
            entity_block=blocks["entity"],
            episodic_block=blocks["episodic"],
            working_block=blocks["working"],
            total_tokens=total_tokens,
        )

    def compile_prompt(
        self,
        query: str,
        working: WorkingBuffer | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        now: float | None = None,
    ) -> str:
        """Convenience wrapper returning just the prompt string."""
        return self.compile(query, working, episodic, semantic, now).to_prompt()

    # -------------------------------------------------------------- internals

    def _render_system(self, query: str) -> str:
        parts = []
        if self.system_instructions.strip():
            parts.append(f"{SECTION_HEADERS['system']}\n{self.system_instructions.strip()}")
        if query.strip():
            parts.append(f"{SECTION_HEADERS['query']}\n{query.strip()}")
        return _BLOCK_SEPARATOR.join(parts)

    def _render_tier(
        self,
        tier: str,
        slice_tokens: int,
        query: str,
        working: WorkingBuffer | None,
        episodic: EpisodicMemory | None,
        semantic: SemanticMemory | None,
        now: float | None,
    ) -> str:
        """Render one tier's block, header included, inside ``slice_tokens``."""
        header = SECTION_HEADERS[tier]
        body_limit = slice_tokens - count_tokens(header) - count_tokens("\n")
        if body_limit <= 0:
            return ""

        if tier == "entity":
            body = semantic.render(body_limit) if semantic is not None else ""
        elif tier == "episodic":
            body = (
                episodic.render(query, body_limit, top_k=self.top_k_episodes, now=now)
                if episodic is not None
                else ""
            )
        else:
            body = working.render(token_limit=body_limit) if working is not None else ""

        if not body.strip():
            return ""
        return f"{header}\n{body}"

    @staticmethod
    def _reclaim(tier: str, block: str, target_tokens: int) -> str:
        """Shrink one block to ``target_tokens`` using its own structure.

        Each tier has a different notion of what a partial block means, so
        each gets a different reclaiming strategy rather than a shared
        character truncation.
        """
        if target_tokens <= 0:
            return ""
        if tier == "entity":
            # Atomic facts: whole lines only, least-confident dropped first
            # (render() already ordered them by confidence).
            return trim_lines_to_tokens(block, target_tokens, drop_from="end")
        if tier == "working":
            # Chronological: the newest turn is the one that must survive.
            return trim_lines_to_tokens(block, target_tokens, drop_from="start")
        # Episodic: compress by information density before cutting.
        return summarize_to_tokens(block, target_tokens, keep_first_line=True)

    @staticmethod
    def _enforce_budget(
        system_block: str,
        blocks: dict[str, str],
        total_budget: int,
    ) -> tuple[dict[str, str], int, dict[str, int]]:
        """Measure the assembled prompt and reclaim tokens until it fits.

        This is the hard guarantee.  Everything before it is an estimate
        built from per-block measurements; only this loop compares the
        budget against the *actual* string that will be sent.
        """
        trimmed: dict[str, int] = {}
        blocks = dict(blocks)

        def assemble() -> str:
            ordered = [system_block, blocks["entity"], blocks["episodic"], blocks["working"]]
            return _BLOCK_SEPARATOR.join(part for part in ordered if part.strip())

        total = count_tokens(assemble())
        for tier in _TRIM_ORDER:
            if total <= total_budget:
                break
            block = blocks[tier]
            if not block:
                continue
            excess = total - total_budget
            before = count_tokens(block)
            reclaimed = ContextCompiler._reclaim(tier, block, max(0, before - excess))
            if count_tokens(reclaimed) > max(0, before - excess):
                # A structure-aware pass can overshoot (a header that will
                # not fit, an indivisible line); fall back to truncation.
                reclaimed = truncate_to_tokens(reclaimed, max(0, before - excess))
            blocks[tier] = reclaimed if reclaimed.strip() else ""
            after = count_tokens(blocks[tier])
            trimmed[tier] = before - after
            total = count_tokens(assemble())

        if total > total_budget:
            # Last resort: the mandatory block plus separators still
            # overflow. Drop every optional block so the contract holds.
            logger.debug(
                "budget %d still exceeded after reclaiming; dropping all tier blocks",
                total_budget,
            )
            for tier in _TRIM_ORDER:
                blocks[tier] = ""
            total = count_tokens(assemble())
        return blocks, total, trimmed
