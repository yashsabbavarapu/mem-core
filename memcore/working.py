"""Tier 1 — Working memory: a verbatim sliding buffer of recent turns.

Working memory is the cheapest and most faithful tier: it stores turns
exactly as they happened so the model never loses conversational
immediacy ("it" in the current turn must still resolve).  Its capacity is
bounded on two axes — number of turns and total tokens — and whatever falls
out of the window is *drained* rather than discarded, so Tier 2 can absorb
it.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence

from memcore.models import Role, Turn, count_tokens, truncate_to_tokens

__all__ = ["WorkingBuffer", "count_tokens"]


class WorkingBuffer:
    """A FIFO window over the most recent :class:`~memcore.models.Turn` records.

    Args:
        max_turns: Hard cap on retained turns; the oldest are evicted first.
        max_tokens: Optional soft cap on retained tokens.  When exceeded,
            the oldest turns are evicted until the buffer fits.  ``None``
            disables token-based eviction.

    Eviction never destroys data: every method that removes turns returns
    them, and the caller is expected to hand them to episodic memory.
    """

    def __init__(self, max_turns: int = 10, max_tokens: int | None = None) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._turns: deque[Turn] = deque()

    # ---------------------------------------------------------------- writes

    def add(self, turn: Turn) -> list[Turn]:
        """Append ``turn`` and return the turns evicted to make room."""
        self._turns.append(turn)
        return self._enforce_limits()

    def add_turn(
        self,
        role: Role,
        content: str,
        timestamp: float | None = None,
    ) -> tuple[Turn, list[Turn]]:
        """Build a :class:`Turn` from raw parts, append it, and return both.

        Returns:
            The newly created turn and the list of turns evicted by it.
        """
        turn = Turn(
            role=role,
            content=content,
            timestamp=time.time() if timestamp is None else timestamp,
        )
        return turn, self.add(turn)

    def extend(self, turns: Iterable[Turn]) -> list[Turn]:
        """Append many turns, returning everything evicted along the way."""
        evicted: list[Turn] = []
        for turn in turns:
            evicted.extend(self.add(turn))
        return evicted

    def _enforce_limits(self) -> list[Turn]:
        """Evict from the head until both capacity limits are satisfied."""
        evicted: list[Turn] = []
        while len(self._turns) > self.max_turns:
            evicted.append(self._turns.popleft())
        if self.max_tokens is not None:
            # Always keep at least one turn: a single turn larger than the
            # token cap is the compiler's problem to truncate, not ours to
            # silently delete.
            while len(self._turns) > 1 and self.total_tokens > self.max_tokens:
                evicted.append(self._turns.popleft())
        return evicted

    # ---------------------------------------------------------------- reads

    @property
    def turns(self) -> list[Turn]:
        """All retained turns, oldest first."""
        return list(self._turns)

    @property
    def total_tokens(self) -> int:
        """Token cost of the retained turns (content only)."""
        return sum(turn.token_count for turn in self._turns)

    def recent(self, k: int) -> list[Turn]:
        """Return the ``k`` most recent turns, oldest first."""
        if k <= 0:
            return []
        return list(self._turns)[-k:]

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self) -> Iterator[Turn]:
        return iter(self._turns)

    def __bool__(self) -> bool:
        return bool(self._turns)

    # ---------------------------------------------------------------- drains

    def drain(self, keep: int = 0) -> list[Turn]:
        """Remove and return all but the newest ``keep`` turns.

        This is the Tier 1 → Tier 2 hand-off: the returned turns are the
        ones that should be summarised into episodic memory.
        """
        keep = max(0, keep)
        drained: list[Turn] = []
        while len(self._turns) > keep:
            drained.append(self._turns.popleft())
        return drained

    def clear(self) -> list[Turn]:
        """Remove and return every turn."""
        return self.drain(keep=0)

    # ---------------------------------------------------------------- render

    def render(self, token_limit: int | None = None, k: int | None = None) -> str:
        """Render the buffer as prompt text, newest-biased.

        Turns are added newest-first until ``token_limit`` is reached, then
        re-ordered chronologically, so a tight budget drops *old* turns
        rather than truncating the most recent one.
        """
        candidates: Sequence[Turn] = self.recent(k) if k is not None else self.turns
        if token_limit is None:
            return "\n".join(turn.render() for turn in candidates)
        if token_limit <= 0:
            return ""

        selected: list[Turn] = []
        used = 0
        for turn in reversed(candidates):
            line = turn.render()
            cost = count_tokens(line) + (1 if selected else 0)  # +1 for newline
            if used + cost <= token_limit:
                selected.append(turn)
                used += cost
                continue
            if not selected:
                # Even the newest turn does not fit: keep a truncated head of
                # it rather than emitting an empty working block.
                return truncate_to_tokens(line, token_limit)
            break
        selected.reverse()
        return "\n".join(turn.render() for turn in selected)
