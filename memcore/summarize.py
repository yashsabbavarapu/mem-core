"""Deterministic extractive compression for low-density context.

The budget compiler's original fallback was a hard character truncation,
which is lossy in the worst possible way: it keeps the *first* half of a
block regardless of what the half contains, and it can cut a fact line in
two.  This module provides two better primitives, both deterministic (no
model call, identical output for identical input):

* :func:`trim_lines_to_tokens` — drop whole lines, never part of one.  A
  fact is either fully present or absent; ``user.budget = 5000`` can never
  degrade into ``user.budget = 500``.
* :func:`summarize_to_tokens` — rank sentences by information density and
  keep the densest ones in their original order, so "Sounds good, thanks!"
  is discarded before "the budget is $5,000".

Density is a heuristic, deliberately a transparent one: numbers, proper
nouns and rare words carry information; filler phrases and stopwords do
not.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Final

from memcore.models import count_tokens, truncate_to_tokens

__all__ = [
    "compress_turns",
    "density_score",
    "sentence_split",
    "summarize_to_tokens",
    "trim_lines_to_tokens",
]

_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+|\n+|\s+\|\s+")
_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9$€£%.\-']+")
_NUMBERISH: Final[re.Pattern[str]] = re.compile(r"[\d]")

_STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has", "have", "he", "her",
        "his", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
        "them", "then", "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
        "will", "with", "would", "you", "your"
    ]
)

#: Phrases that carry acknowledgement but no information.
_FILLER: Final[tuple[str, ...]] = (
    "sounds good",
    "thanks",
    "thank you",
    "no problem",
    "sure thing",
    "got it",
    "understood",
    "makes sense",
    "let me know",
    "you are welcome",
    "you're welcome",
    "of course",
    "good to meet you",
    "nice to meet you",
    "happy to help",
)


def sentence_split(text: str) -> list[str]:
    """Split ``text`` into sentence-ish units, preserving their order.

    Units are stripped of the turn separators used by episodic chunks, so a
    split that lands beside a ``|`` does not leave it dangling at the head
    of the next unit.
    """
    if not text.strip():
        return []
    units = (part.strip().strip("|").strip() for part in _SENTENCE_SPLIT.split(text))
    return [unit for unit in units if unit]


def density_score(sentence: str, corpus_counts: Counter[str] | None = None) -> float:
    """Score a sentence's information density (higher = more worth keeping).

    Args:
        sentence: The unit being scored.
        corpus_counts: Word frequencies across the whole text, used to
            reward rare terms.  Optional; scoring degrades gracefully
            without it.
    """
    words = _WORD.findall(sentence.lower())
    if not words:
        return 0.0

    content = [word for word in words if word not in _STOPWORDS]
    score = len(content) / len(words)  # content-word ratio

    # Numbers and currency are almost always the payload of a constraint.
    score += 1.5 * sum(1 for word in words if _NUMBERISH.search(word)) / len(words)

    # Proper nouns (capitalised, not sentence-initial) name entities.
    raw_words = _WORD.findall(sentence)
    score += 1.0 * sum(1 for word in raw_words[1:] if word[:1].isupper()) / len(raw_words)

    # Rare terms carry more signal than terms repeated throughout the text.
    if corpus_counts:
        rarity = sum(1.0 / corpus_counts[word] for word in content if corpus_counts[word])
        score += 0.5 * rarity / max(1, len(content))

    lowered = sentence.lower()
    for phrase in _FILLER:
        if phrase in lowered:
            score -= 1.0

    # Very short fragments rarely survive on their own merit.
    if len(content) <= 1:
        score -= 0.5
    return score


def summarize_to_tokens(
    text: str,
    max_tokens: int,
    keep_first_line: bool = False,
) -> str:
    """Compress ``text`` to at most ``max_tokens`` by dropping low-density units.

    Sentences are ranked by :func:`density_score`, the densest are kept
    greedily until the budget is spent, and the survivors are re-emitted in
    their **original order** so the result still reads as prose.

    Args:
        keep_first_line: Treat line one as a mandatory header (a section
            title) that is always retained if it fits at all.

    The return value is guaranteed to cost at most ``max_tokens`` tokens.
    """
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    hard_limit = max_tokens
    header = ""
    body = text
    if keep_first_line and "\n" in text:
        header, _, body = text.partition("\n")
        header_cost = count_tokens(header) + 1
        if header_cost >= max_tokens:
            return truncate_to_tokens(text, hard_limit)
        max_tokens -= header_cost

    units = sentence_split(body)
    if not units:
        return truncate_to_tokens(text, hard_limit)

    corpus_counts: Counter[str] = Counter(
        word for unit in units for word in _WORD.findall(unit.lower()) if word not in _STOPWORDS
    )
    scores = [density_score(unit, corpus_counts) for unit in units]

    # Filler (negative density) is never worth a slot while any informative
    # unit exists: at a tight budget, a truncated fact beats an intact
    # "Sounds good, thanks!".
    informative = [index for index in range(len(units)) if scores[index] > 0.0]
    candidates = informative or list(range(len(units)))
    ranked = sorted(candidates, key=lambda index: (-scores[index], index))

    chosen: set[int] = set()
    used = 0
    for index in ranked:
        cost = count_tokens(units[index]) + (1 if chosen else 0)
        if used + cost <= max_tokens:
            chosen.add(index)
            used += cost

    # With nothing fitting whole, keep a truncated head of the densest unit.
    kept = (
        " ".join(units[index] for index in sorted(chosen))
        if chosen
        else truncate_to_tokens(units[ranked[0]], max_tokens)
    )

    result = f"{header}\n{kept}" if header else kept
    # The ceiling is a contract, not an estimate: verify before returning.
    return truncate_to_tokens(result, hard_limit)


def trim_lines_to_tokens(
    text: str,
    max_tokens: int,
    keep_first_line: bool = True,
    drop_from: str = "end",
) -> str:
    """Shrink ``text`` by removing whole lines, never partial ones.

    This is what protects structured blocks: an entity fact is atomic, so
    dropping ``user.budget = 5000`` entirely is correct where truncating it
    to ``user.budget = 500`` would be a fabrication.

    Args:
        keep_first_line: Retain line one (the section header) if it fits.
        drop_from: ``"end"`` drops the last lines first (blocks ordered
            most-important-first, like entity facts); ``"start"`` drops the
            earliest lines first (chronological blocks, like recent turns,
            where the newest line matters most).
    """
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    lines = text.split("\n")
    header: str | None = None
    if keep_first_line and lines:
        header = lines[0]
        lines = lines[1:]
        if count_tokens(header) > max_tokens:
            return truncate_to_tokens(header, max_tokens)

    def assemble(body: Sequence[str]) -> str:
        parts = ([header] if header is not None else []) + list(body)
        return "\n".join(part for part in parts if part)

    body_lines = list(lines)
    while body_lines and count_tokens(assemble(body_lines)) > max_tokens:
        body_lines.pop() if drop_from == "end" else body_lines.pop(0)

    result = assemble(body_lines)
    # If even the header alone overflows, fall back to character truncation.
    return result if count_tokens(result) <= max_tokens else truncate_to_tokens(result, max_tokens)


def compress_turns(rendered_turns: Sequence[str], max_tokens: int | None = None) -> str:
    """Build an episodic summary from a group of drained turns.

    Filler-only turns are dropped outright, the rest are joined, and the
    result is optionally compressed to ``max_tokens``.  Returning a
    *summary* rather than a raw transcript is what keeps Tier 2 dense.
    """
    kept = [turn for turn in rendered_turns if density_score(turn) > 0.0]
    if not kept:
        kept = list(rendered_turns)
    joined = " | ".join(kept)
    if max_tokens is None:
        return joined
    return summarize_to_tokens(joined, max_tokens)
