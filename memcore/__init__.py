"""mem-core — hierarchical ephemeral and long-term memory for LLM agents.

Three tiers, one budget:

* :mod:`memcore.working`  — Tier 1, verbatim sliding buffer.
* :mod:`memcore.episodic` — Tier 2, vector recall with exponential decay.
* :mod:`memcore.semantic` — Tier 3, durable SQLite entity facts.
* :mod:`memcore.compiler` — the budget compiler that packs all three.

Supporting modules: :mod:`memcore.embeddings` (pluggable vectorisers) and
:mod:`memcore.summarize` (deterministic extractive compression).
"""

from __future__ import annotations

from memcore.compiler import CompilationTrace, ContextCompiler
from memcore.embeddings import (
    CachingEmbedder,
    Embedder,
    GeminiEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from memcore.episodic import EpisodicMemory, ScoredChunk, half_life_to_lambda
from memcore.models import (
    CompiledContext,
    ContextBudget,
    EntityFact,
    EpisodicChunk,
    Turn,
    count_tokens,
)
from memcore.semantic import Extractor, RegexExtractor, SemanticMemory, extract_facts
from memcore.summarize import summarize_to_tokens, trim_lines_to_tokens
from memcore.working import WorkingBuffer

__version__ = "0.1.0"

__all__ = [
    "CachingEmbedder",
    "CompilationTrace",
    "CompiledContext",
    "ContextBudget",
    "ContextCompiler",
    "Embedder",
    "EntityFact",
    "EpisodicChunk",
    "EpisodicMemory",
    "Extractor",
    "GeminiEmbedder",
    "HashingEmbedder",
    "RegexExtractor",
    "ScoredChunk",
    "SemanticMemory",
    "SentenceTransformerEmbedder",
    "Turn",
    "WorkingBuffer",
    "__version__",
    "build_embedder",
    "count_tokens",
    "extract_facts",
    "half_life_to_lambda",
    "summarize_to_tokens",
    "trim_lines_to_tokens",
]
