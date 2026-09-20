"""Command line front-end for the mem-core memory engine.

Two subcommands:

* ``chat`` — an interactive loop showing the working buffer sliding, turns
  draining into episodic memory, and entity facts being extracted live.
* ``pack`` — a scripted demonstration that synthesises a conversation and
  compiles it under a hard token budget, printing the allocation trace so
  the trimming decisions are visible.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from memcore.compiler import ContextCompiler
from memcore.embeddings import Embedder, build_embedder
from memcore.episodic import EpisodicMemory, half_life_to_lambda
from memcore.models import ContextBudget, Role, count_tokens
from memcore.semantic import DEFAULT_DB_PATH, SemanticMemory
from memcore.working import WorkingBuffer

__all__ = ["build_parser", "main"]

_RULE = "─" * 72

#: Scripted conversation used by ``pack`` (and to seed ``chat --seed``).
_SCRIPT: tuple[tuple[Role, str], ...] = (
    ("user", "Hi! My name is Alex and I am a backend engineer."),
    ("assistant", "Nice to meet you, Alex. What are you building?"),
    ("user", "We use FastAPI and DuckDB for an internal analytics service."),
    ("assistant", "FastAPI over DuckDB is a solid pairing for read-heavy analytics."),
    ("user", "My budget is $5000 and we can't use Kubernetes."),
    ("assistant", "Understood — single VM deployment then, no orchestration layer."),
    ("user", "The deadline is March 3rd, so we need something simple."),
    ("assistant", "I would ship a single container behind a reverse proxy."),
    ("user", "I prefer SQL over ORMs for the query layer."),
    ("assistant", "Then DuckDB's SQL interface can stay the primary abstraction."),
    ("user", "We also use Redis for caching hot aggregates."),
    ("assistant", "Cache invalidation on write should keep aggregates fresh."),
    ("user", "I live in Berlin, so reviews land in your evening."),
    ("assistant", "Noted, I will batch feedback to fit that window."),
)


def _embedder(name: str) -> Embedder:
    """Resolve the ``--embedder`` flag (falls back to local, with a warning)."""
    return build_embedder(name, cache=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m memcore.cli`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="memcore",
        description="Hierarchical agent memory with a strict context budget.",
    )
    parser.add_argument(
        "--embedder",
        choices=("local", "sentence-transformers", "gemini"),
        default="local",
        help=(
            "embedding backend for episodic memory (default: local). "
            "'local' is lexical only; 'sentence-transformers' gives real "
            "semantic recall offline; 'gemini' needs GEMINI_API_KEY"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log tier activity (evictions, compaction, retractions) to stderr",
    )
    parser.add_argument(
        "--half-life",
        type=float,
        default=24.0,
        help="episodic half-life in hours; lambda = ln(2)/half_life (default: 24)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="interactive memory loop")
    chat.add_argument("--budget", type=int, default=1500, help="token budget per turn")
    chat.add_argument("--window", type=int, default=6, help="working buffer size in turns")
    chat.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite path for entity memory")
    chat.add_argument(
        "--ephemeral",
        action="store_true",
        help="use an in-memory SQLite store (nothing is written to disk)",
    )
    chat.add_argument("--seed", action="store_true", help="preload the scripted conversation")
    chat.add_argument(
        "--episodes-db",
        default=None,
        help="SQLite path for durable episodic memory (default: alongside --db)",
    )
    chat.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="compact episodic memory to at most this many chunks",
    )

    pack = sub.add_parser("pack", help="demonstrate strict budget packing")
    pack.add_argument("--turns", type=int, default=10, help="number of turns to synthesise")
    pack.add_argument("--budget", type=int, default=500, help="hard token budget")
    pack.add_argument("--window", type=int, default=4, help="working buffer size in turns")
    pack.add_argument("--query", default="What stack and budget are we working with?")
    pack.add_argument(
        "--show-prompt",
        action="store_true",
        default=True,
        help="print the compiled prompt (default: on)",
    )
    pack.add_argument(
        "--no-show-prompt",
        dest="show_prompt",
        action="store_false",
        help="print only the allocation trace",
    )
    return parser


def _make_memories(
    args: argparse.Namespace,
    db_path: str,
    episodes_path: str | None = None,
) -> tuple[WorkingBuffer, EpisodicMemory, SemanticMemory]:
    working = WorkingBuffer(max_turns=args.window)
    episodic = EpisodicMemory(
        embedder=_embedder(args.embedder),
        lambda_decay=half_life_to_lambda(args.half_life),
        path=episodes_path,
        max_chunks=getattr(args, "max_episodes", None),
    )
    semantic = SemanticMemory(db_path)
    return working, episodic, semantic


def _ingest(
    role: Role,
    content: str,
    timestamp: float,
    working: WorkingBuffer,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
) -> tuple[int, list[str]]:
    """Run one turn through all three tiers; report what each tier did."""
    _, evicted = working.add_turn(role, content, timestamp=timestamp)
    episodic.ingest_turns(evicted)
    learned: list[str] = []
    if role == "user":
        learned = [fact.render() for fact in semantic.ingest_text(content, timestamp=timestamp)]
    return len(evicted), learned


def _run_chat(args: argparse.Namespace) -> int:
    db_path = ":memory:" if args.ephemeral else args.db
    if args.ephemeral:
        episodes_path = None
    elif args.episodes_db:
        episodes_path = args.episodes_db
    else:
        episodes_path = str(Path(args.db).with_suffix(".episodes.sqlite"))
    working, episodic, semantic = _make_memories(args, db_path, episodes_path)
    compiler = ContextCompiler(ContextBudget(total_token_budget=args.budget))

    print(_RULE)
    print(
        f"mem-core chat | window={args.window} turns | budget={args.budget} tokens | "
        f"entities={db_path} | episodes={episodes_path or 'in-memory'} | "
        f"embedder={args.embedder}"
    )
    if args.embedder == "local":
        print(
            "[note] the local embedder matches words, not meaning; use "
            "--embedder sentence-transformers for semantic recall"
        )
    print("Commands: /facts  /episodes  /context <query>  /stats  /trace  /quit")
    print(_RULE)

    if args.seed:
        base = time.time() - len(_SCRIPT) * 900
        for index, (role, content) in enumerate(_SCRIPT):
            _ingest(role, content, base + index * 900, working, episodic, semantic)
        print(
            f"[seeded] {len(_SCRIPT)} turns | working={len(working)} "
            f"episodes={len(episodic)} facts={len(semantic)}"
        )

    try:
        while True:
            try:
                line = input("\nyou> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in {"/quit", "/exit"}:
                break
            if line == "/facts":
                facts = semantic.all_facts()
                print("\n".join(f"  {fact.render()}  (conf={fact.confidence})" for fact in facts)
                      or "  (no facts yet)")
                continue
            if line == "/episodes":
                print("\n".join(f"  [{chunk.id[:8]}] {chunk.content[:88]}" for chunk in episodic.chunks)
                      or "  (no episodes yet)")
                continue
            if line == "/stats":
                print("  working :", {"turns": len(working), "tokens": working.total_tokens})
                print("  episodic:", episodic.stats())
                print("  semantic:", semantic.stats())
                continue
            if line == "/trace":
                print(compiler.last_trace.render())
                continue
            if line.startswith("/context"):
                query = line[len("/context"):].strip() or "summarise what you know about me"
                context = compiler.compile(query, working, episodic, semantic)
                print(_RULE)
                print(context.to_prompt())
                print(_RULE)
                print(f"total_tokens={context.total_tokens} / budget={args.budget}")
                continue

            evicted, learned = _ingest(
                "user", line, time.time(), working, episodic, semantic
            )
            context = compiler.compile(line, working, episodic, semantic)
            reply = (
                f"(demo) packed {context.total_tokens}/{args.budget} tokens from "
                f"{len(working)} live turns, {len(episodic)} episodes, {len(semantic)} facts"
            )
            _ingest("assistant", reply, time.time(), working, episodic, semantic)

            if evicted:
                print(f"  [tier 1→2] {evicted} turn(s) drained into episodic memory")
            for fact in learned:
                print(f"  [tier 3]   learned {fact}")
            print(f"  {reply}")
    finally:
        semantic.close()
        episodic.close()
    return 0


def _run_pack(args: argparse.Namespace) -> int:
    working, episodic, semantic = _make_memories(args, ":memory:")
    compiler = ContextCompiler(ContextBudget(total_token_budget=args.budget))

    # Synthesise `--turns` turns, oldest first, 15 minutes apart, so the
    # decay curve has something to bite on.
    now = time.time()
    total = max(1, args.turns)
    for index in range(total):
        role, content = _SCRIPT[index % len(_SCRIPT)]
        timestamp = now - (total - index) * 900
        _ingest(role, content, timestamp, working, episodic, semantic)

    context = compiler.compile(args.query, working, episodic, semantic, now=now)
    prompt = context.to_prompt()
    measured = count_tokens(prompt)

    print(_RULE)
    print(f"query   : {args.query}")
    print(
        f"state   : {total} turns ingested | working={len(working)} "
        f"episodes={len(episodic)} facts={len(semantic)}"
    )
    print(_RULE)
    print(compiler.last_trace.render())
    print(_RULE)
    if args.show_prompt:
        print(prompt)
        print(_RULE)
    status = "OK" if measured <= args.budget else "BREACH"
    print(
        f"measured tokens = {measured} | reported = {context.total_tokens} | "
        f"budget = {args.budget} | {status}"
    )
    semantic.close()
    episodic.close()
    return 0 if measured <= args.budget else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m memcore.cli``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.command == "chat":
        return _run_chat(args)
    return _run_pack(args)


if __name__ == "__main__":
    raise SystemExit(main())
