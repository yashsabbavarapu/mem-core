# mem-core

[![CI](https://github.com/yashsabbavarapu/mem-core/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yashsabbavarapu/mem-core/actions/workflows/ci.yml?query=branch%3Amain)

Hierarchical ephemeral & long-term agent memory with a hard context budget.

Naive agents fail at memory in one of two ways. Either they replay the whole
transcript into every prompt (blowing the token budget, adding latency, and
triggering *lost-in-the-middle* degradation) or they use a sliding window and
silently drop the constraint the user stated ten turns ago ("we can't use
Kubernetes", "the budget is $5,000").

`mem-core` splits memory into three tiers with different retention physics and
puts a **deterministic budget compiler** in front of them that guarantees the
assembled prompt never exceeds a fixed token limit.

```
                       incoming query
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
 ┌──────▼──────┐     ┌───────▼───────┐    ┌───────▼───────┐
 │  Tier 1     │     │  Tier 2       │    │  Tier 3       │
 │  Working    │     │  Episodic     │    │  Semantic     │
 │  verbatim   │ ──▶ │  vectors +    │    │  SQLite facts │
 │  FIFO turns │drain│  time decay   │    │  no decay     │
 └──────┬──────┘     └───────┬───────┘    └───────┬───────┘
        └────────────────────┼────────────────────┘
                   ┌─────────▼─────────┐
                   │ Context Budget    │   total_tokens ≤ budget
                   │ Compiler          │   (measured, not estimated)
                   └─────────┬─────────┘
                             ▼
                      compiled prompt
```

---

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras, neither required:

| Extra | What it buys |
|---|---|
| `".[tokenizer]"` | Exact `tiktoken` BPE counting instead of the 4-chars-per-token heuristic |
| `".[embeddings]"` | `sentence-transformers` for real semantic recall, computed locally |

Core dependencies are Pydantic v2, NumPy, and the standard library's SQLite.
No paid services are required for any code path.

## Quickstart

```python
from memcore import ContextBudget, ContextCompiler, EpisodicMemory, SemanticMemory, WorkingBuffer

working  = WorkingBuffer(max_turns=6)
episodic = EpisodicMemory(path="episodes.sqlite")   # durable, 24h half-life
semantic = SemanticMemory("memory.sqlite")          # durable facts

turn, evicted = working.add_turn("user", "We use FastAPI and DuckDB, my budget is $5000")
episodic.ingest_turns(evicted)              # tier 1 -> tier 2 hand-off
semantic.ingest_text(turn.content)          # tier 3 extraction

compiler = ContextCompiler(ContextBudget(total_token_budget=1500))
context  = compiler.compile("what stack are we on?", working, episodic, semantic)

print(context.to_prompt())
assert context.total_tokens <= 1500         # guaranteed, not hoped for
```

## CLI

```bash
python -m memcore.cli chat --window 6 --budget 1500 --seed
python -m memcore.cli pack --turns 10 --budget 500
python -m memcore.cli chat --embedder sentence-transformers --verbose
```

`chat` prints tier transitions as they happen (`[tier 1→2] 2 turn(s) drained`,
`[tier 3] learned user.budget = 5000`) and supports `/facts`, `/episodes`,
`/context <query>`, `/stats`, `/trace`, `/quit`. Both durable tiers survive
restarts; `--ephemeral` opts out.

`pack` compiles a synthetic conversation under a hard budget and exits
non-zero if the budget were ever breached:

```
budget            : 500
reserved (system) : 53
available to tiers: 444
  entity   : allocated=89    refill=+0    used=60    trimmed=0
  episodic : allocated=177   refill=+0    used=94    trimmed=0
  working  : allocated=178   refill=+0    used=64    trimmed=0
final total       : 272 / 500
```

---

## The three tiers

### Tier 1: Working memory (`working.py`)

A FIFO window of verbatim `Turn` records, bounded by turn count *and*
optionally by tokens. Nothing is ever destroyed: every eviction path returns
the evicted turns so the caller can hand them to Tier 2.

```python
turn, evicted = working.add_turn("user", "...")
episodic.ingest_turns(evicted)   # the tier 1 -> tier 2 lifecycle in one line
```

Rendering is **newest-biased**: turns are packed newest-first then re-ordered
chronologically, so a tight budget drops the *oldest* turn rather than
truncating the most recent one.

### Tier 2: Episodic memory (`episodic.py`)

Past exchanges, embedded and retrieved by similarity **penalised by age**.
Drained turns are *summarised* on ingest (filler dropped, capped at
`summary_token_limit`) so Tier 2 stays dense instead of accumulating chat.

* **Durable**. Pass `path=` and episodes, vectors included, persist in SQLite
  across restarts (float32 on disk, float64 in RAM).
* **Vectorised**. Scoring is one mat-vec product over a contiguous `(N, d)`
  matrix, and only the top `k` results are materialised as objects.
* **Bounded**. `max_chunks` compacts the store instead of growing forever.
* **Thread-safe**. A re-entrant lock guards every read and write.

Measured on this machine (5,000 episodes, 256 dims):

| | Per query |
|---|---|
| Naive per-chunk Python loop | 72 ms |
| Vectorised + top-k materialisation | **3.9 ms** |
| Same, at 50,000 episodes | 38 ms |

Beyond ~100k episodes an ANN index would be the next step; the linear scan is
deliberate at this scale, where it is simpler and exact.

### Choosing an embedder

This is the single biggest lever on recall quality, so pick deliberately:

| Backend | Cost | Semantic? | Use when |
|---|---|---|---|
| `HashingEmbedder` *(default)* | free, offline | **no**, lexical only | tests, demos, determinism |
| `SentenceTransformerEmbedder` | free, offline, `[embeddings]` extra | yes | **production default** |
| `GeminiEmbedder` | API | yes | you already pay for the API |

The default matches *words*, not *meaning*. Hashed character n-grams (3–5)
with `log1p` term weighting bridge morphology (`database`/`databases`,
`deploy`/`deployment`) which lifts recall@1 from 0.40 to 0.60 on the labelled
set in `tests/test_embeddings.py`, but no hashing scheme produces true
synonymy. If paraphrased queries must retrieve, use a learned backend:

```python
from memcore import EpisodicMemory, build_embedder
episodic = EpisodicMemory(embedder=build_embedder("sentence-transformers"))
```

`build_embedder` wraps the backend in an LRU cache and falls back to the local
embedder with a warning if a learned one is unavailable. A missing API key
degrades recall *quality*, never availability.

The Gemini path has bounded retries with exponential backoff, true batching
via `batchEmbedContents`, and an injectable transport. Its request and
response handling is exercised against a faithful stub in the test suite;
the live API has not been called, verify before relying on it.

### Tier 3: Semantic entity memory (`semantic.py`)

Durable `entity.attribute = value` rows in SQLite, primary-keyed on
`(entity, attribute)`. Extraction is **deterministic** (regex rules, no model
call) so the same transcript always yields the same facts:

| Utterance | Fact |
|---|---|
| `My name is Alex` / `My name's Sam` | `user.name = Alex` (conf 0.95) |
| `We use FastAPI and DuckDB` | `user.tech_stack = ["FastAPI", "DuckDB"]` (0.85) |
| `We ended up going with Postgres` | `user.tech_stack = ["Postgres"]` (0.85) |
| `We stopped using Redis` | *retraction*, Redis is removed from the stack |
| `My budget is $10k` | `user.budget = 10000` (0.90) |
| `We can't use Kubernetes` | `user.constraint = must not use Kubernetes` (0.90) |
| `Our team is based in Lisbon` | `user.location = Lisbon` (0.75) |
| `I'm in CET` | `user.timezone = CET` (0.80) |

**Conflict resolution.** The incoming value wins when
`new.confidence >= existing.confidence`. Equality included, so a later
restatement of an equally-trusted fact overwrites the earlier one and recency
breaks the tie. A lower-confidence assertion never clobbers a higher-confidence
one. List-valued attributes (`tech_stack`) merge as a case-insensitive union,
which is how "we also use Redis" extends a stack; retractions subtract from it.

**Pluggable.** Extraction is a protocol, so a model-based extractor can be
layered over the rules and arbitrated by the same confidence policy:

```python
SemanticMemory("memory.sqlite", extractors=[RegexExtractor(), MyLLMExtractor()])
```

**Operational.** WAL journaling, a busy timeout, `check_same_thread=False` and
a re-entrant lock make one store safe to share across a threaded or async
server. The schema carries a `user_version` and migrates forward on open; a
database written by a *newer* mem-core is refused rather than misread.

Credentials are never persisted. Values matching credential shapes (API
keys, tokens, passwords, payment-card numbers, long opaque blobs) are refused
at both `ingest_text` and `upsert`. Memory is written to disk and replayed into
prompts, so a false positive costs nothing and a false negative leaks a secret.

---

## Deriving the decay curve

Episodic relevance must combine *what the memory is about* with *how long ago
it happened*:

$$\text{score} = \underbrace{\cos(\vec{q}, \vec{c})}_{\text{semantic match}} \times \underbrace{e^{-\lambda \Delta t}}_{\text{recency weight}}$$

where `Δt` is the age of the chunk **in hours**.

**Why exponential?** Assume relevance decays at a rate proportional to its
current value, each additional hour costs the same *fraction* of what is
left, not the same absolute amount:

```
dR/dt = -λR      ⟹      R(t) = R₀ · e^(−λt)
```

That is the only form with a constant *proportional* decay rate, which makes
it self-similar: "one day older" is the same multiplier whether the memory is
one day or one week old. Linear decay would hit zero and truncate history; a
power law would keep ancient memories competitive far too long.

Choosing λ from a half-life. Solve for the age at which a memory is worth
half of its fresh self:

```
e^(−λ·t½) = 0.5   ⟹   λ = ln(2) / t½
```

| Half-life | λ (per hour) | Weight after 24h | After 7 days |
|---|---|---|---|
| 1 hour  | 0.6931  | 6.0e-8 | ~0 |
| 6 hours | 0.1155  | 0.0625 | 3.7e-9 |
| **24 hours (default)** | **0.0289** | **0.500** | 0.0078 |
| 7 days  | 0.0041  | 0.906  | 0.500 |
| ∞ (λ=0) | 0       | 1.000  | 1.000 (pure similarity) |

```python
from memcore import EpisodicMemory, half_life_to_lambda
memory = EpisodicMemory(lambda_decay=half_life_to_lambda(24.0))
```

The property that matters, asserted directly in `tests/test_episodic.py`:
two chunks with *identical* content (hence identical cosine similarity) always
rank newer-first, and the entire gap is attributable to the decay term. With
`λ = 0` the formula collapses to pure similarity. Negative ages (clock skew)
are clamped to a weight of 1.0 rather than allowed to amplify a score.

---

## The budget packing algorithm

The compiler's contract is one line:

> `count_tokens(compiled.to_prompt()) <= budget.total_token_budget`, always.

It holds at *any* budget (1,500, 40, 6, or 0) and for any ratio split. Four
phases:

**1. Reserve.** System instructions and the current query are mandatory and
charged first. If they alone exceed the budget they are truncated and every
other block is empty; the invariant survives even a 1-token budget.

**2. Allocate.** The remainder (minus a separator reserve) is split by the
`ContextBudget` ratios using **largest-remainder rounding**, so slices sum to
*exactly* the available tokens rather than losing a few to floor division.

**3. Refill.** A tier that cannot spend its slice (an empty entity store, a
short buffer) donates the remainder to the others in priority order
`entity → working → episodic`. Unused capacity becomes extra recall instead of
dead space.

**4. Enforce.** Everything so far is arithmetic on per-block estimates, and
real tokenizers are not additive: `count(a) + count(b) ≠ count(a + b)`. So the
compiler assembles the actual prompt, **measures it**, and reclaims the
overflow from the lowest-priority block first, re-measuring after each pass.

Reclaiming is **structure-aware**, which matters more than it sounds:

| Block | Strategy | Why |
|---|---|---|
| Entity facts | drop whole lines, least-confident first | a truncated `user.budget = 5000` reads as `user.budget = 500`, a fabricated constraint is worse than a missing one |
| Recent turns | drop whole lines from the oldest end | the live exchange must stay intact |
| Episodic | summarise by information density, then cut | "Sounds good, thanks!" goes before "the budget is $5,000" |

Hard character truncation remains only as a last resort, and the final
measured check runs regardless, so the contract holds unconditionally.

The priority order encodes the design thesis: durable constraints outlive
conversational immediacy, which outlives recalled history. When the budget
gets tight, the agent would rather forget last Tuesday than forget that the
budget is $5,000.

### Token accounting

One counter, `memcore.models.count_tokens`, is used by every tier and by the
compiler. If they disagreed about what a token is, the guarantee would be
meaningless. It uses `tiktoken` (`cl100k_base`) when installed and otherwise
falls back to `ceil(len(text) / 4)`. The fallback is a *ceiling* on purpose:
over-estimating can only pack less context, never overflow.

---

## Testing

```bash
pytest -v                  # 174 tests
mypy --strict memcore      # strict typing, package and tests
python -m memcore.cli pack --turns 10 --budget 500
```

| Suite | What it pins down |
|---|---|
| `test_working.py` | FIFO eviction order, token caps, drain hand-off, newest-biased rendering |
| `test_episodic.py` | Decay ranking matches `e^(−λΔt)`; durability across restarts; compaction; top-k fast path agrees with exhaustive ranking; concurrent reads and writes; no mutation of stored chunks |
| `test_semantic.py` | 20 extraction rules, retractions, upsert/persistence, confidence-gated conflicts, 8-thread concurrency, schema migration, credential refusal, atomic rendering |
| `test_embeddings.py` | Determinism and normalisation, LRU behaviour, retrieval-quality regression guard, Gemini request shape / batching / retry / backoff / error handling via a stub transport |
| `test_summarize.py` | Density ranking, order preservation, token ceilings, and the atomicity property that facts are never emitted partially |
| `test_compiler.py` | The budget is never breached, 12 budget sizes, 4 ratio splits, empty memories, durable stores, adversarially long content |

Beyond the suite, the two core invariants were fuzzed over **300 randomized
trials** (unicode and emoji content, random ratio splits, 14 budget sizes from
0 to 4096, 25% against durable SQLite stores) under both the `tiktoken` and
heuristic counters: zero budget breaches, zero partial facts.

## Layout

```
memcore/
├── models.py      # Turn, EntityFact, EpisodicChunk, ContextBudget, CompiledContext, count_tokens
├── working.py     # Tier 1: sliding verbatim buffer
├── episodic.py    # Tier 2: durable vector recall with exponential decay
├── semantic.py    # Tier 3: SQLite entity facts + deterministic extraction
├── compiler.py    # Context budget compiler (reserve → allocate → refill → enforce)
├── embeddings.py  # Pluggable vectorisers (hashing / sentence-transformers / Gemini / cache)
├── summarize.py   # Deterministic extractive compression
└── cli.py         # chat / pack
```

## Known limits

Stated plainly, because a memory system that oversells its recall is worse
than one that documents its floor:

* The default embedder is **lexical**. Paraphrased queries need
  `[embeddings]`.
* `GeminiEmbedder` is stub-tested, not live-tested.
* Retrieval is an exact linear scan. Excellent to ~50k episodes, wants an ANN
  index beyond ~100k.
* Extraction is English, first-person and declarative. It has high precision
  and modest recall by design; layer an LLM `Extractor` for conversational
  phrasing.
* Summarisation is extractive, not abstractive. It selects sentences, it does
  not rewrite them.

## License

MIT
