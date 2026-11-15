# Delphi RAG

[![CI](https://github.com/zeusydev-lang/delphi-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/zeusydev-lang/delphi-rag/actions/workflows/ci.yml)

**Ask questions about Plutarch's *Lives* and get answers that cite the
passages they came from. Retrieval mixes keyword (BM25) and vector search, and
a semantic cache answers repeat questions without calling the model again.**

> Status: early. The project plumbing is in place; search, answers and the
> cache are being built one phase at a time.

## The problem

TODO: why near-miss names (two Gracchi, two Catos) break naive vector search,
and why a semantic cache can give the right answer to the wrong question.

## How it works

TODO: diagram of question → cache gate (hit / gray zone / miss) → hybrid search
(BM25 + vector, Reciprocal Rank Fusion) → answer with citations.

## Results

**Retrieval** (33 hand-labelled questions on seven Lives, 2026-11-15):

| | hit@5 | hit@1 | near-miss: right Life first |
|---|---|---|---|
| BM25 alone | 0.73 | 0.61 | 0.92 |
| **Vector alone** | **0.97** | **0.76** | 0.92 |
| Hybrid (RRF) | 0.91 | 0.70 | 0.92 |

The expected result was that hybrid search would beat both. On this corpus it
doesn't: BM25 matches words like "Utica" in the wrong passages, and plain fusion
lets them push correct passages down. Halving BM25's weight didn't fix it.

**Answers** (Claude Haiku 5.5 over hybrid search, every citation checked
against the passage it quotes): 94% of answerable questions answered, 91% cite
an expected passage, the main citation is from the right Life 97% of the time
(100% on near-miss questions), and all 5 out-of-corpus questions were
declined. $0.022 for 38 questions. Details in the
[operations guide](docs/OPERATIONS.md#answers-agentanswerpy).

**Semantic cache** (48 labelled question pairs, 10 of them held out):
**0 false hits**, meaning no question was served another question's answer,
and 86–100% of true paraphrases were served from the cache. A cheap
names/ordinals/numbers check stops near-misses like Cato the Elder vs the
Younger (0.969 similar) before any model is asked; a small judge decides the
gray zone. Its first prompt let one false hit through ("asked" vs "promised"
to pay the pirates); the fix was checked on pairs written before it.

## Key decisions

TODO

## What went wrong, and what changed

TODO

## Tech stack

Python 3.13 · Pydantic · litellm · Claude Haiku 5.5 (answers) · Voyage
embeddings · Neon Postgres with pgvector and lakebase_text (BM25) · psycopg ·
pytest · Ruff · GitHub Actions

## Repository layout

```text
agent/        LLM layer (chat + embeddings), structured output, tools
core/         settings, database connections (one login per job), corpus parsing and chunking, search
scripts/      fetch the source text, load the corpus, the Neon spike
migrations/   SQL schema, database logins and permissions
tests/        unit tests with fake model servers; database tests
docs/         operations guide
```

## Running it

Setup, database branches and logins are in the
**[operations guide](docs/OPERATIONS.md)**.

## What I'd do next

TODO

## Credits

Built by Vijay Kauvangal. Built with Claude Code as a pair-programming partner.

The text is Plutarch's *Lives* in the translation associated with John Dryden,
revised by Arthur Hugh Clough: [Project Gutenberg eBook #674](https://www.gutenberg.org/ebooks/674),
public domain in the USA. The first corpus covers seven Lives: both Catos, both
Gracchi, Caesar, Pompey and Crassus.
