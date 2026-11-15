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

TODO: retrieval and cache eval results, by version.

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
