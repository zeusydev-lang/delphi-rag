# delphi-rag

Hybrid-search RAG over Plutarch's Lives (BM25 + pgvector, Reciprocal Rank
Fusion) with a semantic answer cache. Setup and operations: docs/OPERATIONS.md.

## Commands

- `uv run pytest -q -rs` (database tests need `.env.test`)
- `uv run ruff check .` and `uv run ruff format .`

## Conventions

- Only `agent/llm.py` imports litellm. Chat goes through `LlmClient`,
  embeddings through `EmbeddingClient`; both report `LlmUsage`. Jobs add usage
  up and call `check_budget` before each further call.
- App code connects with `core.db.connect("ingest" | "query")`, never the owner
  `DATABASE_URL` (migrations only). Grants live in the migration that creates
  the table.
- Every prompt module has a `PROMPT_VERSION`; bump it when the wording changes
  and save it with each eval result and cached answer.
- Untrusted text (corpus chunks, user questions, cached questions) goes to the
  model inside tags (`<passage>`, `<question>`), and the instructions say to
  ignore any instructions inside them.
- Eval results are saved per version in `evals/results/<version>.json`.
- Every cached answer and chunk carries the corpus version; a new corpus
  version never serves an old cached answer.
- Tests never call real APIs: use the fake servers in `tests/fake_anthropic.py`.
