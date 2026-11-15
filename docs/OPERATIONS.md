# Delphi RAG: operations guide

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Install dependencies:
   ```
   uv sync
   ```
3. Copy `.env.example` to `.env` and set `DATABASE_URL` to the `dev` branch
   owner connection string (see below), plus `ANTHROPIC_API_KEY` and
   `VOYAGE_API_KEY`.
4. Create the schema and the limited logins on `dev`:
   ```
   uv run python migrations/migrate.py
   uv run python migrations/logins.py --env-file .env
   ```

## Database (Neon Postgres)

The Neon project (Postgres 18) has two branches to start with, each with its
own env file. CI's Postgres uses the same major version. None of these files is committed to git.

| Neon branch | Use                                   | Env file    |
|-------------|---------------------------------------|-------------|
| `dev`       | Building and trying things            | `.env`      |
| `test`      | Automated tests (wiped by the tests)  | `.env.test` |

Each env file starts with one line, the branch's **owner** connection string.
In Neon: **Connect** → choose the branch, database `neondb`, role
`neondb_owner`, pooling **off** (host without `-pooler`):

```
DATABASE_URL="postgresql://neondb_owner:...@ep-.../neondb?sslmode=require"
```

`logins.py` then adds the limited logins to the same file.

### Database logins

Each job uses its own login with only the rights it needs.

| Login | Used by | Can |
|---|---|---|
| `neondb_owner` | Migration runner only | Everything. Never used by app code or GitHub |
| `delphi_ingest` | Loading the corpus | Write chunks (grants added with the chunks table) |
| `delphi_query` | Answering questions | Read chunks; read and write the cache (grants added with those tables) |

The roles are created by `001_roles.sql` without passwords. `logins.py` turns
on login with random passwords and writes `INGEST_DATABASE_URL` and
`QUERY_DATABASE_URL` into the env file you name. Passwords are never printed.
Running it again rotates both. Neither login can create tables.

### Running migrations

The runner applies every `.sql` file in `migrations/` in name order
(`001_`, `002_`, ...). Each file runs in its own transaction, so a failed
file makes no changes. It prints the target host before applying anything.

| Branch | Commands |
|---|---|
| `dev` | `uv run python migrations/migrate.py`<br>`uv run python migrations/logins.py --env-file .env` |
| `test` | `uv run --env-file .env.test python migrations/migrate.py`<br>`uv run python migrations/logins.py --env-file .env.test` |

`logins.py` is only needed once per branch (or to rotate passwords).

### Adding a migration

1. Create the next numbered file, for example `migrations/002_extensions.sql`.
2. Make it safe to run again: `create table if not exists`,
   `add column if not exists`, `on conflict do nothing`.
3. Put the table's grants in the same file (revoke all, then grant), so each
   file owns its own permissions.
4. Neon-only features (lakebase_text) go inside a block that checks
   `pg_available_extensions`, so the same file also runs on CI's plain Postgres.
5. Run it on `dev`, then `test`, and run the tests.

## Tests

```
uv run pytest
```

- Model calls go to fake servers on localhost (`tests/fake_anthropic.py`):
  no API keys, no cost.
- Tests marked `db` run only against `.env.test`. If it's missing they are
  skipped; if it points at the same database as `.env` or `.env.production`,
  the run stops.
- Tests marked `neon` also need lakebase_text, so they run against the Neon
  `test` branch and are skipped elsewhere.
- CI starts a throwaway Postgres with pgvector, writes its own `.env.test`,
  runs the migrations and logins, then the tests. It uses no secrets.

## Corpus

| | |
|---|---|
| Source | Project Gutenberg eBook #674, *Plutarch: Lives of the noble Grecians and Romans*, Dryden's translation as revised by Arthur Hugh Clough ([plain text](https://www.gutenberg.org/ebooks/674.txt.utf-8)) |
| Pinned SHA-256 | `e849186d3483c1779053a5f70c8ae477316c019907c740f878de0d2c788975f8` (downloaded 2026-11-15) |
| Version | `lives-v1`, chunker `para-v1` |
| Lives | Cato the Elder (heading MARCUS CATO), Cato the Younger, Tiberius Gracchus, Caius Gracchus, Caesar, Pompey, Crassus |
| Size | 470 chunks, 123,739 words |

The Comparison essays (for example *Comparison of Crassus with Nicias*) are
left out of v1: they bring in people who aren't in the corpus.

### How the text is chunked (`core/corpus.py`)

- The text has **no chapter numbers**. Each Life is a capitalised heading
  (listed once in the contents and once in the text) followed by paragraphs.
  A Life ends at the next heading.
- `section` is the paragraph number within the Life, from 1.
- Paragraphs under 40 words (verse quotations) join the paragraph before.
- Paragraphs over 450 words are cut at sentence ends into pieces of about
  300 words (`chunk_index` 0, 1, ...). A last piece under 100 words joins the
  one before.
- Stored `text` is exactly the source's words. What gets **embedded** starts
  with `Plutarch, Life of <life>, section <n>.`, and the BM25 column indexes
  the Life's name with the text, because most paragraphs say "he", not the
  person's name.

### Loading it

```
uv run python -m scripts.fetch_corpus            # download once, check the hash
uv run python -m scripts.ingest --dry-run        # chunk counts per Life; no API, no database
uv run python -m scripts.ingest                  # embed and load into dev (.env)
```

`ingest` embeds only the chunks the database doesn't already have, one batch
per transaction, so a second run costs nothing and an interrupted run resumes.
It stops if the spend passes `DAILY_SPEND_CAP_USD`.

Voyage accounts without a payment method are limited to **3 requests and
10K tokens per minute** (Voyage's own 429 message). Adding a payment method
raises the limits, and the free tokens (200M for the Voyage 3 series) still
apply. Without one, the default batch of 64 chunks is refused. Pace the calls:

```
uv run python -m scripts.ingest --batch-size 10 --pause 62   # ~4K tokens per call, one call a minute
```

20-chunk batches (~8K tokens by word count) were still refused: Voyage counts
more tokens per word than the estimate, so leave plenty of room.

A 429 that still gets through is waited out (65 s, up to 5 times); other
errors stop the run, and the next run resumes where it stopped. `--fake-embeddings` loads
free hashed vectors under a separate version (`lives-v1-fake`), so they never
mix with real ones.

### Changing the corpus

Any change to the source file, the Lives or the chunking rules needs a new
`CORPUS_VERSION` (and a new `CHUNKER` name if the rules changed). `ingest`
refuses to load an existing version from a different source or chunker, and
cached answers are keyed to the version, so old answers are never served for
new text.

## Search

Two queries in `core/search.py`; lower scores are better for both, but the
scores can't be compared with each other, so hybrid search combines them by rank.

| Query | Index | Score |
|---|---|---|
| `vector_search` | HNSW on `embedding` (pgvector, cosine) | cosine distance, 0 = identical |
| `bm25_search` | `lakebase_bm25` on the generated `tsv` column (Neon only) | negative BM25; 0 = no shared words |

### Spike results (Neon test branch, 2026-11-15)

Run with `uv run --env-file .env.test python -m scripts.neon_spike` (Voyage
embeddings; needs `VOYAGE_API_KEY`) or with `--fake-embeddings` (no key).

- Neon Postgres 18.7 offers `vector` 0.8.7 and `lakebase_text` 0.1.3. Both
  install with the owner login, through the normal migration runner.
- **The read-only `delphi_query` login can run BM25 queries** (`<@>` and
  `to_bm25query` need no extra grants). The rank_bm25 fallback isn't needed.
- **Whether non-matching rows (score 0) come back depends on the plan.**
  Checked with `EXPLAIN` on 6 rows: the planner chose a sequential scan, which
  returned every row, zeros included; with `enable_seqscan = off` it used the
  `chunks_tsv_bm25` index scan, which returned only the match. `bm25_search`
  always filters out `score = 0`, so a non-match never earns a rank in fusion
  whichever plan runs.
- BM25 told the near-miss passages apart: "Opimius Aventine" ranked Caius
  Gracchus first, "Utica" ranked Cato the Younger first and returned nothing else.
- **Real embeddings (voyage-3.5) rank the near-miss brothers correctly, but
  barely.** For "Which brother was killed on the Capitol by the senators?":
  Tiberius Gracchus 0.5505, Caius Gracchus 0.5597 (cosine distance), a gap
  under 0.01. This is the case the project is built around: BM25 on exact
  names, and the cache's name check, have to cover what vector search alone
  can't separate. Seven short texts cost 241 tokens, $0.000014.
- BM25 statistics refresh at `VACUUM`; the ingest login has `MAINTAIN` on
  `chunks`, so a load can vacuum without the owner login.
- Use the **direct** connection string for `DATABASE_URL` (no `-pooler` in the
  host). The migration runner and `logins.py` need a session, not the pooler.
