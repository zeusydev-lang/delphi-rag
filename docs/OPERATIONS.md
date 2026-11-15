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

The Neon project (Postgres 17) has two branches to start with, each with its
own env file. None of these files is committed to git.

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
