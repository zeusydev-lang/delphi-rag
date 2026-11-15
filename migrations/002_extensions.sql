-- 002_extensions.sql
-- Search extensions. Safe to run more than once.
-- Apply with: uv run python migrations/migrate.py
--
--   vector         pgvector: embedding column and nearest-neighbour search.
--   lakebase_text  Neon's BM25 index (replaces the deprecated pg_search).
--                  Neon only, so it is installed only where it is available;
--                  CI's plain Postgres skips it and the `neon` tests skip too.

create extension if not exists vector;

do $$
begin
    if exists (select 1 from pg_available_extensions where name = 'lakebase_text') then
        create extension if not exists lakebase_text;
    else
        raise notice 'lakebase_text is not available here; BM25 search is off';
    end if;
end;
$$;
