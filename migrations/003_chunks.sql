-- 003_chunks.sql
-- Chunks of the corpus, searchable two ways:
--   embedding  vector search (pgvector, HNSW, cosine distance)
--   tsv        BM25 keyword search (lakebase_bm25 index, Neon only)
-- Safe to run more than once. Apply with: uv run python migrations/migrate.py

create table if not exists chunks (
    id              bigint generated always as identity primary key,
    corpus_version  text not null,   -- e.g. 'lives-v1'; cache entries are keyed to it
    life            text not null,   -- e.g. 'Tiberius Gracchus'
    chapter         int  not null check (chapter >= 0),  -- renamed to section in 004
    chunk_index     int  not null check (chunk_index >= 0),
    text            text not null check (length(text) > 0),
    embedding       vector(1024) not null,   -- voyage-3.5 default size
    tsv             tsvector generated always as (to_tsvector('english', text)) stored,
    created_at      timestamptz not null default now(),
    unique (corpus_version, life, chapter, chunk_index)
);

create index if not exists chunks_embedding_hnsw
    on chunks using hnsw (embedding vector_cosine_ops);

-- BM25 statistics are refreshed at VACUUM. Neon's lakebase_text docs recommend
-- vacuuming insert-only tables promptly, so scores stay current after loads.
alter table chunks set (autovacuum_vacuum_insert_scale_factor = 0);

-- The BM25 index, only where lakebase_text is installed (see 002).
do $$
begin
    if exists (select 1 from pg_extension where extname = 'lakebase_text') then
        execute 'create index if not exists chunks_tsv_bm25 on chunks using lakebase_bm25 (tsv)';
    end if;
end;
$$;

-- Permissions for chunks live in 004, after `chapter` is renamed to
-- `section` (the runner re-applies every file, so a grant naming the old
-- column here would fail on every later run).
