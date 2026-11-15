-- 004_sections_and_versions.sql
-- 1. chunks.chapter -> section: the Gutenberg text has no chapter numbers, so
--    positions are paragraph sections within each Life, and the name says so.
-- 2. BM25 indexes the Life's name with the text: most paragraphs say "he",
--    so "Tiberius" must match Tiberius's chunks even when the text doesn't name him.
-- 3. corpus_versions: where each corpus version came from. The cache keys on it.
-- Safe to run more than once. Apply with: uv run python migrations/migrate.py

-- 1. Rename (only if not done yet) -------------------------------------------
do $$
begin
    if exists (
        select 1 from information_schema.columns
         where table_schema = 'public' and table_name = 'chunks' and column_name = 'chapter'
    ) then
        alter table chunks rename column chapter to section;
    end if;
end;
$$;

-- 2. tsv includes the Life's name (only if not done yet) ----------------------
-- SET EXPRESSION (Postgres 17+) rewrites the column and rebuilds its indexes,
-- including the BM25 one, so it is skipped once the expression is already right.
do $$
begin
    if not exists (
        select 1
          from pg_attrdef d
          join pg_attribute a on a.attrelid = d.adrelid and a.attnum = d.adnum
         where d.adrelid = 'public.chunks'::regclass
           and a.attname = 'tsv'
           and pg_get_expr(d.adbin, d.adrelid) like '%life%'
    ) then
        alter table chunks alter column tsv
            set expression as (to_tsvector('english', life || ' ' || text));
    end if;
end;
$$;

-- Permissions on chunks (moved here from 003; see the note there) ---------------
revoke all on chunks from delphi_ingest, delphi_query;

-- Ingest adds chunks (ON CONFLICT DO NOTHING makes reloads safe). It can't
-- change or delete them. MAINTAIN (Postgres 17) lets it VACUUM after a load,
-- so BM25 statistics are fresh without using the owner login.
grant select, insert (corpus_version, life, section, chunk_index, text, embedding)
    on chunks to delphi_ingest;
grant maintain on chunks to delphi_ingest;

-- Query only reads.
grant select on chunks to delphi_query;

-- 3. Corpus versions ------------------------------------------------------------
create table if not exists corpus_versions (
    version        text primary key,      -- e.g. 'lives-v1'
    source_url     text not null,
    source_sha256  text not null check (source_sha256 ~ '^[0-9a-f]{64}$'),
    chunker        text not null,         -- e.g. 'para-v1'
    lives          text[] not null,
    chunk_count    int  not null check (chunk_count >= 0),
    created_at     timestamptz not null default now()
);

revoke all on corpus_versions from delphi_ingest, delphi_query;
grant select, insert on corpus_versions to delphi_ingest;
grant select on corpus_versions to delphi_query;
