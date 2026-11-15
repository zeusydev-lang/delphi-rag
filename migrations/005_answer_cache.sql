-- 005_answer_cache.sql
-- Semantic answer cache: questions already answered, with their answers.
-- A cached answer is only ever served for the same corpus, prompt, search and
-- model versions, so changing any of them starts a fresh cache.
-- Safe to run more than once. Apply with: uv run python migrations/migrate.py

create table if not exists answer_cache (
    id                  bigint generated always as identity primary key,
    corpus_version      text not null,
    prompt_version      text not null,
    search_version      text not null,
    model               text not null,
    question            text not null check (length(question) > 0),
    question_embedding  vector(1024) not null,
    signature           jsonb not null,   -- names, ordinals, numbers (core/signature.py)
    answer              jsonb not null,   -- the full AnswerResult
    created_at          timestamptz not null default now(),
    hit_count           int not null default 0,
    last_hit_at         timestamptz
);

create index if not exists answer_cache_embedding_hnsw
    on answer_cache using hnsw (question_embedding vector_cosine_ops);
create index if not exists answer_cache_versions_idx
    on answer_cache (corpus_version, prompt_version, search_version, model);

-- Permissions: the query job reads and adds cached answers and counts hits.
-- It can't change or delete an answer. Ingest has no access.
revoke all on answer_cache from delphi_ingest, delphi_query;
grant select, insert (
    corpus_version, prompt_version, search_version, model,
    question, question_embedding, signature, answer
) on answer_cache to delphi_query;
grant update (hit_count, last_hit_at) on answer_cache to delphi_query;
