-- 006_eval_runs.sql
-- One row per eval run (retrieval, answers, cache): the summary rates, the
-- versions measured and the git commit. Per-question results stay in the
-- JSON files under evals/results/, which are the source of truth in git;
-- this table also keeps runs that were never committed.
-- Safe to run more than once. Apply with: uv run python migrations/migrate.py

-- A login for recording eval runs and nothing else.
do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'delphi_eval') then
        create role delphi_eval nologin;
    end if;
end;
$$;
grant usage on schema public to delphi_eval;

create table if not exists eval_runs (
    id            bigint generated always as identity primary key,
    kind          text not null check (kind in ('retrieval', 'answers', 'cache')),
    versions      jsonb not null,   -- search/prompt/cache/judge/corpus/label versions, models
    pair_set      text not null default 'main',  -- cache eval: main or heldout
    rates         jsonb not null,   -- the summary; per-question results are in results_file
    cost_usd      numeric(12, 6),
    git_commit    text,
    git_dirty     boolean,
    results_file  text,
    measured_at   timestamptz not null,
    recorded_at   timestamptz not null default now(),
    unique (kind, versions, pair_set, measured_at)
);

revoke all on eval_runs from delphi_ingest, delphi_query, delphi_eval;
grant select, insert (
    kind, versions, pair_set, rates, cost_usd, git_commit, git_dirty,
    results_file, measured_at
) on eval_runs to delphi_eval;
