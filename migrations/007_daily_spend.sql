-- 007_daily_spend.sql
-- What the public site spent each day (UTC), for the site's daily cap.
-- Past the cap, the site answers only from the cache (core/spend.py).
-- Safe to run more than once. Apply with: uv run python migrations/migrate.py

create table if not exists daily_spend (
    day               date primary key,
    cost_usd          numeric(12, 6) not null default 0 check (cost_usd >= 0),
    fresh_answers     int not null default 0,
    cache_hits        int not null default 0,
    refused_over_cap  int not null default 0
);

-- The API (query login) reads today's total and adds to it. It can't delete
-- rows or rewrite past days' totals except by adding.
revoke all on daily_spend from delphi_ingest, delphi_query, delphi_eval;
grant select, insert (day, cost_usd, fresh_answers, cache_hits, refused_over_cap),
      update (cost_usd, fresh_answers, cache_hits, refused_over_cap)
    on daily_spend to delphi_query;
