-- 001_roles.sql
-- Least-privilege database logins, one per job.
-- Safe to run more than once. Apply with: uv run python migrations/migrate.py
--
-- Roles are created WITHOUT login here, so no passwords live in git.
-- Turn on login and set passwords with: uv run python migrations/logins.py
--
--   delphi_ingest  writes chunks of the corpus. Cannot delete anything.
--   delphi_query   reads chunks; reads and writes the answer cache.
--
-- Table grants live in the migration that creates each table, so this file
-- never needs to know about later tables.

do $$
declare
    r text;
begin
    foreach r in array array['delphi_ingest', 'delphi_query'] loop
        if not exists (select 1 from pg_roles where rolname = r) then
            execute format('create role %I nologin', r);
        end if;
    end loop;
end;
$$;

grant usage on schema public to delphi_ingest, delphi_query;
