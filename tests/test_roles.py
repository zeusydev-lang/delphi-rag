"""The per-job logins exist, can connect, and can't change the schema."""

import psycopg
import pytest

from core.db import connect

pytestmark = pytest.mark.db


@pytest.mark.parametrize("role", ["ingest", "query"])
def test_login_connects_as_its_own_role(role):
    with connect(role) as conn:
        row = conn.execute(
            "select current_user as who, current_setting('application_name') as app"
        ).fetchone()
    assert row == {"who": f"delphi_{role}", "app": f"delphi-{role}"}


@pytest.mark.parametrize("role", ["ingest", "query"])
def test_login_cannot_create_tables(role):
    with pytest.raises(psycopg.errors.InsufficientPrivilege), connect(role) as conn:
        conn.execute("create table public.not_allowed (id int)")
