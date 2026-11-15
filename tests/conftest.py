"""Test setup.

Database tests (marked `db`) run ONLY against the test database in .env.test:
a Neon test branch locally, or the throwaway Postgres service in CI.
Tests that need Neon-only extensions (lakebase_text) are also marked `neon`;
they are skipped when the test database doesn't offer the extension.

Safety:
  - If .env.test is missing or incomplete, database tests are skipped.
  - If .env.test points at the same host as .env (dev) or .env.production,
    the whole test run stops before touching anything.

One-time setup for the test branch:
    uv run --env-file .env.test python migrations/migrate.py
    uv run python migrations/logins.py --env-file .env.test
"""

import os
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
TEST_ENV = ROOT / ".env.test"
KEYS = (
    "DATABASE_URL",
    "INGEST_DATABASE_URL",
    "QUERY_DATABASE_URL",
)
NEON_EXTENSIONS = ("lakebase_text",)


def _host(url: str) -> str | None:
    return urlsplit(url).hostname


def _other_hosts() -> dict[str, str]:
    hosts = {}
    for name in (".env", ".env.production"):
        path = ROOT / name
        url = dotenv_values(path).get("DATABASE_URL") if path.exists() else None
        if url and _host(url):
            hosts[name] = _host(url)
    return hosts


def _load_test_env() -> tuple[str | None, str | None]:
    """Point every login at the test branch.

    Returns (skip_reason, stop_reason). Both None means database tests can run.
    """
    if not TEST_ENV.exists():
        return ".env.test not found (see tests/conftest.py for setup)", None

    values = dotenv_values(TEST_ENV)
    missing = [k for k in KEYS if not values.get(k)]
    if missing:
        return (
            f".env.test is missing {', '.join(missing)}. "
            "Run: uv run python migrations/logins.py --env-file .env.test"
        ), None

    hosts = {_host(values[k]) for k in KEYS}
    if len(hosts) != 1:
        return None, ".env.test logins point at different hosts"
    test_host = hosts.pop()
    for name, host in _other_hosts().items():
        if host == test_host:
            return (
                None,
                f".env.test points at the same database as {name}. Refusing to run.",
            )

    # Set before core.settings is imported, and override anything from .env.
    for k in KEYS:
        os.environ[k] = values[k]
    return None, None


def _missing_extensions() -> list[str]:
    """Neon-only extensions the test database can't install."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=15) as conn:
        rows = conn.execute(
            "select name from pg_available_extensions where name = any(%s)",
            [list(NEON_EXTENSIONS)],
        ).fetchall()
    available = {name for (name,) in rows}
    return [e for e in NEON_EXTENSIONS if e not in available]


SKIP_REASON, STOP_REASON = _load_test_env()


def pytest_configure(config):
    config.addinivalue_line("markers", "db: needs the test database (.env.test)")
    config.addinivalue_line(
        "markers", "neon: needs Neon-only extensions (lakebase_text)"
    )
    if STOP_REASON:
        pytest.exit(STOP_REASON, returncode=2)


def pytest_collection_modifyitems(config, items):
    db_skip = pytest.mark.skip(reason=SKIP_REASON) if SKIP_REASON else None
    neon_skip = None
    if not SKIP_REASON and any("neon" in item.keywords for item in items):
        missing = _missing_extensions()
        if missing:
            neon_skip = pytest.mark.skip(
                reason=f"test database has no {', '.join(missing)} (Neon only)"
            )
    for item in items:
        if db_skip and ("db" in item.keywords or "neon" in item.keywords):
            item.add_marker(db_skip)
        elif neon_skip and "neon" in item.keywords:
            item.add_marker(neon_skip)
