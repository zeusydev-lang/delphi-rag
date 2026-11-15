"""Per-role connection settings and the vector text helper."""

import pytest

from core.db import to_vector
from core.settings import SettingsError, database_url


class TestDatabaseUrl:
    def test_each_role_reads_its_own_variable(self, monkeypatch):
        monkeypatch.setenv("INGEST_DATABASE_URL", "postgresql://ingest@h/db")
        monkeypatch.setenv("QUERY_DATABASE_URL", "postgresql://query@h/db")
        assert database_url("ingest") == "postgresql://ingest@h/db"
        assert database_url("query") == "postgresql://query@h/db"

    def test_missing_says_how_to_fix_it(self, monkeypatch):
        monkeypatch.delenv("QUERY_DATABASE_URL", raising=False)
        with pytest.raises(SettingsError, match="QUERY_DATABASE_URL.*logins.py"):
            database_url("query")

    def test_blank_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("INGEST_DATABASE_URL", "   ")
        with pytest.raises(SettingsError):
            database_url("ingest")

    def test_owner_login_is_not_a_role(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://owner@h/db")
        with pytest.raises(SettingsError, match="Unknown role"):
            database_url("owner")  # type: ignore[arg-type]


class TestToVector:
    def test_text_form(self):
        assert to_vector([0.5, -1, 2.25]) == "[0.5,-1.0,2.25]"

    @pytest.mark.parametrize("bad", [[], [float("nan")], [1.0, float("inf")]])
    def test_refuses_bad_vectors(self, bad):
        with pytest.raises(ValueError):
            to_vector(bad)
