"""The public API: auth, input limits, the daily cap, and what it reveals."""

import asyncio
import os
from contextlib import contextmanager

import psycopg
import pytest
from fastapi.testclient import TestClient

import api.app as app_module
from agent.answer import AnswerResult, CitedPassage
from agent.cache_judge import CacheDecision, decide
from agent.cached_answer import CachedAnswer
from agent.llm import LlmError, LlmUsage
from core.db import connect
from core.spend import add_spend, spent_today

KEY = "test-shared-secret"


def result(answer="Tiberius was killed on the Capitol.") -> AnswerResult:
    return AnswerResult(
        question="q",
        answerable=True,
        answer=answer,
        citations=[
            CitedPassage(
                chunk_id=99, life="Tiberius Gracchus", section=24, quote="clubs"
            )
        ],
        passages=[
            CitedPassage(chunk_id=99, life="Tiberius Gracchus", section=24, quote="")
        ],
        usage=LlmUsage(input_tokens=3000, output_tokens=300, cost_usd=0.0005),
        model="anthropic/claude-haiku-5-5",
        corpus_version="lives-v1",
    )


class Fake:
    """Stands in for the database and the answer pipeline."""

    def __init__(self, monkeypatch, spent=0.0, reply=None, error=None):
        self.spent = spent
        self.recorded: list[tuple[float, str]] = []
        self.calls: list[dict] = []

        @contextmanager
        def fake_connect(role):
            assert role == "query"
            yield None

        async def fake_answer(question, **kw):
            self.calls.append({"question": question, **kw})
            if error:
                raise error
            return reply

        monkeypatch.setattr(app_module, "connect", fake_connect)
        monkeypatch.setattr(app_module, "spent_today", lambda conn: self.spent)
        monkeypatch.setattr(
            app_module,
            "add_spend",
            lambda conn, cost, kind: self.recorded.append((cost, kind)),
        )
        monkeypatch.setattr(app_module, "answer_with_cache", fake_answer)


FRESH = CachedAnswer(result=result(), decision=None, usage=LlmUsage(cost_usd=0.0006))
HIT = CachedAnswer(
    result=result(),
    decision=CacheDecision(outcome="hit", similarity=0.9891),
    matched_question="How did Tiberius Gracchus die?",
    usage=LlmUsage(cost_usd=0.000001),
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_SHARED_SECRET", KEY)
    monkeypatch.setenv("SITE_DAILY_CAP_USD", "0.10")
    return TestClient(app_module.app)


def ask(client, question="How did Tiberius Gracchus die?", key=KEY):
    return client.post(
        "/ask", json={"question": question}, headers={"X-Delphi-Key": key}
    )


# --- Auth and input ------------------------------------------------------------------


class TestGatekeeping:
    def test_health_needs_no_key(self, client):
        assert client.get("/health").json() == {"ok": True}

    @pytest.mark.parametrize("key", ["", "wrong"])
    def test_wrong_or_missing_key(self, client, monkeypatch, key):
        fake = Fake(monkeypatch, reply=FRESH)
        assert ask(client, key=key).status_code == 401
        assert fake.calls == []

    def test_no_secret_configured_refuses_everyone(self, client, monkeypatch):
        monkeypatch.delenv("API_SHARED_SECRET")
        Fake(monkeypatch, reply=FRESH)
        assert ask(client, key="").status_code == 401

    @pytest.mark.parametrize("question", ["", "   ", "x" * 301])
    def test_question_length(self, client, monkeypatch, question):
        fake = Fake(monkeypatch, reply=FRESH)
        assert ask(client, question=question).status_code == 422
        assert fake.calls == []

    def test_whitespace_is_tidied(self, client, monkeypatch):
        fake = Fake(monkeypatch, reply=FRESH)
        ask(client, question="  How did\n Crassus   die?  ")
        assert fake.calls[0]["question"] == "How did Crassus die?"

    def test_no_api_docs_are_published(self, client):
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


# --- Answers and the cap ------------------------------------------------------------------


class TestAnswers:
    def test_fresh_answer(self, client, monkeypatch):
        fake = Fake(monkeypatch, reply=FRESH)
        body = ask(client).json()
        assert body["status"] == "answered"
        assert body["citations"] == [
            {"life": "Tiberius Gracchus", "section": 24, "quote": "clubs"}
        ]
        assert body["cache"]["served_from_cache"] is False
        assert body["cache_only"] is False
        assert fake.recorded == [(0.0006, "fresh")]
        assert fake.calls[0]["cache_only"] is False

    def test_cache_hit(self, client, monkeypatch):
        fake = Fake(monkeypatch, reply=HIT)
        body = ask(client).json()
        assert body["cache"] == {
            "served_from_cache": True,
            "outcome": "hit",
            "similarity": 0.989,
            "matched_question": "How did Tiberius Gracchus die?",
        }
        assert fake.recorded[0][1] == "hit"

    def test_nothing_internal_is_revealed(self, client, monkeypatch):
        Fake(monkeypatch, reply=FRESH)
        text = ask(client).text
        for leak in ("chunk_id", "cost", "usage", "prompt_version", "passages", "99"):
            assert leak not in text

    def test_over_cap_asks_for_cache_only(self, client, monkeypatch):
        fake = Fake(monkeypatch, spent=0.10, reply=HIT)
        body = ask(client).json()
        assert fake.calls[0]["cache_only"] is True
        assert body["cache_only"] is True and body["status"] == "answered"

    def test_over_cap_with_no_safe_cached_answer(self, client, monkeypatch):
        miss = CachedAnswer(
            result=None,
            decision=CacheDecision(outcome="gray_no_judge", similarity=0.9),
            usage=LlmUsage(cost_usd=0.000001),
            cache_only_miss=True,
        )
        fake = Fake(monkeypatch, spent=0.25, reply=miss)
        body = ask(client).json()
        assert body["status"] == "over_cap" and body["answer"] is None
        assert fake.recorded[0][1] == "refused"

    def test_rate_limited_upstream_is_a_503(self, client, monkeypatch):
        Fake(
            monkeypatch,
            error=LlmError("Embedding call failed: RateLimitError (HTTP 429)", 429),
        )
        r = ask(client)
        assert r.status_code == 503 and "minute" in r.json()["detail"]

    def test_other_failures_are_generic(self, client, monkeypatch):
        Fake(
            monkeypatch,
            error=LlmError("LLM call failed: APIError (HTTP 500) secret", 500),
        )
        r = ask(client)
        assert r.status_code == 502
        assert "secret" not in r.text and "500" not in r.text


def test_gray_zone_without_a_judge_is_a_miss():
    d = asyncio.run(
        decide("How did Crassus die?", "How did Crassus die?", 0.9, judge=None)
    )
    assert d.outcome == "gray_no_judge" and not d.serves_cached


# --- daily_spend (runs in CI) ------------------------------------------------------------


@pytest.fixture
def no_spend():
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        conn.execute("truncate daily_spend")
    yield


@pytest.mark.db
class TestDailySpend:
    def test_adds_up_for_today(self, no_spend):
        with connect("query") as conn:
            assert spent_today(conn) == 0.0
            add_spend(conn, 0.0005, "fresh")
            add_spend(conn, 0.000001, "hit")
            add_spend(conn, 0.0, "refused")
            assert spent_today(conn) == pytest.approx(0.000501)
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            row = conn.execute(
                "select fresh_answers, cache_hits, refused_over_cap from daily_spend"
            ).fetchone()
        assert row == (1, 1, 1)

    def test_query_login_cannot_delete_or_reset(self, no_spend):
        with connect("query") as conn:
            add_spend(conn, 0.05, "fresh")
        denied = pytest.raises(psycopg.errors.InsufficientPrivilege)
        with denied, connect("query") as conn:
            conn.execute("delete from daily_spend")
