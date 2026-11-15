"""EmbeddingClient, usage totals and the budget check. No key, no cost."""

import asyncio

import pytest

from agent.llm import (
    BudgetExceeded,
    EmbeddingClient,
    LlmError,
    LlmUsage,
    check_budget,
    price,
)
from tests.fake_anthropic import FakeEmbeddings, fake_vector


@pytest.fixture(scope="module")
def fake():
    server = FakeEmbeddings()
    yield server
    server.close()


@pytest.fixture
def embedder(fake, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test-not-real")
    fake.requests.clear()
    fake.fail_with = None
    return EmbeddingClient(
        "voyage/voyage-3.5", api_base=fake.url, num_retries=0, batch_size=2
    )


def run(coro):
    return asyncio.run(coro)


class TestEmbed:
    def test_one_vector_per_text_in_order(self, embedder):
        texts = ["Tiberius Gracchus", "Caius Gracchus", "Cato the Younger"]
        result = run(embedder.embed(texts, input_type="document"))
        # The fake answers out of order; the client sorts by index.
        assert result.vectors == [fake_vector(t) for t in texts]
        assert result.model == "voyage/voyage-3.5"

    def test_batches_requests(self, embedder, fake):
        run(embedder.embed(["a", "b", "c"], input_type="document"))
        assert [r["input"] for r in fake.requests] == [["a", "b"], ["c"]]

    @pytest.mark.parametrize("input_type", ["document", "query"])
    def test_input_type_reaches_voyage(self, embedder, fake, input_type):
        run(embedder.embed(["Who was Crassus?"], input_type=input_type))
        assert fake.requests[-1]["input_type"] == input_type
        assert fake.requests[-1]["model"] == "voyage-3.5"

    def test_usage_and_cost_add_up_across_batches(self, embedder):
        result = run(embedder.embed(["a b", "c", "d e f"], input_type="query"))
        assert result.usage.input_tokens == 6  # the fake counts words
        assert result.usage.output_tokens == 0
        assert result.usage.cost_usd == pytest.approx(6 * 0.06 / 1e6)

    def test_unknown_model_has_no_cost(self, fake, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-test-not-real")
        client = EmbeddingClient("voyage/voyage-unknown", api_base=fake.url)
        result = run(client.embed(["a"], input_type="query"))
        assert result.usage.cost_usd is None

    def test_empty_input_makes_no_call(self, embedder, fake):
        result = run(embedder.embed([], input_type="query"))
        assert result.vectors == [] and fake.requests == []

    def test_errors_are_redacted(self, embedder, fake):
        fake.fail_with = 401
        with pytest.raises(LlmError) as info:
            run(embedder.embed(["a"], input_type="query"))
        message = str(info.value)
        assert "401" in message
        assert "secret-looking" not in message
        assert "pa-test-not-real" not in message

    def test_model_from_environment(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "voyage/voyage-3.5-lite")
        assert EmbeddingClient().model == "voyage/voyage-3.5-lite"


class TestUsageAndBudget:
    def test_price(self):
        assert price("anthropic/claude-haiku-5-5", 1_000_000, 1_000_000) == 0.60
        assert price("nobody/knows", 10) is None

    def test_add_keeps_unknown_cost_unknown(self):
        total = LlmUsage()
        total.add(LlmUsage(input_tokens=5, cost_usd=None))
        assert (total.input_tokens, total.cost_usd) == (5, None)
        total.add(LlmUsage(input_tokens=5, output_tokens=2, cost_usd=0.01))
        assert (total.input_tokens, total.output_tokens) == (10, 2)
        assert total.cost_usd == 0.01

    def test_under_cap_is_fine(self):
        check_budget(LlmUsage(cost_usd=0.24), cap=0.25)
        check_budget(LlmUsage(cost_usd=None), cap=0.25)  # unknown cost can't stop

    def test_over_cap_stops(self):
        with pytest.raises(BudgetExceeded, match="over the \\$0.25 cap"):
            check_budget(LlmUsage(cost_usd=0.26), cap=0.25)
