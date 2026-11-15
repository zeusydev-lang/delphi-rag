"""Structured output (final_answer tool) against the fake Anthropic server."""

import asyncio
import copy

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.datatypes import Message, ToolCall
from agent.llm import LlmClient, LlmResponse, LlmUsage
from agent.structured import (
    StructuredOutputError,
    generate_structured,
    make_output_tool,
    parse_output,
)
from tests.fake_anthropic import FakeAnthropic, text_reply, tool_reply

# --- Small local models (stand-ins for the real answer models) ----------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Citation(_Strict):
    chunk_id: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=300)


class Answer(_Strict):
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1, max_length=5)


class Verdict(_Strict):
    same_question: bool


ANSWER = {
    "text": "Tiberius Gracchus proposed the land law in 133 BC.",
    "citations": [
        {"chunk_id": 12, "quote": "Tiberius, being elected tribune of the people"},
        {"chunk_id": 13, "quote": "he proposed the agrarian law"},
    ],
}
VERDICT = {"same_question": True}


def answer(args: dict, call_id: str = "toolu_1", **kw) -> dict:
    return tool_reply([(call_id, "final_answer", args)], **kw)


@pytest.fixture(scope="module")
def fake():
    server = FakeAnthropic()
    yield server
    server.close()


@pytest.fixture
def client(fake, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    fake.reset()
    return LlmClient("anthropic/claude-haiku-5-5", api_base=fake.url, num_retries=0)


def run(coro):
    return asyncio.run(coro)


# --- make_output_tool ---------------------------------------------------------


class TestMakeOutputTool:
    def test_definition_uses_the_model_schema(self):
        definition = make_output_tool(Answer).tool_definition["function"]
        assert definition["name"] == "final_answer"
        assert "Answer" in definition["description"]
        params = definition["parameters"]
        assert {"text", "citations"} <= set(params["properties"])
        assert params["additionalProperties"] is False

    def test_custom_name_and_description(self):
        tool = make_output_tool(Verdict, name="submit", description="Send it")
        assert (tool.name, tool.description) == ("submit", "Send it")


# --- parse_output -------------------------------------------------------------


def response_with(*calls: ToolCall) -> LlmResponse:
    return LlmResponse(content=list(calls), stop_reason="tool_calls")


class TestParseOutput:
    def test_valid_arguments_become_the_model(self):
        response = response_with(
            ToolCall(tool_call_id="t", name="final_answer", arguments=ANSWER)
        )
        result = parse_output(response, Answer)
        assert isinstance(result, Answer)
        assert result.citations[1].chunk_id == 13

    def test_no_output_call(self):
        response = LlmResponse(
            content=[Message(role="assistant", content="Here you go")],
            stop_reason="stop",
        )
        with pytest.raises(StructuredOutputError, match="didn't call final_answer"):
            parse_output(response, Answer)

    def test_invalid_arguments(self):
        bad = ANSWER | {"citations": []}  # needs at least one
        response = response_with(
            ToolCall(tool_call_id="t", name="final_answer", arguments=bad)
        )
        with pytest.raises(ValidationError, match="citations"):
            parse_output(response, Answer)


# --- generate_structured ------------------------------------------------------


class TestGenerateStructured:
    def test_answer(self, client, fake):
        fake.reset(answer(ANSWER))
        result = run(generate_structured(client, Answer, "Answer.", "Question"))
        assert result.text.startswith("Tiberius Gracchus")

    def test_small_model(self, client, fake):
        fake.reset(answer(VERDICT))
        result = run(generate_structured(client, Verdict, "Judge.", "Two questions"))
        assert result.same_question is True

    def test_request_forces_the_output_tool(self, client, fake):
        fake.reset(answer(ANSWER))
        run(generate_structured(client, Answer, "Answer.", "Question text"))
        body = fake.requests[-1]
        assert body["tool_choice"]["type"] == "any"  # litellm's "required"
        assert [t["name"] for t in body["tools"]] == ["final_answer"]
        assert "citations" in body["tools"][0]["input_schema"]["properties"]
        assert body["messages"][0]["content"][0]["text"] == "Question text"

    def test_nested_model_schema_is_sent(self, client, fake):
        fake.reset(answer(ANSWER))
        run(generate_structured(client, Answer, "Answer.", "Question"))
        schema = fake.requests[-1]["tools"][0]["input_schema"]
        assert "Citation" in str(schema)  # nested model travels via $defs

    def test_retry_shows_the_model_its_mistakes(self, client, fake):
        bad = ANSWER | {"citations": [{"chunk_id": 0, "quote": "Private chunk text"}]}
        fake.reset(answer(bad, call_id="toolu_bad"), answer(ANSWER, call_id="toolu_ok"))
        result = run(generate_structured(client, Answer, "Answer.", "Question"))
        assert result.citations[0].chunk_id == 12
        assert len(fake.requests) == 2

        retry_messages = fake.requests[1]["messages"]
        assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
        tool_result = retry_messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "toolu_bad"
        feedback = str(tool_result["content"])
        assert "chunk_id" in feedback
        assert "Private chunk text" not in feedback  # values aren't echoed back

    def test_retry_when_the_tool_was_not_called(self, client, fake):
        fake.reset(text_reply("I'd rather chat."), answer(VERDICT))
        result = run(generate_structured(client, Verdict, "Judge.", "Questions"))
        assert result.same_question is True
        assert len(fake.requests) == 2

    def test_gives_up_after_max_attempts(self, client, fake):
        bad = copy.deepcopy(ANSWER)
        bad["citations"][0]["chunk_id"] = -1
        fake.reset(answer(bad))
        with pytest.raises(StructuredOutputError, match="after 2 attempts"):
            run(generate_structured(client, Answer, "Answer.", "Question"))
        assert len(fake.requests) == 2

    def test_usage_adds_up_across_attempts(self, client, fake):
        bad = ANSWER | {"text": ""}
        fake.reset(
            answer(bad, input_tokens=1000, output_tokens=100),
            answer(ANSWER, input_tokens=2000, output_tokens=300),
        )
        usage = LlmUsage()
        run(generate_structured(client, Answer, "A.", "Q", usage=usage))
        assert (usage.input_tokens, usage.output_tokens) == (3000, 400)
        assert usage.cost_usd == pytest.approx((3000 * 0.10 + 400 * 0.50) / 1e6)
