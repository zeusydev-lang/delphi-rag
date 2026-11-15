"""Answer a question from retrieved passages, with checked citations.

    result = await answer_question("How did Cato die at Utica?")
    result.answer, result.citations      # each citation: chunk, life, section, quote

Steps: embed the question, hybrid search (top 5), then one structured call to
the answer model (Claude Haiku 5.5 by default, via LLM_MODEL). Every citation
must name a passage it was given and quote it exactly; answer_model_for()
checks that, and a failed check goes back to the model once to fix
(generate_structured's retry).
"""

import os
import re
from collections.abc import Mapping, Sequence
from html import unescape
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.llm import EmbeddingClient, LlmClient, LlmUsage, check_budget
from agent.prompts import ANSWER_INSTRUCTIONS, PROMPT_VERSION, answer_input
from agent.structured import generate_structured
from core.corpus import CORPUS_VERSION
from core.db import connect
from core.search import SEARCH_VERSION, FusedHit, hybrid_search

ANSWER_K = 5  # passages given to the model (hybrid hit@5 on the eval: 0.85)
MAX_QUOTE_CHARS = 300


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Citation(_Output):
    passage: str = Field(pattern=r"^P\d+$", description='Passage id, e.g. "P2"')
    quote: str = Field(
        min_length=3,
        max_length=MAX_QUOTE_CHARS,
        description="A short exact quote from that passage, copied word for word",
    )


class AnswerOutput(_Output):
    """What the model returns."""

    answerable: bool = Field(
        description="False if the passages don't answer the question"
    )
    answer: str = Field(min_length=1, max_length=2000)
    citations: list[Citation] = Field(default_factory=list, max_length=10)


_QUOTE_MARKS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalise(text: str) -> str:
    """For comparing quotes: unescaped, straight quote marks, collapsed spaces, casefolded."""
    text = unescape(text).translate(_QUOTE_MARKS)
    return re.sub(r"\s+", " ", text).strip().strip("\"'").strip().casefold()


def answer_model_for(passages: Mapping[str, str]) -> type[AnswerOutput]:
    """AnswerOutput plus a check that every citation quotes a passage it was given.

    Error messages name the citation and passage only, never the quoted text.
    """
    texts = {pid: normalise(text) for pid, text in passages.items()}

    class Answer(AnswerOutput):
        @model_validator(mode="after")
        def citations_are_real(self) -> Self:
            problems = []
            for n, c in enumerate(self.citations, start=1):
                if c.passage not in texts:
                    problems.append(
                        f"citation {n} names {c.passage}, which wasn't given"
                    )
                elif normalise(c.quote) not in texts[c.passage]:
                    problems.append(
                        f"citation {n}: the quote isn't in {c.passage} word for word"
                    )
            if self.answerable and not self.citations:
                problems.append("an answer needs at least one citation")
            if problems:
                raise ValueError("; ".join(problems))
            return self

    return Answer


# --- The whole step ------------------------------------------------------------


class CitedPassage(BaseModel):
    chunk_id: int
    life: str
    section: int
    quote: str


class AnswerResult(BaseModel):
    question: str
    answerable: bool
    answer: str
    citations: list[CitedPassage]
    passages: list[CitedPassage]  # what the model was given (quote empty)
    usage: LlmUsage
    model: str
    prompt_version: str = PROMPT_VERSION
    search_version: str = SEARCH_VERSION
    corpus_version: str


def _cap() -> float:
    return float(os.environ.get("DAILY_SPEND_CAP_USD", "0.25"))


async def answer_from_hits(
    question: str,
    hits: Sequence[FusedHit],
    *,
    llm: LlmClient,
    usage: LlmUsage,
    corpus_version: str,
    cap_usd: float,
) -> AnswerResult:
    """The model call and citation mapping, for passages already retrieved."""
    by_id = {f"P{n}": h for n, h in enumerate(hits, start=1)}
    output = await generate_structured(
        llm,
        answer_model_for({pid: h.text for pid, h in by_id.items()}),
        ANSWER_INSTRUCTIONS,
        answer_input(question, hits),
        max_tokens=2000,
        usage=usage,
    )
    check_budget(usage, cap_usd)
    return AnswerResult(
        question=question,
        answerable=output.answerable,
        answer=output.answer,
        citations=[
            CitedPassage(
                chunk_id=by_id[c.passage].chunk_id,
                life=by_id[c.passage].life,
                section=by_id[c.passage].section,
                quote=c.quote,
            )
            for c in output.citations
        ],
        passages=[
            CitedPassage(chunk_id=h.chunk_id, life=h.life, section=h.section, quote="")
            for h in hits
        ],
        usage=usage,
        model=llm.model,
        corpus_version=corpus_version,
    )


async def answer_question(
    question: str,
    *,
    llm: LlmClient | None = None,
    embedder: EmbeddingClient | None = None,
    query_vector: Sequence[float] | None = None,  # skip embedding (evals embed in bulk)
    corpus_version: str = CORPUS_VERSION,
    k: int = ANSWER_K,
    cap_usd: float | None = None,
) -> AnswerResult:
    """Embed, retrieve (hybrid, top k), answer with checked citations."""
    llm = llm or LlmClient()
    cap = _cap() if cap_usd is None else cap_usd
    usage = LlmUsage(cost_usd=0.0)

    if query_vector is None:
        embedded = await (embedder or EmbeddingClient()).embed(
            [question], input_type="query"
        )
        usage.add(embedded.usage)
        check_budget(usage, cap)
        query_vector = embedded.vectors[0]

    with connect("query") as conn:
        hits = hybrid_search(
            conn, question, query_vector, corpus_version=corpus_version, k=k
        )
    return await answer_from_hits(
        question,
        hits,
        llm=llm,
        usage=usage,
        corpus_version=corpus_version,
        cap_usd=cap,
    )
