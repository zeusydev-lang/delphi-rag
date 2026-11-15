"""Answer a question, from the cache when it's safe, otherwise fresh.

    reply = await answer_with_cache("How did Tiberius Gracchus die?")
    reply.result          # the AnswerResult (cached or new)
    reply.decision        # why: hit / judged_same / miss_signature / ...
    reply.usage           # what THIS request cost (embedding, judge, answer)

The question is embedded once; the same vector is used for the cache lookup,
for retrieval on a miss, and for storing the new answer.
"""

import os
from collections.abc import Sequence

from pydantic import BaseModel

from agent.answer import AnswerResult, answer_question
from agent.cache_judge import CacheDecision, decide, judge_client
from agent.llm import EmbeddingClient, LlmClient, LlmUsage, check_budget
from agent.prompts import PROMPT_VERSION
from core.cache import Versions, lookup, record_hit, store
from core.corpus import CORPUS_VERSION
from core.db import connect
from core.search import SEARCH_VERSION
from core.signature import signature


class CachedAnswer(BaseModel):
    result: AnswerResult
    decision: CacheDecision | None  # None: the cache was empty or skipped
    matched_question: str | None = None
    usage: LlmUsage


async def answer_with_cache(
    question: str,
    *,
    llm: LlmClient | None = None,
    judge: LlmClient | None = None,
    embedder: EmbeddingClient | None = None,
    query_vector: Sequence[float] | None = None,
    corpus_version: str = CORPUS_VERSION,
    cap_usd: float | None = None,
    use_cache: bool = True,
) -> CachedAnswer:
    llm = llm or LlmClient()
    cap = (
        float(os.environ.get("DAILY_SPEND_CAP_USD", "0.25"))
        if cap_usd is None
        else cap_usd
    )
    usage = LlmUsage(cost_usd=0.0)

    if query_vector is None:
        embedded = await (embedder or EmbeddingClient()).embed(
            [question], input_type="query"
        )
        usage.add(embedded.usage)
        check_budget(usage, cap)
        query_vector = embedded.vectors[0]

    versions = Versions(
        corpus_version=corpus_version,
        prompt_version=PROMPT_VERSION,
        search_version=SEARCH_VERSION,
        model=llm.model,
    )

    decision = None
    matched = None
    if use_cache:
        with connect("query") as conn:
            candidates = lookup(conn, query_vector, versions)
        # Prefer the nearest question with the same names/ordinals/numbers: the
        # very nearest may be the other brother.
        mine = signature(question)
        same_sig = [c for c in candidates if signature(c.question) == mine]
        candidate = (same_sig or candidates or [None])[0]
        if candidate is not None:
            decision = await decide(
                question,
                candidate.question,
                candidate.similarity,
                judge=judge or judge_client(),
                usage=usage,
            )
            check_budget(usage, cap)
            if decision.serves_cached:
                with connect("query") as conn:
                    record_hit(conn, candidate.id)
                return CachedAnswer(
                    result=AnswerResult.model_validate(candidate.answer),
                    decision=decision,
                    matched_question=candidate.question,
                    usage=usage,
                )
            matched = candidate.question

    result = await answer_question(
        question,
        llm=llm,
        query_vector=query_vector,
        corpus_version=corpus_version,
        cap_usd=max(cap - (usage.cost_usd or 0), 0),
    )
    usage.add(result.usage)
    if use_cache:
        with connect("query") as conn:
            store(
                conn, question, query_vector, versions, result.model_dump(mode="json")
            )
    return CachedAnswer(
        result=result, decision=decision, matched_question=matched, usage=usage
    )
