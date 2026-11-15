"""The HTTP API behind the public site. Only the Cloudflare Worker calls it.

    POST /ask   {"question": "..."}  ->  answer, citations, cache info
    GET  /health                      ->  {"ok": true}

Requests need X-Delphi-Key equal to API_SHARED_SECRET. The Worker handles
Turnstile and rate limits; this API enforces the question length again and
the site's daily spend cap (core/spend.py): past the cap, it answers only
from the cache (clear hits, no model calls).

Run locally:
    uv run uvicorn api.app:app --reload
"""

import hmac
import logging
import os
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from agent.cached_answer import CachedAnswer, answer_with_cache
from agent.llm import BudgetExceeded, EmbeddingClient, LlmError
from agent.structured import StructuredOutputError
from core.db import connect
from core.spend import add_spend, site_cap, spent_today

log = logging.getLogger("delphi.api")
MAX_QUESTION_CHARS = 300
PER_REQUEST_CAP_USD = 0.01  # one answer is ~$0.0005; this only stops runaways

app = FastAPI(title="delphi-rag", docs_url=None, redoc_url=None, openapi_url=None)


# --- Auth ------------------------------------------------------------------------


def require_key(x_delphi_key: str = Header(default="")) -> None:
    expected = os.environ.get("API_SHARED_SECRET", "")
    if not expected or not hmac.compare_digest(x_delphi_key, expected):
        raise HTTPException(status_code=401, detail="Not allowed.")


# --- Shapes ------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def sensible_length(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("Ask a question.")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"Questions can be up to {MAX_QUESTION_CHARS} characters.")
        return v


class Citation(BaseModel):
    life: str
    section: int
    quote: str


class CacheInfo(BaseModel):
    served_from_cache: bool
    outcome: str | None = None  # hit, judged_same, miss_signature, ...
    similarity: float | None = None
    matched_question: str | None = None


class AskResponse(BaseModel):
    status: Literal["answered", "over_cap"]
    answer: str | None = None
    answerable: bool | None = None
    citations: list[Citation] = Field(default_factory=list)
    cache: CacheInfo
    cache_only: bool  # the site is over today's cap


def to_response(reply: CachedAnswer, cache_only: bool) -> AskResponse:
    d = reply.decision
    cache = CacheInfo(
        served_from_cache=bool(d and d.serves_cached),
        outcome=d.outcome if d else None,
        similarity=round(d.similarity, 3) if d else None,
        matched_question=reply.matched_question if d and d.serves_cached else None,
    )
    if reply.result is None:
        return AskResponse(status="over_cap", cache=cache, cache_only=cache_only)
    r = reply.result
    return AskResponse(
        status="answered",
        answer=r.answer,
        answerable=r.answerable,
        citations=[
            Citation(life=c.life, section=c.section, quote=c.quote) for c in r.citations
        ],
        cache=cache,
        cache_only=cache_only,
    )


# --- Routes ------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
async def ask(body: AskRequest) -> AskResponse:
    with connect("query") as conn:
        cache_only = spent_today(conn) >= site_cap()
    try:
        reply = await answer_with_cache(
            body.question,
            embedder=EmbeddingClient(num_retries=0),
            cap_usd=PER_REQUEST_CAP_USD,
            cache_only=cache_only,
        )
    except LlmError as err:
        log.warning("model call failed: %s", type(err.__cause__).__name__)
        if err.status_code == 429:
            raise HTTPException(503, "Busy right now. Try again in a minute.") from None
        raise HTTPException(502, "Couldn't answer right now.") from None
    except (StructuredOutputError, BudgetExceeded) as err:
        log.warning("answer failed: %s", type(err).__name__)
        raise HTTPException(502, "Couldn't answer right now.") from None

    served = bool(reply.decision and reply.decision.serves_cached)
    kind = "refused" if reply.cache_only_miss else "hit" if served else "fresh"
    with connect("query") as conn:
        add_spend(conn, reply.usage.cost_usd or 0.0, kind)
    return to_response(reply, cache_only)
