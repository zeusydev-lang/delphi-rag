"""The answer cache table: find, store and count cached answers.

Rows are keyed by four versions (corpus, prompt, search, model); a lookup
only ever sees rows with all four equal to the current ones.
"""

from collections.abc import Sequence

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from core.db import to_vector
from core.signature import signature

# The gate (agent/cache_judge.py). Fixed from the spec before measuring.
HIT_AT = 0.95  # at or above, with the same signature: serve the cached answer
MISS_BELOW = 0.85  # below: answer fresh; in between: ask the judge
CANDIDATES = 5  # nearest cached questions considered per lookup
# Saved with cache eval results. Change it when the gate's rules change.
CACHE_VERSION = "c2"  # c2: judge prompt j2


class Versions(BaseModel):
    corpus_version: str
    prompt_version: str
    search_version: str
    model: str


class Cached(BaseModel):
    id: int
    question: str
    answer: dict
    similarity: float  # 1 - cosine distance; 1 = identical


def lookup(
    conn: psycopg.Connection,
    query_vector: Sequence[float],
    versions: Versions,
    k: int = CANDIDATES,
) -> list[Cached]:
    """The k most similar cached questions for these versions, most similar first."""
    rows = conn.execute(
        """
        select id, question, answer,
               1 - (question_embedding <=> %(vec)s::vector) as similarity
          from answer_cache
         where corpus_version = %(corpus_version)s
           and prompt_version = %(prompt_version)s
           and search_version = %(search_version)s
           and model = %(model)s
         order by question_embedding <=> %(vec)s::vector
         limit %(k)s
        """,
        {"vec": to_vector(query_vector), "k": k, **versions.model_dump()},
    ).fetchall()
    return [Cached(**row) for row in rows]


def store(
    conn: psycopg.Connection,
    question: str,
    query_vector: Sequence[float],
    versions: Versions,
    answer: dict,
) -> int:
    row = conn.execute(
        """
        insert into answer_cache
            (corpus_version, prompt_version, search_version, model,
             question, question_embedding, signature, answer)
        values (%(corpus_version)s, %(prompt_version)s, %(search_version)s, %(model)s,
                %(question)s, %(vec)s::vector, %(signature)s, %(answer)s)
        returning id
        """,
        {
            **versions.model_dump(),
            "question": question,
            "vec": to_vector(query_vector),
            "signature": Jsonb(signature(question).model_dump()),
            "answer": Jsonb(answer),
        },
    ).fetchone()
    return row["id"]


def record_hit(conn: psycopg.Connection, cache_id: int) -> None:
    conn.execute(
        "update answer_cache set hit_count = hit_count + 1, last_hit_at = now()"
        " where id = %s",
        [cache_id],
    )
