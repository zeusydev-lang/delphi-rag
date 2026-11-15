"""The two retrieval queries hybrid search is built from.

    with connect("query") as conn:
        by_meaning = vector_search(conn, query_vector, corpus_version="lives-v1")
        by_words = bm25_search(conn, "Utica", corpus_version="lives-v1")

Each returns Hits ranked best first. Scores are NOT comparable between the two:
  - vector_search: cosine distance, 0 = identical, lower is better
  - bm25_search:   lakebase_text's negative BM25 score, lower is better
Hybrid search will combine them by rank (Reciprocal Rank Fusion), not by score.
"""

from collections.abc import Sequence

import psycopg
from pydantic import BaseModel

from core.db import to_vector

BM25_INDEX = "chunks_tsv_bm25"


class Hit(BaseModel):
    chunk_id: int
    life: str
    chapter: int
    chunk_index: int
    text: str
    score: float  # lower is better for both searches


def vector_search(
    conn: psycopg.Connection,
    query_vector: Sequence[float],
    *,
    corpus_version: str,
    k: int = 10,
) -> list[Hit]:
    """The k chunks whose embeddings are closest to query_vector (cosine)."""
    rows = conn.execute(
        """
        select id as chunk_id, life, chapter, chunk_index, text,
               embedding <=> %(vec)s::vector as score
          from chunks
         where corpus_version = %(version)s
         order by embedding <=> %(vec)s::vector
         limit %(k)s
        """,
        {"vec": to_vector(query_vector), "version": corpus_version, "k": k},
    ).fetchall()
    return [Hit(**row) for row in rows]


def bm25_search(
    conn: psycopg.Connection,
    query: str,
    *,
    corpus_version: str,
    k: int = 10,
) -> list[Hit]:
    """The k chunks that best match the words of query (BM25, Neon only).

    Chunks that share no words with the query score exactly 0. Whether they
    come back depends on the plan (an index scan may skip them, a sequential
    scan keeps them), so they are always filtered out: a non-match must never
    earn a rank in fusion.

    The corpus_version filter runs after the index picks its candidates
    (lakebase_text's default), which is fine while the table holds one or two
    versions. Revisit with the index's prefilter option if that changes.
    """
    rows = conn.execute(
        """
        select * from (
            select id as chunk_id, life, chapter, chunk_index, text,
                   tsv <@> to_bm25query(to_tsvector('english', %(q)s), %(index)s)
                       as score
              from chunks
             where corpus_version = %(version)s
             order by score
             limit %(k)s
        ) ranked
         where score < 0
         order by score
        """,
        {"q": query, "index": BM25_INDEX, "version": corpus_version, "k": k},
    ).fetchall()
    return [Hit(**row) for row in rows]
