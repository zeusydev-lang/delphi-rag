"""Retrieval: BM25, vector, and the two fused (hybrid).

    with connect("query") as conn:
        by_meaning = vector_search(conn, query_vector, corpus_version="lives-v1")
        by_words = bm25_search(conn, "Utica", corpus_version="lives-v1")
        both = hybrid_search(conn, question, query_vector, corpus_version="lives-v1")

vector_search and bm25_search return Hits ranked best first. Their scores are
NOT comparable:
  - vector_search: cosine distance, 0 = identical, lower is better
  - bm25_search:   lakebase_text's negative BM25 score, lower is better
so hybrid_search combines them by rank, with Reciprocal Rank Fusion (RRF):
each chunk scores sum(1 / (RRF_K + rank)) over the lists it appears in.
"""

from collections.abc import Mapping, Sequence

import psycopg
from pydantic import BaseModel

from core.db import to_vector

BM25_INDEX = "chunks_tsv_bm25"

# Saved with every eval result. Change it whenever retrieval behaves differently.
SEARCH_VERSION = "rrf-v1"
RRF_K = 60  # the usual constant: damps the gap between rank 1 and rank 2
CANDIDATES = 20  # taken from each search before fusing


class Hit(BaseModel):
    chunk_id: int
    life: str
    section: int  # paragraph section within the Life, from 1
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
        select id as chunk_id, life, section, chunk_index, text,
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
            select id as chunk_id, life, section, chunk_index, text,
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


# --- Fusion -------------------------------------------------------------------


def rrf(
    rankings: Mapping[str, Sequence[int]], k: int = RRF_K
) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse ranked lists of chunk ids, best first.

    Returns (chunk_id, score, {list name: rank}) for every chunk in any list,
    highest score first. Ranks count from 1. Ties go to the chunk with the
    better single rank, then the lower id, so the order never varies.
    """
    ranks: dict[int, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            ranks.setdefault(chunk_id, {}).setdefault(name, rank)  # first one counts
    fused = [
        (chunk_id, sum(1 / (k + r) for r in by_list.values()), by_list)
        for chunk_id, by_list in ranks.items()
    ]
    fused.sort(key=lambda f: (-f[1], min(f[2].values()), f[0]))
    return fused


class FusedHit(BaseModel):
    chunk_id: int
    life: str
    section: int
    chunk_index: int
    text: str
    rrf_score: float  # higher is better (unlike Hit.score)
    ranks: dict[str, int]  # rank in each list that found it, e.g. {"bm25": 3}

    @property
    def bm25_rank(self) -> int | None:
        return self.ranks.get("bm25")

    @property
    def vector_rank(self) -> int | None:
        return self.ranks.get("vector")


def fuse_hits(
    lists: Mapping[str, Sequence[Hit]], *, k: int = 10, rrf_k: int = RRF_K
) -> list[FusedHit]:
    """RRF over lists of Hits; the top k as FusedHits."""
    by_id = {hit.chunk_id: hit for hits in lists.values() for hit in hits}
    fused = rrf(
        {name: [h.chunk_id for h in hits] for name, hits in lists.items()}, rrf_k
    )
    return [
        FusedHit(
            **by_id[chunk_id].model_dump(exclude={"score"}),
            rrf_score=score,
            ranks=ranks,
        )
        for chunk_id, score, ranks in fused[:k]
    ]


def hybrid_search(
    conn: psycopg.Connection,
    question: str,
    query_vector: Sequence[float],
    *,
    corpus_version: str,
    k: int = 10,
    candidates: int = CANDIDATES,
) -> list[FusedHit]:
    """BM25 and vector search, top `candidates` each, fused with RRF."""
    lists = {
        "bm25": bm25_search(
            conn, question, corpus_version=corpus_version, k=candidates
        ),
        "vector": vector_search(
            conn, query_vector, corpus_version=corpus_version, k=candidates
        ),
    }
    return fuse_hits(lists, k=k)
