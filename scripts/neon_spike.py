"""Spike: prove one pgvector query and one lakebase_text BM25 query work.

Seeds a few near-miss passages through the ingest login, vacuums so BM25
statistics are fresh, then searches through the query login and prints the
rankings. Safe to run again (existing passages are kept).

Usage (against the Neon test branch):
    uv run --env-file .env.test python -m scripts.neon_spike --fake-embeddings
    uv run --env-file .env.test python -m scripts.neon_spike   # real Voyage embeddings
"""

import argparse
import asyncio
import os
import sys

import psycopg

from agent.llm import EmbeddingClient, LlmUsage, check_budget
from core.db import connect, to_vector
from core.search import bm25_search, vector_search
from core.settings import database_url
from scripts.spike_data import PASSAGES, hashed_vector

VECTOR_QUESTION = "Which brother was killed on the Capitol by the senators?"
BM25_QUESTIONS = ["Utica", "Opimius Aventine"]


async def embed(texts: list[str], kind: str, fake: bool, usage: LlmUsage):
    if fake:
        return [hashed_vector(t) for t in texts]
    result = await EmbeddingClient().embed(texts, input_type=kind)
    usage.add(result.usage)
    check_budget(usage, float(os.environ.get("DAILY_SPEND_CAP_USD", "0.25")))
    return result.vectors


def seed(version: str, vectors: list[list[float]]) -> int:
    with connect("ingest") as conn:
        added = 0
        for (life, text), vec in zip(PASSAGES, vectors, strict=True):
            cur = conn.execute(
                """
                insert into chunks
                    (corpus_version, life, section, chunk_index, text, embedding)
                values (%s, %s, 1, 0, %s, %s::vector)
                on conflict do nothing
                """,
                [version, life, text, to_vector(vec)],
            )
            added += cur.rowcount
    # VACUUM can't run inside a transaction, so this uses its own connection.
    with psycopg.connect(database_url("ingest"), autocommit=True) as conn:
        conn.execute("vacuum (analyze) chunks")
    return added


def show(title: str, hits) -> None:
    print(f"\n{title}")
    for rank, hit in enumerate(hits, 1):
        print(f"  {rank}. {hit.life:<18} score {hit.score:+.4f}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="hashed word vectors instead of Voyage (no key, no cost)",
    )
    args = parser.parse_args()
    fake = args.fake_embeddings
    version = "spike-fake" if fake else "spike-voyage"
    usage = LlmUsage()

    vectors = await embed([t for _, t in PASSAGES], "document", fake, usage)
    print(f"Seeded {seed(version, vectors)} new passage(s) as {version!r}")

    (question_vec,) = await embed([VECTOR_QUESTION], "query", fake, usage)
    with connect("query") as conn:
        hits = vector_search(conn, question_vec, corpus_version=version, k=3)
    show(f"Vector: {VECTOR_QUESTION!r}", hits)

    for q in BM25_QUESTIONS:
        try:
            with connect("query") as conn:
                hits = bm25_search(conn, q, corpus_version=version, k=3)
        except psycopg.Error as err:
            # The spike's open question: can the non-owner login run BM25?
            print(f"\nBM25: {q!r} FAILED: {type(err).__name__}: {err}")
            return 1
        show(f"BM25: {q!r}", hits)

    if not fake:
        print(
            f"\nEmbedding spend: {usage.input_tokens} tokens, ${usage.cost_usd or 0:.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
