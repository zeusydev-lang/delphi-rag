"""Load the Lives into the chunks table, through the ingest login.

Parses data/raw/pg674.txt (run scripts.fetch_corpus first), chunks it, embeds
only the chunks the database doesn't have yet, and inserts them in batches.
A second run adds nothing and costs nothing; a run that stops halfway resumes.

Usage:
    uv run python -m scripts.ingest --dry-run          # chunk and report, no API, no database
    uv run python -m scripts.ingest --fake-embeddings  # free; loads as '<version>-fake'
    uv run python -m scripts.ingest                    # Voyage embeddings (costs money)
"""

import argparse
import asyncio
import os
import sys
from collections import Counter
from collections.abc import Sequence

import psycopg
from pydantic import BaseModel

from agent.llm import EmbeddingClient, LlmError, LlmUsage, check_budget
from core.corpus import (
    CHUNKER,
    CORPUS_VERSION,
    LIVES,
    SOURCE_SHA256,
    SOURCE_URL,
    Chunk,
    chunk_corpus,
    embedding_input,
    split_lives,
    strip_gutenberg,
)
from core.db import connect, to_vector
from core.settings import database_url
from scripts.fetch_corpus import RAW_PATH, sha256
from scripts.spike_data import hashed_vector

BATCH_SIZE = 64
RATE_LIMIT_WAIT_S = 65  # rate limits are per minute
RATE_LIMIT_RETRIES = 5


class VersionMismatch(RuntimeError):
    """The corpus version already exists with a different source or chunker."""


class IngestReport(BaseModel):
    version: str
    total_chunks: int
    new_chunks: int
    usage: LlmUsage


def load_chunks(path=RAW_PATH) -> list[Chunk]:
    return chunk_corpus(split_lives(strip_gutenberg(path.read_text(encoding="utf-8"))))


def print_chunk_report(chunks: Sequence[Chunk]) -> None:
    print(
        f"{'Life':<18} {'chunks':>6} {'sections':>8} {'words':>7} {'min':>5} {'max':>5}"
    )
    for life in LIVES.values():
        mine = [c for c in chunks if c.life == life]
        words = [c.words for c in mine]
        print(
            f"{life:<18} {len(mine):>6} {max(c.section for c in mine):>8} "
            f"{sum(words):>7} {min(words):>5} {max(words):>5}"
        )
    print(f"{'Total':<18} {len(chunks):>6} {'':>8} {sum(c.words for c in chunks):>7}")


def _check_version(conn: psycopg.Connection, version: str, source_sha256: str) -> None:
    row = conn.execute(
        "select source_sha256, chunker from corpus_versions where version = %s",
        [version],
    ).fetchone()
    if row and (row["source_sha256"], row["chunker"]) != (source_sha256, CHUNKER):
        raise VersionMismatch(
            f"{version!r} was built from source {row['source_sha256'][:12]}… with "
            f"chunker {row['chunker']!r}; this run has {source_sha256[:12]}… and "
            f"{CHUNKER!r}. Use a new CORPUS_VERSION."
        )


def _existing_keys(conn: psycopg.Connection, version: str) -> set[tuple]:
    rows = conn.execute(
        "select life, section, chunk_index from chunks where corpus_version = %s",
        [version],
    ).fetchall()
    return {(r["life"], r["section"], r["chunk_index"]) for r in rows}


async def _embed_with_backoff(embedder: EmbeddingClient, texts, wait_s: float):
    """Embed, waiting out rate limits (HTTP 429) instead of failing the run."""
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            return await embedder.embed(texts, input_type="document")
        except LlmError as err:
            if err.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
                raise
            print(f"  rate limited; waiting {wait_s:.0f}s (attempt {attempt})")
            await asyncio.sleep(wait_s)
    raise AssertionError("unreachable")


async def ingest(
    chunks: Sequence[Chunk],
    *,
    version: str,
    source_sha256: str,
    embedder: EmbeddingClient | None,  # None: free hashed vectors
    cap_usd: float,
    batch_size: int = BATCH_SIZE,
    pause_s: float = 0.0,  # wait between embedding calls (low rate limits)
    rate_limit_wait_s: float = RATE_LIMIT_WAIT_S,
) -> IngestReport:
    """Embed and insert the chunks the database doesn't have yet."""
    with connect("ingest") as conn:
        _check_version(conn, version, source_sha256)
        existing = _existing_keys(conn, version)
    missing = [c for c in chunks if (c.life, c.section, c.chunk_index) not in existing]

    usage = LlmUsage(cost_usd=0.0)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        texts = [embedding_input(c) for c in batch]
        if embedder is None:
            vectors = [hashed_vector(t) for t in texts]
        else:
            if start and pause_s:
                await asyncio.sleep(pause_s)
            result = await _embed_with_backoff(embedder, texts, rate_limit_wait_s)
            vectors = result.vectors
            usage.add(result.usage)
            check_budget(usage, cap_usd)
        # One transaction per batch: a failure later keeps the batches already
        # paid for, and the next run picks up from there.
        with connect("ingest") as conn, conn.cursor() as cur:
            cur.executemany(
                """
                insert into chunks
                    (corpus_version, life, section, chunk_index, text, embedding)
                values (%s, %s, %s, %s, %s, %s::vector)
                on conflict do nothing
                """,
                [
                    (version, c.life, c.section, c.chunk_index, c.text, to_vector(v))
                    for c, v in zip(batch, vectors, strict=True)
                ],
            )
        print(f"  inserted {start + len(batch)}/{len(missing)}")

    with connect("ingest") as conn:
        count = conn.execute(
            "select count(*) as n from chunks where corpus_version = %s", [version]
        ).fetchone()["n"]
        conn.execute(
            """
            insert into corpus_versions
                (version, source_url, source_sha256, chunker, lives, chunk_count)
            values (%s, %s, %s, %s, %s, %s)
            on conflict (version) do nothing
            """,
            [version, SOURCE_URL, source_sha256, CHUNKER, list(LIVES.values()), count],
        )
    if missing:
        # BM25 statistics refresh at VACUUM; it can't run inside a transaction.
        with psycopg.connect(database_url("ingest"), autocommit=True) as conn:
            conn.execute("vacuum (analyze) chunks")
    return IngestReport(
        version=version, total_chunks=count, new_chunks=len(missing), usage=usage
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="chunk and report only")
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help=f"hashed word vectors, loaded as '{CORPUS_VERSION}-fake' (free)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"chunks per embedding call (default {BATCH_SIZE})",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="seconds between embedding calls, for low rate limits",
    )
    args = parser.parse_args()

    if not RAW_PATH.exists():
        print(f"{RAW_PATH} not found. Run: uv run python -m scripts.fetch_corpus")
        return 1
    digest = sha256(RAW_PATH)
    if digest != SOURCE_SHA256:
        print("Source file doesn't match the pinned hash. Run scripts.fetch_corpus.")
        return 1

    chunks = load_chunks()
    print_chunk_report(chunks)
    per_life = Counter(c.life for c in chunks)
    if set(per_life) != set(LIVES.values()):
        print("Some Lives produced no chunks.", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    version = f"{CORPUS_VERSION}-fake" if args.fake_embeddings else CORPUS_VERSION
    print(f"\nLoading {version!r} ...")
    report = await ingest(
        chunks,
        version=version,
        source_sha256=digest,
        # No quick retries inside litellm: they spend the per-minute request
        # allowance at once. _embed_with_backoff waits out 429s instead.
        embedder=None if args.fake_embeddings else EmbeddingClient(num_retries=0),
        cap_usd=float(os.environ.get("DAILY_SPEND_CAP_USD", "0.25")),
        batch_size=args.batch_size,
        pause_s=args.pause,
    )
    print(
        f"\n{report.new_chunks} new chunk(s); {report.total_chunks} in {version!r}. "
        f"Embedding spend: {report.usage.input_tokens:,} tokens, "
        f"${report.usage.cost_usd or 0:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
