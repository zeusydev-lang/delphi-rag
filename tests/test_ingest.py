"""Loading chunks through the ingest login, end to end, with fake embeddings."""

import asyncio
import os

import psycopg
import pytest

from agent.llm import EmbeddingClient, LlmError
from core.corpus import CHUNKER, Chunk
from core.db import connect
from core.search import bm25_search
from scripts.ingest import VersionMismatch, ingest
from tests.fake_anthropic import FakeEmbeddings

VERSION = "test-ingest-v1"
SHA = "a" * 64

CHUNKS = [
    Chunk(
        life="Tiberius Gracchus",
        section=1,
        chunk_index=0,
        text="He proposed a law to give public land to the poor citizens of Rome.",
    ),
    Chunk(
        life="Tiberius Gracchus",
        section=2,
        chunk_index=0,
        text="His fellow tribune Octavius opposed the law and was removed.",
    ),
    Chunk(
        life="Caius Gracchus",
        section=1,
        chunk_index=0,
        text="The younger brother passed a grain law and founded colonies.",
    ),
    Chunk(
        life="Crassus",
        section=1,
        chunk_index=0,
        text="The richest man in Rome defeated the revolt of Spartacus.",
    ),
]


@pytest.fixture(scope="module")
def fake():
    server = FakeEmbeddings(dims=1024)  # matches vector(1024)
    yield server
    server.close()


@pytest.fixture
def embedder(fake, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test-not-real")
    fake.requests.clear()
    fake.fail_next.clear()
    return EmbeddingClient("voyage/voyage-3.5", api_base=fake.url, num_retries=0)


@pytest.fixture
def empty():
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        conn.execute("truncate chunks, corpus_versions restart identity")
    yield


def run_ingest(chunks, embedder, **kw):
    params = {"version": VERSION, "source_sha256": SHA, "cap_usd": 0.25}
    return asyncio.run(ingest(chunks, embedder=embedder, **(params | kw)))


@pytest.mark.db
class TestIngest:
    def test_loads_every_chunk_and_records_the_version(self, empty, embedder, fake):
        report = run_ingest(CHUNKS, embedder, batch_size=3)
        assert (report.new_chunks, report.total_chunks) == (4, 4)
        assert len(fake.requests) == 2  # two batches
        assert report.usage.input_tokens > 0

        with connect("query") as conn:
            rows = conn.execute(
                "select life, section, chunk_index from chunks"
                " where corpus_version = %s order by id",
                [VERSION],
            ).fetchall()
            version = conn.execute(
                "select * from corpus_versions where version = %s", [VERSION]
            ).fetchone()
        assert [(r["life"], r["section"]) for r in rows] == [
            (c.life, c.section) for c in CHUNKS
        ]
        assert version["source_sha256"] == SHA
        assert version["chunker"] == CHUNKER
        assert version["chunk_count"] == 4

    def test_embeds_the_life_name_with_the_text(self, empty, embedder, fake):
        run_ingest(CHUNKS[:1], embedder)
        (sent,) = fake.requests[0]["input"]
        assert sent.startswith("Plutarch, Life of Tiberius Gracchus, section 1.")
        assert fake.requests[0]["input_type"] == "document"

    def test_second_run_adds_nothing_and_costs_nothing(self, empty, embedder, fake):
        run_ingest(CHUNKS, embedder)
        fake.requests.clear()
        report = run_ingest(CHUNKS, embedder)
        assert report.new_chunks == 0 and report.total_chunks == 4
        assert fake.requests == []
        assert report.usage.cost_usd == 0

    def test_interrupted_run_resumes(self, empty, embedder, fake):
        run_ingest(CHUNKS[:2], embedder)
        fake.requests.clear()
        report = run_ingest(CHUNKS, embedder)
        assert report.new_chunks == 2
        assert [len(r["input"]) for r in fake.requests] == [2]

    def test_same_version_from_a_different_source_is_refused(self, empty, embedder):
        run_ingest(CHUNKS, embedder)
        with pytest.raises(VersionMismatch, match="new CORPUS_VERSION"):
            run_ingest(CHUNKS, embedder, source_sha256="b" * 64)

    def test_rate_limit_is_waited_out(self, empty, embedder, fake):
        fake.fail_next = [429, 429]
        report = run_ingest(CHUNKS, embedder, rate_limit_wait_s=0)
        assert report.new_chunks == 4
        assert len(fake.requests) == 3  # two refusals, then the real call

    def test_other_errors_still_stop_the_run(self, empty, embedder, fake):
        fake.fail_next = [401]
        with pytest.raises(LlmError, match="401"):
            run_ingest(CHUNKS, embedder, rate_limit_wait_s=0)

    def test_tsv_includes_the_life_name(self, empty, embedder):
        run_ingest(CHUNKS[:1], embedder)
        with connect("query") as conn:
            tsv = conn.execute("select tsv::text as t from chunks").fetchone()["t"]
        assert "'tiberius'" in tsv and "'gracchus'" in tsv


@pytest.mark.neon
def test_bm25_finds_a_life_by_name_its_text_never_uses(empty, embedder):
    run_ingest(CHUNKS, embedder)
    assert "Tiberius" not in CHUNKS[1].text
    with connect("query") as conn:
        hits = bm25_search(conn, "Tiberius Octavius", corpus_version=VERSION, k=3)
    assert hits[0].life == "Tiberius Gracchus" and hits[0].section == 2
