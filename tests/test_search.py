"""The chunks table, its permissions, and the two search queries.

`db` tests run anywhere with pgvector (CI's Postgres, the Neon test branch).
`neon` tests need lakebase_text, so they run only on the Neon test branch.
"""

import os

import psycopg
import pytest

from core.db import connect, to_vector
from core.search import bm25_search, hybrid_search, vector_search
from scripts.spike_data import PASSAGES, hashed_vector

VERSION = "test-v1"


@pytest.fixture
def seeded():
    """An empty chunks table, then the spike passages loaded through ingest."""
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        conn.execute("truncate chunks restart identity")
    with connect("ingest") as conn:
        for i, (life, text) in enumerate(PASSAGES):
            conn.execute(
                "insert into chunks"
                " (corpus_version, life, section, chunk_index, text, embedding)"
                " values (%s, %s, 1, %s, %s, %s::vector)",
                [VERSION, life, i, text, to_vector(hashed_vector(text))],
            )
    with psycopg.connect(os.environ["INGEST_DATABASE_URL"], autocommit=True) as conn:
        conn.execute("vacuum (analyze) chunks")  # ingest has MAINTAIN
    yield


# --- Vector search (pgvector) -------------------------------------------------


@pytest.mark.db
class TestVectorSearch:
    def test_closest_passage_first(self, seeded):
        question = hashed_vector("Who died at Utica rather than accept a pardon?")
        with connect("query") as conn:
            hits = vector_search(conn, question, corpus_version=VERSION, k=3)
        assert hits[0].life == "Cato the Younger"
        assert [h.score for h in hits] == sorted(h.score for h in hits)

    def test_exact_text_has_distance_zero(self, seeded):
        life, text = PASSAGES[0]
        with connect("query") as conn:
            (hit,) = vector_search(
                conn, hashed_vector(text), corpus_version=VERSION, k=1
            )
        assert hit.life == life
        assert hit.score == pytest.approx(0, abs=1e-6)

    def test_other_corpus_versions_are_ignored(self, seeded):
        with connect("query") as conn:
            hits = vector_search(
                conn, hashed_vector("Crassus"), corpus_version="other", k=3
            )
        assert hits == []


# --- Permissions ----------------------------------------------------------------


@pytest.mark.db
class TestChunkPermissions:
    def test_tsv_is_generated(self, seeded):
        with connect("query") as conn:
            row = conn.execute(
                "select tsv::text as tsv from chunks where life = 'Crassus'"
            ).fetchone()
        assert "'spartacus'" in row["tsv"]

    def test_ingest_cannot_change_or_delete(self, seeded):
        denied = pytest.raises(psycopg.errors.InsufficientPrivilege)
        with denied, connect("ingest") as conn:
            conn.execute("update chunks set text = 'changed'")
        denied = pytest.raises(psycopg.errors.InsufficientPrivilege)
        with denied, connect("ingest") as conn:
            conn.execute("delete from chunks")

    def test_query_cannot_write(self, seeded):
        denied = pytest.raises(psycopg.errors.InsufficientPrivilege)
        with denied, connect("query") as conn:
            conn.execute(
                "insert into chunks"
                " (corpus_version, life, section, chunk_index, text, embedding)"
                " values ('x', 'x', 1, 0, 'x', %s::vector)",
                [to_vector(hashed_vector("x"))],
            )

    def test_reload_is_safe(self, seeded):
        life, text = PASSAGES[0]
        with connect("ingest") as conn:
            cur = conn.execute(
                "insert into chunks"
                " (corpus_version, life, section, chunk_index, text, embedding)"
                " values (%s, %s, 1, 0, %s, %s::vector) on conflict do nothing",
                [VERSION, life, text, to_vector(hashed_vector(text))],
            )
        assert cur.rowcount == 0


# --- BM25 search (lakebase_text, Neon only) -----------------------------------


@pytest.mark.neon
class TestBm25Search:
    def test_rare_word_finds_its_passage(self, seeded):
        with connect("query") as conn:
            hits = bm25_search(conn, "Utica", corpus_version=VERSION, k=3)
        assert hits[0].life == "Cato the Younger"

    def test_near_miss_brothers_are_told_apart(self, seeded):
        with connect("query") as conn:
            hits = bm25_search(conn, "Opimius Aventine", corpus_version=VERSION, k=3)
        assert hits[0].life == "Caius Gracchus"

    def test_non_matches_are_not_returned(self, seeded):
        # Only one passage mentions Utica; the others score 0 and must not
        # come back to earn a rank in fusion.
        with connect("query") as conn:
            hits = bm25_search(conn, "Utica", corpus_version=VERSION, k=5)
        assert [h.life for h in hits] == ["Cato the Younger"]

    def test_no_matching_words_returns_nothing(self, seeded):
        with connect("query") as conn:
            assert bm25_search(conn, "Hannibal", corpus_version=VERSION) == []

    def test_scores_are_negative_and_ascending(self, seeded):
        with connect("query") as conn:
            hits = bm25_search(conn, "Caesar", corpus_version=VERSION, k=5)
        scores = [h.score for h in hits]
        assert scores and scores[0] < 0
        assert scores == sorted(scores)


# --- Hybrid search (needs BM25, so Neon only) ---------------------------------


@pytest.mark.neon
class TestHybridSearch:
    def test_found_by_both_searches_ranks_first(self, seeded):
        question = "Who died at Utica rather than accept a pardon?"
        with connect("query") as conn:
            hits = hybrid_search(
                conn, question, hashed_vector(question), corpus_version=VERSION, k=3
            )
        top = hits[0]
        assert top.life == "Cato the Younger"
        assert top.bm25_rank == 1 and top.vector_rank == 1
        assert hits == sorted(hits, key=lambda h: -h.rrf_score)

    def test_a_chunk_only_one_search_finds_is_still_returned(self, seeded):
        # No passage contains "Hannibal": BM25 finds nothing, vector still ranks.
        question = "Hannibal"
        with connect("query") as conn:
            hits = hybrid_search(
                conn,
                question,
                hashed_vector("Spartacus revolt"),
                corpus_version=VERSION,
            )
        assert hits and all(h.bm25_rank is None for h in hits)
        assert hits[0].life == "Crassus"
