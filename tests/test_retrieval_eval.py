"""RRF, the eval metrics, and the eval set itself. No network, no database."""

import pytest

from core.corpus import LIVES
from core.search import Hit, fuse_hits, rrf
from scripts.fetch_corpus import RAW_PATH
from scripts.retrieval_eval import (
    Question,
    hit_at,
    life_at_1,
    load_eval,
    rates,
    reciprocal_rank,
    score,
)

# --- RRF ---------------------------------------------------------------------


class TestRrf:
    def test_found_by_both_beats_found_by_one(self):
        fused = rrf({"bm25": [1, 2, 3], "vector": [4, 2, 5]})
        assert fused[0][0] == 2  # rank 2 in both beats rank 1 in one
        assert fused[0][2] == {"bm25": 2, "vector": 2}

    def test_score_formula_and_k(self):
        (chunk, score_, ranks), *_ = rrf({"a": [7]}, k=10)
        assert (chunk, ranks) == (7, {"a": 1})
        assert score_ == pytest.approx(1 / 11)

    def test_ties_break_by_best_rank_then_id(self):
        # 5 and 9 both score 1/61 + 1/63 and both have a rank 1: lower id wins.
        fused = rrf({"a": [5, 8, 9], "b": [9, 8, 5]})
        assert [c for c, _, _ in fused] == [5, 9, 8]
        assert fused[0][1] == fused[1][1]
        # 8 scores 2/62, just below 1/61 + 1/63: rank 2 twice loses to 1 and 3.
        assert fused[2][1] < fused[1][1]

    def test_better_best_rank_wins_a_tie(self):
        # 1/61 + 1/65 vs 1/62 + 1/64: not equal, so build an exact tie instead.
        fused = rrf({"a": [4, 6], "b": [6, 4]}, k=0)  # 1/1 + 1/2 each
        assert [c for c, _, _ in fused] == [4, 6]

    def test_same_input_same_order(self):
        rankings = {"bm25": [3, 1, 2], "vector": [2, 3, 4]}
        assert rrf(rankings) == rrf(rankings)

    def test_empty_lists(self):
        assert rrf({"bm25": [], "vector": []}) == []
        assert [c for c, _, _ in rrf({"bm25": [], "vector": [4, 5]})] == [4, 5]

    def test_duplicate_in_one_list_counts_once(self):
        (_, s, ranks), *_ = rrf({"a": [1, 1]})
        assert ranks == {"a": 1} and s == pytest.approx(1 / 61)


def hit(chunk_id, life="Crassus", section=1):
    return Hit(
        chunk_id=chunk_id, life=life, section=section, chunk_index=0, text="t", score=0
    )


def test_fuse_hits_keeps_each_lists_rank():
    fused = fuse_hits({"bm25": [hit(1), hit(2)], "vector": [hit(2), hit(3)]}, k=2)
    assert [h.chunk_id for h in fused] == [2, 1]
    assert (fused[0].bm25_rank, fused[0].vector_rank) == (2, 1)
    assert (fused[1].bm25_rank, fused[1].vector_rank) == (1, None)


# --- Metrics ---------------------------------------------------------------------

T, C = "Tiberius Gracchus", "Caius Gracchus"


RANKED = ((C, 1), (T, 25), (T, 3))


class TestMetrics:
    def test_hit_at(self):
        assert not hit_at(RANKED, {(T, 25)}, 1)
        assert hit_at(RANKED, {(T, 25)}, 2)
        assert not hit_at([], {(T, 25)}, 5)

    def test_reciprocal_rank(self):
        assert reciprocal_rank(RANKED, {(T, 25)}) == 0.5
        assert reciprocal_rank(RANKED, {(T, 99)}) == 0.0
        assert reciprocal_rank(RANKED, {(T, 3)}, k=2) == 0.0

    def test_life_at_1_is_the_near_miss_measure(self):
        assert not life_at_1(RANKED, {(T, 25)}, {T})  # top hit is Caius
        assert life_at_1([(T, 7)], {(T, 25)}, {T})  # right Life, other section
        # An expected retelling in another Life counts, other sections of it don't.
        assert life_at_1([(C, 19)], {(T, 25), (C, 19)}, {T})
        assert not life_at_1([(C, 1)], {(T, 25), (C, 19)}, {T})

    def test_score_and_rates(self):
        q = Question(
            id="x",
            category="near_miss",
            question="Which brother?",
            expected=[{"life": T, "section": 25}],
            evidence="clubs and staves",
        )
        row = score(q, RANKED)
        assert (row["hit@1"], row["hit@5"], row["mrr"], row["life@1"]) == (
            False,
            True,
            0.5,
            False,
        )
        assert rates([row, row]) == {
            "hit@1": 0.0,
            "hit@5": 1.0,
            "mrr": 0.5,
            "life@1": 0.0,
        }


# --- The eval set ------------------------------------------------------------------


class TestEvalSet:
    evalset = load_eval()

    def test_ids_are_unique(self):
        ids = [q.id for q in self.evalset.questions]
        assert len(ids) == len(set(ids))

    def test_every_category_is_covered(self):
        cats = {q.category for q in self.evalset.questions}
        assert cats == {"near_miss", "named", "paraphrase", "shared"}

    def test_lives_exist(self):
        lives = set(LIVES.values())
        for q in self.evalset.questions:
            assert {e.life for e in q.expected} <= lives, q.id
            assert q.answer_life is None or q.answer_life in lives, q.id

    def test_near_miss_questions_are_about_the_gracchi_or_catos(self):
        pairs = {T, C, "Cato the Elder", "Cato the Younger"}
        for q in self.evalset.questions:
            if q.category == "near_miss":
                assert q.answer_lives <= pairs, q.id

    @pytest.mark.skipif(not RAW_PATH.exists(), reason="run scripts.fetch_corpus first")
    def test_every_label_points_at_real_text(self):
        from scripts.ingest import load_chunks

        chunks = load_chunks()
        sections = {(c.life, c.section) for c in chunks}
        for q in self.evalset.questions:
            assert q.expected_sections <= sections, f"{q.id}: section doesn't exist"
            texts = [
                c.text for c in chunks if (c.life, c.section) in q.expected_sections
            ]
            assert any(q.evidence in t for t in texts), f"{q.id}: evidence not found"
