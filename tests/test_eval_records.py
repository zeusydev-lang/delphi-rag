"""Eval records (eval_runs) and the README results block."""

import json
import os
from pathlib import Path

import psycopg
import pytest

from core.db import connect
from core.eval_records import record_run, run_row
from scripts.eval_report import END, START, render_results, replace_block

RETRIEVAL = {
    "search_version": "rrf-test",
    "corpus_version": "test-v1",
    "labels": "2",
    "embedding_model": "voyage/voyage-3.5",
    "measured_at": "2026-11-15T12:00:00+00:00",
    "questions": 2,
    "cost_usd": 0.00002,
    "rates": {
        m: {
            "overall": {"hit@1": h1, "hit@5": h5, "mrr": 0.5, "life@1": 1.0},
            "by_category": {"near_miss": {"life@1": 1.0}},
        }
        for m, h1, h5 in [
            ("bm25", 0.5, 0.5),
            ("vector", 1.0, 1.0),
            ("hybrid", 0.5, 1.0),
        ]
    },
    "results": [{"id": "x"}],
}


# --- Rows (no database) ---------------------------------------------------------


def test_row_keeps_versions_and_summary_but_not_per_question_results():
    row = run_row("retrieval", RETRIEVAL, None)
    assert row["versions"] == {
        "search_version": "rrf-test",
        "corpus_version": "test-v1",
        "labels": "2",
        "embedding_model": "voyage/voyage-3.5",
    }
    assert set(row["rates"]) == {"rates", "questions"}
    assert "results" not in row["rates"]
    assert row["pair_set"] == "main"
    assert row["git_commit"] is None or len(row["git_commit"]) == 40


def test_backfilled_rows_have_no_commit_and_old_files_are_labels_v1():
    old_file = {k: v for k, v in RETRIEVAL.items() if k != "labels"}
    row = run_row("retrieval", old_file, None, with_git=False)
    assert (row["git_commit"], row["git_dirty"]) == (None, None)
    assert row["versions"]["labels"] == "1"


def test_missing_eval_login_warns_and_does_not_fail(monkeypatch, capsys):
    monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
    assert record_run("retrieval", RETRIEVAL) is False
    assert "not recorded" in capsys.readouterr().err


# --- The table (pgvector not needed; runs in CI) ------------------------------------


@pytest.fixture
def no_runs():
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        conn.execute("truncate eval_runs restart identity")
    yield


@pytest.mark.db
class TestEvalRuns:
    def test_recorded_once_even_if_imported_twice(self, no_runs):
        assert record_run("retrieval", RETRIEVAL) is True
        assert record_run("retrieval", RETRIEVAL) is False  # same run, skipped
        later = RETRIEVAL | {"measured_at": "2026-11-16T12:00:00+00:00"}
        assert record_run("retrieval", later) is True
        with connect("eval") as conn:
            rows = conn.execute(
                "select kind, versions, rates from eval_runs"
            ).fetchall()
        assert len(rows) == 2
        assert rows[0]["versions"]["search_version"] == "rrf-test"

    def test_cache_pair_sets_are_separate_runs(self, no_runs):
        cache = {
            "cache_version": "c-test",
            "measured_at": "2026-11-15T12:00:00+00:00",
            "false_hits": 0,
            "pairs": 3,
        }
        assert record_run("cache", cache | {"pair_set": "main"})
        assert record_run("cache", cache | {"pair_set": "heldout"})

    def test_eval_login_sees_only_eval_runs(self, no_runs):
        for table in ("chunks", "answer_cache", "corpus_versions"):
            denied = pytest.raises(psycopg.errors.InsufficientPrivilege)
            with denied, connect("eval") as conn:
                conn.execute(f"select 1 from {table} limit 1")

    def test_other_logins_cannot_write_eval_runs(self, no_runs):
        for role in ("query", "ingest"):
            denied = pytest.raises(psycopg.errors.InsufficientPrivilege)
            with denied, connect(role) as conn:
                conn.execute(
                    "insert into eval_runs (kind, versions, rates, measured_at)"
                    " values ('cache', '{}', '{}', now())"
                )


# --- The README block ------------------------------------------------------------------


@pytest.fixture
def results(tmp_path: Path) -> Path:
    def save(kind, name, data):
        (tmp_path / kind).mkdir(exist_ok=True)
        (tmp_path / kind / f"{name}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    save("retrieval", "rrf-test", RETRIEVAL)
    save(
        "answers",
        "a-test",
        {
            "model": "anthropic/claude-haiku-5-5",
            "prompt_version": "a-test",
            "search_version": "rrf-test",
            "measured_at": "2026-11-15T12:00:00+00:00",
            "questions": 3,
            "cost_usd": 0.001,
            "rates": {
                "answered": 1.0,
                "cites_expected": 0.5,
                "right_life": 1.0,
                "declined": 1.0,
            },
            "by_category": {"near_miss": {"right_life": 1.0}},
            "results": [
                {"category": "near_miss"},
                {"category": "named"},
                {"category": "unanswerable"},
            ],
        },
    )
    cache = {
        "cache_version": "c-test",
        "hit_at": 0.95,
        "miss_below": 0.85,
        "judge_prompt_version": "j-test",
        "pairs": 4,
        "false_hits": 0,
        "hit_rate_on_paraphrases": 0.75,
        "judge_calls": 2,
    }
    save("cache", "c-test", cache)
    save(
        "cache", "c-test-heldout", cache | {"pairs": 2, "hit_rate_on_paraphrases": 1.0}
    )
    return tmp_path


def render(results):
    return render_results(results, search="rrf-test", prompt="a-test", cache="c-test")


class TestReadmeBlock:
    def test_block_comes_from_the_files(self, results):
        block = render(results)
        assert block.startswith(START) and block.endswith(END)
        assert "| **Vector alone** | 1.00 | 1.00 |" in block  # best hit@5 in bold
        assert "| answered (2 answerable questions) | 1.00 |" in block
        assert "| declined (1 unanswerable questions) | 1.00 |" in block
        assert "| **false hits** (a wrong answer served) | **0** | **0** |" in block
        assert "| true paraphrases served from the cache | 0.75 | 1.00 |" in block

    def test_only_the_block_is_replaced(self, results):
        readme = f"# Title\n\nIntro.\n\n{START}\nold numbers\n{END}\n\nAfter.\n"
        new = replace_block(readme, render(results))
        assert new.startswith("# Title\n\nIntro.\n\n")
        assert new.endswith("\n\nAfter.\n")
        assert "old numbers" not in new
        assert replace_block(new, render(results)) == new  # stable

    def test_check_catches_a_hand_edit(self, results):
        good = replace_block(f"{START}\n{END}", render(results))
        edited = good.replace("| 0.75 |", "| 0.95 |")
        assert replace_block(good, render(results)) == good
        assert replace_block(edited, render(results)) != edited

    def test_missing_markers_are_an_error(self, results):
        with pytest.raises(ValueError, match="markers"):
            replace_block("no markers here", render(results))
