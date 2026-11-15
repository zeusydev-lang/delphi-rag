"""Measure retrieval on a fixed set of questions: BM25 alone, vector alone, hybrid.

Reads evals/retrieval.json, embeds every question in ONE call (input_type
"query"), runs each search method through the query login, and prints the
rates overall and per category, plus the questions hybrid search got wrong.
Saves evals/results/retrieval/<SEARCH_VERSION>.json so versions can be compared.

Metrics (each question, then averaged):
  hit@1, hit@5  an expected (life, section) is the top hit / in the top 5
  mrr           1 / rank of the first expected hit in the top 10, else 0
  life@1        the top hit is an expected section or from the Life the answer
                is about: the near-miss measure (the wrong Gracchus fails it)

Usage (against dev, .env; costs a fraction of a cent):
    uv run python -m scripts.retrieval_eval
    uv run python -m scripts.retrieval_eval --compare
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.llm import EmbeddingClient
from core.db import connect
from core.search import SEARCH_VERSION, bm25_search, hybrid_search, vector_search

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "evals" / "retrieval.json"
RESULTS_DIR = ROOT / "evals" / "results" / "retrieval"
TOP = 10  # how deep each method's ranking goes
METHODS = ("bm25", "vector", "hybrid")

Category = Literal["near_miss", "named", "paraphrase", "shared"]
Section = tuple[str, int]  # (life, section)


class Expected(BaseModel):
    life: str
    section: int = Field(ge=1)


class Question(BaseModel):
    id: str
    category: Category
    question: str = Field(min_length=5)
    expected: list[Expected] = Field(min_length=1)
    evidence: str = Field(min_length=5)
    answer_life: str | None = None
    note: str = ""

    @property
    def expected_sections(self) -> set[Section]:
        return {(e.life, e.section) for e in self.expected}

    @property
    def answer_lives(self) -> set[str]:
        return (
            {self.answer_life} if self.answer_life else {e.life for e in self.expected}
        )


class EvalSet(BaseModel):
    about: str
    corpus_version: str
    labels: str = "1"  # bumped whenever expected sections change
    questions: list[Question]


def load_eval(path: Path = EVAL_PATH) -> EvalSet:
    return EvalSet.model_validate_json(path.read_text(encoding="utf-8"))


# --- Metrics (pure) -------------------------------------------------------------


def hit_at(ranked: Sequence[Section], expected: set[Section], k: int) -> bool:
    return any(s in expected for s in ranked[:k])


def reciprocal_rank(
    ranked: Sequence[Section], expected: set[Section], k: int = TOP
) -> float:
    for rank, s in enumerate(ranked[:k], start=1):
        if s in expected:
            return 1 / rank
    return 0.0


def life_at_1(
    ranked: Sequence[Section], expected: set[Section], answer_lives: set[str]
) -> bool:
    if not ranked:
        return False
    return ranked[0] in expected or ranked[0][0] in answer_lives


def score(question: Question, ranked: Sequence[Section]) -> dict:
    exp = question.expected_sections
    return {
        "top": [list(s) for s in ranked[:5]],
        "hit@1": hit_at(ranked, exp, 1),
        "hit@5": hit_at(ranked, exp, 5),
        "mrr": reciprocal_rank(ranked, exp),
        "life@1": life_at_1(ranked, exp, question.answer_lives),
    }


METRICS = ("hit@1", "hit@5", "mrr", "life@1")


def rates(rows: Sequence[dict]) -> dict[str, float]:
    if not rows:
        return dict.fromkeys(METRICS, 0.0)
    return {m: round(sum(float(r[m]) for r in rows) / len(rows), 3) for m in METRICS}


# --- Running -------------------------------------------------------------------

Runner = Callable[[object, Question, list[float], str], list[Section]]


def _rank_bm25(conn, q, vec, version):
    return [
        (h.life, h.section)
        for h in bm25_search(conn, q.question, corpus_version=version, k=TOP)
    ]


def _rank_vector(conn, q, vec, version):
    return [
        (h.life, h.section)
        for h in vector_search(conn, vec, corpus_version=version, k=TOP)
    ]


def _rank_hybrid(conn, q, vec, version):
    return [
        (h.life, h.section)
        for h in hybrid_search(conn, q.question, vec, corpus_version=version, k=TOP)
    ]


RUNNERS: dict[str, Runner] = {
    "bm25": _rank_bm25,
    "vector": _rank_vector,
    "hybrid": _rank_hybrid,
}


async def run_eval(evalset: EvalSet, embedder: EmbeddingClient) -> dict:
    questions = evalset.questions
    embedded = await embedder.embed([q.question for q in questions], input_type="query")
    per_question: list[dict] = []
    with connect("query") as conn:
        for q, vec in zip(questions, embedded.vectors, strict=True):
            row = {"id": q.id, "category": q.category}
            for method, runner in RUNNERS.items():
                row[method] = score(q, runner(conn, q, vec, evalset.corpus_version))
            per_question.append(row)

    summary = {}
    for method in RUNNERS:
        summary[method] = {
            "overall": rates([r[method] for r in per_question]),
            "by_category": {
                cat: rates([r[method] for r in per_question if r["category"] == cat])
                for cat in sorted({r["category"] for r in per_question})
            },
        }
    return {
        "search_version": SEARCH_VERSION,
        "corpus_version": evalset.corpus_version,
        "labels": evalset.labels,
        "embedding_model": embedder.model,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": len(questions),
        "cost_usd": embedded.usage.cost_usd,
        "rates": summary,
        "results": per_question,
    }


# --- Printing --------------------------------------------------------------------


def print_table(report: dict) -> None:
    methods = list(report["rates"])
    cats = ["overall", *report["rates"][methods[0]]["by_category"]]
    for metric in METRICS:
        print(f"\n{metric:<12}" + "".join(f"{m:>10}" for m in methods))
        for cat in cats:
            vals = [
                report["rates"][m]["overall" if cat == "overall" else "by_category"]
                for m in methods
            ]
            vals = [v[metric] if cat == "overall" else v[cat][metric] for v in vals]
            print(f"  {cat:<10}" + "".join(f"{v:>10.2f}" for v in vals))


def print_failures(report: dict, evalset: EvalSet, method: str = "hybrid") -> None:
    by_id = {q.id: q for q in evalset.questions}
    failed = [
        r for r in report["results"] if not (r[method]["hit@5"] and r[method]["life@1"])
    ]
    print(f"\n{method} misses ({len(failed)} of {len(report['results'])}):")
    for r in failed:
        q = by_id[r["id"]]
        top = ", ".join(f"{life.split()[-1]} s{s}" for life, s in r[method]["top"][:3])
        print(f"  {r['id']:<5} {q.question}")
        print(
            f"        top 3: {top}   (expected: "
            + ", ".join(f"{e.life.split()[-1]} s{e.section}" for e in q.expected)
            + ")"
        )


def compare() -> int:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No saved results yet.")
        return 1
    print(
        f"{'version':<16} {'method':<14} {'hit@1':>6} {'hit@5':>6} {'mrr':>6} "
        f"{'life@1':>7} {'near-miss life@1':>17}"
    )
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        for method, r in report["rates"].items():
            o, nm = r["overall"], r["by_category"].get("near_miss", {})
            print(
                f"{report['search_version']:<16} {method:<14} {o['hit@1']:>6.2f} "
                f"{o['hit@5']:>6.2f} {o['mrr']:>6.2f} {o['life@1']:>7.2f} "
                f"{nm.get('life@1', 0):>17.2f}"
            )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--compare", action="store_true", help="table of saved versions"
    )
    args = parser.parse_args()
    if args.compare:
        return compare()

    evalset = load_eval()
    report = await run_eval(evalset, EmbeddingClient(num_retries=0))
    print(
        f"Search {report['search_version']} on {report['corpus_version']}, "
        f"{report['questions']} questions, embeddings {report['embedding_model']}"
    )
    print_table(report)
    print_failures(report, evalset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{SEARCH_VERSION}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nSaved {out.relative_to(ROOT)}  (query embeddings: ${report['cost_usd'] or 0:.6f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
