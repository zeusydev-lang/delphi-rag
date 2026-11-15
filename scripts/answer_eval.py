"""Measure answers: are citations real, from the right sections, from the right Life?

Runs the 33 retrieval questions (evals/retrieval.json) and the questions the
corpus can't answer (evals/unanswerable.json) through answer_question, and
checks each answer in code (no judge model):

  answered       the model gave an answer (answerable=true)
  cites_expected at least one citation is from an expected section
  right_life     the first (main) citation is from the answer's Life or an
                 expected section: the near-miss measure. Later citations may
                 come from other Lives that retell the event or tell the
                 brothers apart; that's fine and isn't counted against it.
  declined       for unanswerable questions: answerable=false
  calls          answer-model calls (2 means the citation check sent it back once)

Citations are always real quotes: answer_question refuses anything else. What
this can't tell is whether the answer is *correct*; that needs a judge.

Usage (dev, .env; about $0.02 a run):
    uv run python -m scripts.answer_eval
    uv run python -m scripts.answer_eval --compare
    uv run python -m scripts.answer_eval --rescore   # re-check saved answers, no API calls
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from agent.answer import AnswerResult, answer_question
from agent.llm import EmbeddingClient, LlmClient, LlmUsage, check_budget
from agent.prompts import PROMPT_VERSION
from agent.structured import StructuredOutputError
from core.search import SEARCH_VERSION
from scripts.retrieval_eval import Question, load_eval

ROOT = Path(__file__).resolve().parent.parent
UNANSWERABLE_PATH = ROOT / "evals" / "unanswerable.json"
RESULTS_DIR = ROOT / "evals" / "results" / "answers"
RUN_CAP_USD = 0.25  # the whole eval run stops past this


class Unanswerable(BaseModel):
    id: str
    question: str


def load_unanswerable(path: Path = UNANSWERABLE_PATH) -> list[Unanswerable]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Unanswerable(**q) for q in data["questions"]]


class CountingClient(LlmClient):
    """LlmClient that counts calls, to see how often the citation check sends one back."""

    calls = 0

    async def generate(self, request):
        self.calls += 1
        return await super().generate(request)


# --- Checks (pure) ---------------------------------------------------------------


def check_citations(q: Question, answered: bool, cited: list[tuple[str, int]]) -> dict:
    expected = q.expected_sections
    main = cited[0] if cited else None
    return {
        "answered": answered,
        "cites_expected": any(s in expected for s in cited),
        "right_life": main is not None
        and (main in expected or main[0] in q.answer_lives),
    }


def check_answer(q: Question, result: AnswerResult) -> dict:
    cited = [(c.life, c.section) for c in result.citations]
    return check_citations(q, result.answerable, cited)


def check_decline(result: AnswerResult) -> dict:
    return {"declined": not result.answerable}


def rate(rows: list[dict], key: str) -> float:
    return round(sum(bool(r[key]) for r in rows) / len(rows), 3) if rows else 0.0


# --- Running ---------------------------------------------------------------------


async def run_eval() -> dict:
    evalset = load_eval()
    unanswerable = load_unanswerable()
    texts = [q.question for q in evalset.questions] + [u.question for u in unanswerable]

    embedded = await EmbeddingClient(num_retries=0).embed(texts, input_type="query")
    total = LlmUsage(cost_usd=0.0)
    total.add(embedded.usage)

    rows: list[dict] = []
    items = [*evalset.questions, *unanswerable]
    for item, vec in zip(items, embedded.vectors, strict=True):
        llm = CountingClient()
        started = time.perf_counter()
        row: dict = {
            "id": item.id,
            "category": getattr(item, "category", "unanswerable"),
        }
        try:
            result = await answer_question(
                item.question, llm=llm, query_vector=vec, cap_usd=RUN_CAP_USD
            )
        except StructuredOutputError as err:
            row |= {"error": str(err)[:200], "calls": llm.calls}
            rows.append(row)
            print(f"  {item.id:<5} FAILED after {llm.calls} calls")
            continue
        total.add(result.usage)
        check_budget(total, RUN_CAP_USD)
        checks = (
            check_answer(item, result)
            if isinstance(item, Question)
            else check_decline(result)
        )
        row |= checks | {
            "calls": llm.calls,
            "seconds": round(time.perf_counter() - started, 2),
            "cost_usd": result.usage.cost_usd,
            "answer": result.answer,
            "citations": [[c.life, c.section, c.quote] for c in result.citations],
        }
        rows.append(row)
        flags = " ".join(k for k, v in checks.items() if not v)
        print(f"  {item.id:<5} {'ok' if not flags else 'MISS ' + flags}")

    return {
        "prompt_version": PROMPT_VERSION,
        "search_version": SEARCH_VERSION,
        "corpus_version": evalset.corpus_version,
        "labels": evalset.labels,
        "model": CountingClient().model,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": len(rows),
        "errors": sum("error" in r for r in rows),
        "cost_usd": round(total.cost_usd or 0, 6),
        **summarise(rows),
        "results": rows,
    }


def summarise(rows: list[dict]) -> dict:
    answerable = [
        r for r in rows if r["category"] != "unanswerable" and "error" not in r
    ]
    declinable = [
        r for r in rows if r["category"] == "unanswerable" and "error" not in r
    ]
    ok = [r for r in rows if "error" not in r]
    by_category = {}
    for cat in sorted({r["category"] for r in answerable}):
        mine = [r for r in answerable if r["category"] == cat]
        by_category[cat] = {
            k: rate(mine, k) for k in ("answered", "cites_expected", "right_life")
        }
    return {
        "rates": {
            "answered": rate(answerable, "answered"),
            "cites_expected": rate(answerable, "cites_expected"),
            "right_life": rate(answerable, "right_life"),
            "declined": rate(declinable, "declined"),
            "sent_back_once": rate([{"x": r["calls"] > 1} for r in ok], "x"),
            "avg_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 2)
            if ok
            else 0,
        },
        "by_category": by_category,
    }


def rescore(path: Path) -> dict:
    """Re-check saved answers against the current labels. No API calls."""
    report = json.loads(path.read_text(encoding="utf-8"))
    evalset = load_eval()
    by_id = {q.id: q for q in evalset.questions}
    for row in report["results"]:
        q = by_id.get(row["id"])
        if q is None or "error" in row:
            continue
        cited = [(life, section) for life, section, _ in row["citations"]]
        row |= check_citations(q, row["answered"], cited)
    report |= summarise(report["results"])
    report["labels"] = evalset.labels
    report["rescored_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return report


def compare() -> int:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("No saved results yet.")
        return 1
    keys = ("answered", "cites_expected", "right_life", "declined", "sent_back_once")
    print(
        f"{'prompt':<8} {'search':<10}"
        + "".join(f"{k:>16}" for k in keys)
        + f"{'cost':>9}"
    )
    for path in files:
        r = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{r['prompt_version']:<8} {r['search_version']:<10}"
            + "".join(f"{r['rates'][k]:>16.2f}" for k in keys)
            + f"{r['cost_usd']:>9.4f}"
        )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--compare", action="store_true", help="table of saved versions"
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="re-check the saved results for this prompt version; no API calls",
    )
    parser.add_argument(
        "--no-record", action="store_true", help="don't add the run to eval_runs"
    )
    args = parser.parse_args()
    if args.compare:
        return compare()

    out = RESULTS_DIR / f"{PROMPT_VERSION}.json"
    report = rescore(out) if args.rescore else await run_eval()
    print(
        f"\nPrompt {report['prompt_version']}, search {report['search_version']}, "
        f"{report['model']}: {report['questions']} questions, {report['errors']} errors"
    )
    for key, value in report["rates"].items():
        print(f"  {key:<16} {value}")
    print("\n  by category (answered / cites_expected / right_life):")
    for cat, r in report["by_category"].items():
        print(
            f"    {cat:<11} {r['answered']:.2f} / {r['cites_expected']:.2f} "
            f"/ {r['right_life']:.2f}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}  (total ${report['cost_usd']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
