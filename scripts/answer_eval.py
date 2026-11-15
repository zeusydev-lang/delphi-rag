"""Measure answers: are citations real, from the right sections, from the right Life?

Runs the 33 retrieval questions (evals/retrieval.json) and the questions the
corpus can't answer (evals/unanswerable.json) through answer_question, and
checks each answer in code (no judge model):

  answered       the model gave an answer (answerable=true)
  cites_expected at least one citation is from an expected section
  right_life     every citation is from the answer's Life or an expected section
  declined       for unanswerable questions: answerable=false
  calls          answer-model calls (2 means the citation check sent it back once)

Citations are always real quotes: answer_question refuses anything else. What
this can't tell is whether the answer is *correct*; that needs a judge.

Usage (dev, .env; about $0.02 a run):
    uv run python -m scripts.answer_eval
    uv run python -m scripts.answer_eval --compare
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


def check_answer(q: Question, result: AnswerResult) -> dict:
    cited = [(c.life, c.section) for c in result.citations]
    expected = q.expected_sections
    return {
        "answered": result.answerable,
        "cites_expected": any(s in expected for s in cited),
        "right_life": all(s in expected or s[0] in q.answer_lives for s in cited),
    }


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
        "prompt_version": PROMPT_VERSION,
        "search_version": SEARCH_VERSION,
        "corpus_version": evalset.corpus_version,
        "model": CountingClient().model,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": len(rows),
        "errors": sum("error" in r for r in rows),
        "cost_usd": round(total.cost_usd or 0, 6),
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
        "results": rows,
    }


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
    args = parser.parse_args()
    if args.compare:
        return compare()

    report = await run_eval()
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
    out = RESULTS_DIR / f"{PROMPT_VERSION}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}  (total ${report['cost_usd']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
