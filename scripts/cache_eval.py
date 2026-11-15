"""Measure the cache gate on labelled question pairs: does it ever serve a wrong answer?

For each pair in evals/cache_pairs.json, `a` plays the cached question and `b`
the new one. All questions are embedded in ONE Voyage call; each pair's
similarity is computed, then the real gate runs (signatures, thresholds, and
the judge model for gray-zone pairs). No database is needed.

  false hits   different questions that would be served the same answer.
               This is the number that matters; it must be 0.
  hit rate     share of true paraphrases served from the cache.
  judge calls  how many pairs needed the model, and what it said.

Usage (needs VOYAGE_API_KEY and ANTHROPIC_API_KEY; well under $0.01):
    uv run python -m scripts.cache_eval                    # evals/cache_pairs.json
    uv run python -m scripts.cache_eval --pairs heldout    # evals/cache_pairs_heldout.json
"""

import argparse
import asyncio
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from agent.cache_judge import JUDGE_PROMPT_VERSION, decide, judge_client
from agent.llm import EmbeddingClient, LlmUsage
from core.cache import CACHE_VERSION, HIT_AT, MISS_BELOW

ROOT = Path(__file__).resolve().parent.parent
PAIR_SETS = {
    "main": ROOT / "evals" / "cache_pairs.json",
    "heldout": ROOT / "evals" / "cache_pairs_heldout.json",
}
RESULTS_DIR = ROOT / "evals" / "results" / "cache"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def summarise(rows: list[dict]) -> dict:
    same = [r for r in rows if r["same"]]
    different = [r for r in rows if not r["same"]]
    return {
        "pairs": len(rows),
        "false_hits": sum(r["served"] for r in different),
        "hit_rate_on_paraphrases": round(sum(r["served"] for r in same) / len(same), 3)
        if same
        else 0.0,
        "judge_calls": sum(r["outcome"].startswith("judged") for r in rows),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
    }


async def run_eval(pair_set: str = "main") -> dict:
    data = json.loads(PAIR_SETS[pair_set].read_text(encoding="utf-8"))
    pairs = data["pairs"]
    questions = sorted({q for p in pairs for q in (p["a"], p["b"])})
    embedded = await EmbeddingClient(num_retries=0).embed(questions, input_type="query")
    vec = dict(zip(questions, embedded.vectors, strict=True))

    usage = LlmUsage(cost_usd=0.0)
    usage.add(embedded.usage)
    judge = judge_client()
    rows = []
    for p in pairs:
        sim = cosine(vec[p["a"]], vec[p["b"]])
        d = await decide(p["b"], p["a"], sim, judge=judge, usage=usage)
        rows.append(
            {
                "id": p["id"],
                "kind": p["kind"],
                "same": p["same"],
                "similarity": round(sim, 4),
                "outcome": d.outcome,
                "served": d.serves_cached,
                "differences": d.differences,
                "judge_reason": d.judge_reason,
            }
        )
    return {
        "cache_version": CACHE_VERSION,
        "hit_at": HIT_AT,
        "miss_below": MISS_BELOW,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_model": judge.model,
        "embedding_model": embedded.model,
        "pair_set": pair_set,
        "labels": data.get("labels", "1"),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cost_usd": round(usage.cost_usd or 0, 6),
        **summarise(rows),
        "results": rows,
    }


def print_report(report: dict) -> None:
    print(
        f"Cache {report['cache_version']}: hit at >= {report['hit_at']}, "
        f"miss below {report['miss_below']}, judge {report['judge_model']} "
        f"({report['judge_prompt_version']})\n"
    )
    print(f"{'pair':<6}{'kind':<12}{'label':<7}{'sim':>7}  outcome")
    for r in sorted(report["results"], key=lambda r: -r["similarity"]):
        label = "same" if r["same"] else "diff"
        wrong = (r["served"] and not r["same"]) or (r["same"] and not r["served"])
        flag = "  <-- FALSE HIT" if r["served"] and not r["same"] else ""
        flag = flag or ("  (missed hit)" if wrong else "")
        print(
            f"{r['id']:<6}{r['kind']:<12}{label:<7}{r['similarity']:>7.3f}  "
            f"{r['outcome']}{flag}"
        )
    print(
        f"\nfalse hits: {report['false_hits']}   "
        f"hit rate on paraphrases: {report['hit_rate_on_paraphrases']:.2f}   "
        f"judge calls: {report['judge_calls']}   cost: ${report['cost_usd']:.5f}"
    )
    print(f"outcomes: {report['outcomes']}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", choices=sorted(PAIR_SETS), default="main")
    args = parser.parse_args()
    report = await run_eval(args.pairs)
    print_report(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.pairs == "main" else f"-{args.pairs}"
    out = RESULTS_DIR / f"{report['cache_version']}{suffix}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")
    return 0 if report["false_hits"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
