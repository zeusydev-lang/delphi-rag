"""Ask a question about the Lives and print the answer with its citations.

Uses dev (.env): Voyage for the question's embedding, Claude for the answer
(and for the cache judge when a cached question is a near match).

Usage:
    uv run python -m scripts.ask "How did Cato die at Utica?"
    uv run python -m scripts.ask --no-cache "How did Cato die at Utica?"
"""

import argparse
import asyncio
import sys
import textwrap

from agent.cached_answer import answer_with_cache
from agent.llm import BudgetExceeded, EmbeddingClient, LlmError
from agent.structured import StructuredOutputError


def describe_cache(reply) -> str:
    d = reply.decision
    if d is None:
        return "cache: empty or skipped"
    if d.serves_cached:
        how = "judge said same" if d.outcome == "judged_same" else "hit"
        return f"cache: {how} ({d.similarity:.3f}), matched {reply.matched_question!r}"
    if d.outcome == "miss_signature":
        return f"cache: miss, {'; '.join(d.differences)} ({d.similarity:.3f})"
    if d.outcome == "judged_different":
        return f"cache: miss, judge: {d.judge_reason} ({d.similarity:.3f})"
    return f"cache: miss, nothing close ({d.similarity:.3f})"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--no-cache", action="store_true", help="always answer fresh")
    args = parser.parse_args()

    try:
        reply = await answer_with_cache(
            args.question,
            embedder=EmbeddingClient(num_retries=0),
            use_cache=not args.no_cache,
        )
    except (LlmError, StructuredOutputError, BudgetExceeded) as err:
        print(f"Couldn't answer: {err}", file=sys.stderr)
        return 1

    result = reply.result
    print(textwrap.fill(result.answer, width=88))
    if not result.answerable:
        print("\n(The passages found don't answer this.)")
    for c in result.citations:
        print(f'\n  {c.life} s{c.section}: "{c.quote}"')
    searched = ", ".join(f"{p.life.split()[-1]} s{p.section}" for p in result.passages)
    print(f"\nSearched: {searched}")
    print(describe_cache(reply))
    print(
        f"{result.model}, prompt {result.prompt_version}, search {result.search_version}"
        f" | this request: {reply.usage.input_tokens:,} in /"
        f" {reply.usage.output_tokens:,} out tokens, ${reply.usage.cost_usd or 0:.5f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
