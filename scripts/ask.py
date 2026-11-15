"""Ask a question about the Lives and print the answer with its citations.

Uses dev (.env): Voyage for the question's embedding, Claude for the answer.

Usage:
    uv run python -m scripts.ask "How did Cato die at Utica?"
"""

import argparse
import asyncio
import sys
import textwrap

from agent.answer import answer_question
from agent.llm import BudgetExceeded, EmbeddingClient, LlmError
from agent.structured import StructuredOutputError


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    args = parser.parse_args()

    try:
        result = await answer_question(
            args.question, embedder=EmbeddingClient(num_retries=0)
        )
    except (LlmError, StructuredOutputError, BudgetExceeded) as err:
        print(f"Couldn't answer: {err}", file=sys.stderr)
        return 1

    print(textwrap.fill(result.answer, width=88))
    if not result.answerable:
        print("\n(The passages found don't answer this.)")
    for c in result.citations:
        print(f'\n  {c.life} s{c.section}: "{c.quote}"')
    searched = ", ".join(f"{p.life.split()[-1]} s{p.section}" for p in result.passages)
    print(f"\nSearched: {searched}")
    print(
        f"{result.model}, prompt {result.prompt_version}, search {result.search_version}"
        f" | {result.usage.input_tokens:,} in / {result.usage.output_tokens:,} out tokens,"
        f" ${result.usage.cost_usd or 0:.5f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
