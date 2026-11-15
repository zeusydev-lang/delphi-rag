"""Record an eval run in the eval_runs table, through the eval login.

    record_run("retrieval", report, Path("evals/results/retrieval/rrf-v1.json"))

The JSON results file stays the source of truth (it's in git); the table keeps
the summary of every run, committed or not, with the commit it ran on.
Recording never fails an eval: with no EVAL_DATABASE_URL or no database, it
warns and returns False.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import psycopg
from psycopg.types.json import Jsonb

from core.db import connect
from core.settings import SettingsError

Kind = Literal["retrieval", "answers", "cache"]
ROOT = Path(__file__).resolve().parent.parent

# Report keys that say WHAT was measured (the rest are results).
VERSION_KEYS = (
    "search_version",
    "prompt_version",
    "cache_version",
    "judge_prompt_version",
    "corpus_version",
    "labels",
    "model",
    "embedding_model",
    "judge_model",
    "hit_at",
    "miss_below",
)
# Report keys that are the summary results, per kind.
RATE_KEYS: dict[str, tuple[str, ...]] = {
    "retrieval": ("rates", "questions"),
    "answers": ("rates", "by_category", "questions", "errors"),
    "cache": (
        "false_hits",
        "hit_rate_on_paraphrases",
        "judge_calls",
        "outcomes",
        "pairs",
    ),
}


def git_state() -> tuple[str | None, bool | None]:
    """(commit, dirty) for the repo, or (None, None) outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def run_row(
    kind: Kind, report: dict, results_file: Path | None, *, with_git: bool = True
) -> dict:
    """The eval_runs row for one report (no database needed).

    with_git=False for backfills: the commit a past run used isn't known, and
    today's commit would be wrong.
    """
    commit, dirty = git_state() if with_git else (None, None)
    rel = None
    if results_file is not None:
        try:
            rel = results_file.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(results_file)
    return {
        "kind": kind,
        # Files from before labels were versioned were all measured on labels v1.
        "versions": {"labels": "1"}
        | {k: report[k] for k in VERSION_KEYS if k in report},
        "pair_set": report.get("pair_set", "main"),
        "rates": {k: report[k] for k in RATE_KEYS[kind] if k in report},
        "cost_usd": report.get("cost_usd"),
        "git_commit": commit,
        "git_dirty": dirty,
        "results_file": rel,
        "measured_at": datetime.fromisoformat(report["measured_at"]),
    }


def record_run(
    kind: Kind, report: dict, results_file: Path | None = None, *, with_git: bool = True
) -> bool:
    """Insert one run. True if a row was added; False if skipped or not recorded."""
    row = run_row(kind, report, results_file, with_git=with_git)
    try:
        with connect("eval") as conn:
            cur = conn.execute(
                """
                insert into eval_runs
                    (kind, versions, pair_set, rates, cost_usd, git_commit,
                     git_dirty, results_file, measured_at)
                values (%(kind)s, %(versions)s, %(pair_set)s, %(rates)s, %(cost_usd)s,
                        %(git_commit)s, %(git_dirty)s, %(results_file)s, %(measured_at)s)
                on conflict do nothing
                """,
                row
                | {"versions": Jsonb(row["versions"]), "rates": Jsonb(row["rates"])},
            )
            added = cur.rowcount == 1
    except (SettingsError, psycopg.Error) as err:
        print(
            f"  (eval run not recorded: {type(err).__name__}: {err})", file=sys.stderr
        )
        return False
    return added


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
