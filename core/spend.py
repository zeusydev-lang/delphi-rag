"""What the public site spent today, against its daily cap.

    with connect("query") as conn:
        if spent_today(conn) >= site_cap():
            ...  # cache-only mode
        add_spend(conn, 0.0005, "fresh")

Days are UTC. The cap is checked before each request, so two requests that
arrive together can both pass: the cap can be exceeded by about one answer.
"""

import os
from typing import Literal

import psycopg

Kind = Literal["fresh", "hit", "refused"]
DEFAULT_SITE_CAP_USD = 0.10


def site_cap() -> float:
    return float(os.environ.get("SITE_DAILY_CAP_USD", DEFAULT_SITE_CAP_USD))


def spent_today(conn: psycopg.Connection) -> float:
    row = conn.execute(
        "select cost_usd from daily_spend where day = (now() at time zone 'utc')::date"
    ).fetchone()
    return float(row["cost_usd"]) if row else 0.0


def add_spend(conn: psycopg.Connection, cost_usd: float, kind: Kind) -> None:
    """Add one request to today's totals (atomic: concurrent requests both count)."""
    counts = {
        "fresh": (1, 0, 0),
        "hit": (0, 1, 0),
        "refused": (0, 0, 1),
    }[kind]
    conn.execute(
        """
        insert into daily_spend as d
            (day, cost_usd, fresh_answers, cache_hits, refused_over_cap)
        values ((now() at time zone 'utc')::date, %s, %s, %s, %s)
        on conflict (day) do update set
            cost_usd = d.cost_usd + excluded.cost_usd,
            fresh_answers = d.fresh_answers + excluded.fresh_answers,
            cache_hits = d.cache_hits + excluded.cache_hits,
            refused_over_cap = d.refused_over_cap + excluded.refused_over_cap
        """,
        [max(cost_usd, 0.0), *counts],
    )
