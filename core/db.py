"""Database connections, one role at a time.

Usage:
    with connect("query") as conn:
        conn.execute("select ...")

Everything inside the with-block is one transaction: it commits when the block
ends normally and rolls back if an error is raised. Rows come back as dicts.

Vectors (pgvector) are passed as text and cast in SQL, so no numpy is needed:
    conn.execute("... order by embedding <=> %s::vector", [to_vector(vec)])
"""

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from core.settings import Role, database_url


@contextmanager
def connect(role: Role) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(
        database_url(role),
        row_factory=dict_row,
        application_name=f"delphi-{role}",  # shows up in Neon's monitoring
        connect_timeout=15,
    )
    try:
        with conn.transaction():
            yield conn
    finally:
        conn.close()


def to_vector(values: Sequence[float]) -> str:
    """[0.1, 0.2] -> '[0.1,0.2]', pgvector's text form. Refuses NaN and infinity."""
    if not values:
        raise ValueError("A vector needs at least one value")
    floats = [float(v) for v in values]
    if not all(math.isfinite(v) for v in floats):
        raise ValueError("Vector values must be finite numbers")
    return "[" + ",".join(repr(v) for v in floats) + "]"
