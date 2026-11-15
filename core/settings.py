"""Connection settings for each database login.

Each job asks for its own role:
    ingest -> INGEST_DATABASE_URL  (write chunks of the corpus)
    query  -> QUERY_DATABASE_URL   (read chunks; read and write the answer cache)

The owner login (DATABASE_URL) is deliberately not available here.
It is only for migrations/.
"""

import os
from typing import Literal

from dotenv import load_dotenv

Role = Literal["ingest", "query"]

ENV_KEYS: dict[Role, str] = {
    "ingest": "INGEST_DATABASE_URL",
    "query": "QUERY_DATABASE_URL",
}

# Values already set in the environment (GitHub Secrets, --env-file) win over .env.
load_dotenv(override=False)


class SettingsError(RuntimeError):
    """A required setting is missing."""


def database_url(role: Role) -> str:
    """Return the connection string for one role, or fail with a clear message."""
    if role not in ENV_KEYS:
        raise SettingsError(
            f"Unknown role {role!r}. Use one of: {', '.join(ENV_KEYS)}."
        )
    key = ENV_KEYS[role]
    url = os.environ.get(key, "").strip()
    if not url:
        raise SettingsError(
            f"{key} is not set. Run: uv run python migrations/logins.py --env-file .env"
        )
    return url
