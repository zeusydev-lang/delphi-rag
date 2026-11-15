"""Download the Plutarch source text once and check it is the pinned version.

Saves Project Gutenberg eBook #674 to data/raw/pg674.txt (gitignored). If the
file is already there, nothing is downloaded. Either way its SHA-256 must
match core.corpus.SOURCE_SHA256: a changed file would build a different
corpus under the same version name, so the script stops instead.

Usage:
    uv run python -m scripts.fetch_corpus
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

from core.corpus import SOURCE_SHA256, SOURCE_URL

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "pg674.txt"
USER_AGENT = "delphi-rag-corpus-fetch/0.1 (one-time download of a public-domain text)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(path: Path = RAW_PATH) -> Path:
    if path.exists():
        print(f"Already downloaded: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        print(f"Downloaded {len(data):,} bytes to {path}")
    return path


def main() -> int:
    path = fetch()
    digest = sha256(path)
    if digest != SOURCE_SHA256:
        print(
            f"SHA-256 mismatch.\n  expected {SOURCE_SHA256}\n  got      {digest}\n"
            "The source changed. Check the differences, then pin the new hash in "
            "core/corpus.py with a new CORPUS_VERSION.",
            file=sys.stderr,
        )
        return 1
    print(f"SHA-256 ok: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
