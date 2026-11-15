"""Turn the Gutenberg text of Plutarch's Lives into chunks. No network, no database.

    body = strip_gutenberg(raw_text)
    lives = split_lives(body)                 # {"TIBERIUS GRACCHUS": "...", ...}
    chunks = chunk_corpus(lives)              # list[Chunk], only the Lives in LIVES

The source (eBook #674, Dryden's translation as revised by Clough) has no
chapter numbers: each Life is a capitalised heading followed by paragraphs.
So a chunk's position is its paragraph `section` within the Life (from 1),
plus a `chunk_index` when a long paragraph is split.

Changing any chunking rule below changes the chunks: give CHUNKER a new name
and CORPUS_VERSION a new version, so cached answers from the old corpus are
never served for the new one.
"""

import re

from pydantic import BaseModel, ConfigDict

SOURCE_URL = "https://www.gutenberg.org/ebooks/674.txt.utf-8"
SOURCE_SHA256 = (
    "e849186d3483c1779053a5f70c8ae477316c019907c740f878de0d2c788975f8"  # 2026-11-15
)
CHUNKER = "para-v1"
CORPUS_VERSION = "lives-v1"

# Heading in the text -> the name stored in chunks.life.
LIVES: dict[str, str] = {
    "MARCUS CATO": "Cato the Elder",
    "CATO THE YOUNGER": "Cato the Younger",
    "TIBERIUS GRACCHUS": "Tiberius Gracchus",
    "CAIUS GRACCHUS": "Caius Gracchus",
    "CAESAR": "Caesar",
    "POMPEY": "Pompey",
    "CRASSUS": "Crassus",
}

MIN_PARAGRAPH_WORDS = (
    40  # shorter ones are verse quotations: merged into the one before
)
MAX_CHUNK_WORDS = 450  # longer paragraphs are split at sentence ends...
TARGET_CHUNK_WORDS = 300  # ...into pieces of about this size
MIN_PIECE_WORDS = 100  # a smaller last piece joins the one before it

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z])")


class CorpusError(ValueError):
    """The source text isn't laid out the way the parser expects."""


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    life: str
    section: int  # paragraph number within the Life, from 1
    chunk_index: int  # piece number within the section, from 0
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


# --- Parsing ------------------------------------------------------------------


def strip_gutenberg(text: str) -> str:
    """The text between Gutenberg's START and END markers, with \\n line ends."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^\*\*\* ?START OF .*$", text, re.MULTILINE)
    end = re.search(r"^\*\*\* ?END OF .*$", text, re.MULTILINE)
    if not start or not end or end.start() < start.end():
        raise CorpusError("Gutenberg START/END markers not found")
    return text[start.end() : end.start()]


def _is_heading_line(line: str) -> bool:
    return bool(line) and line == line.upper() and bool(re.fullmatch(r"[A-Z .]+", line))


def table_of_contents(body: str) -> list[str]:
    """The headings, from the first long run of capitalised lines (the contents)."""
    run: list[str] = []
    for raw in body.split("\n"):
        line = raw.strip()
        if _is_heading_line(line):
            run.append(line)
        elif run and len(run) >= 10:
            return run
        else:
            run = []
    if len(run) >= 10:
        return run
    raise CorpusError("Table of contents (a list of capitalised headings) not found")


def split_lives(body: str) -> dict[str, str]:
    """Every section of the book, keyed by its heading, in book order.

    Each heading appears once in the contents and once in the text, so the
    text is cut at lines that exactly equal a contents heading.
    """
    toc = table_of_contents(body)
    headings = set(toc)
    lines = body.split("\n")

    # Skip past the contents itself, so only the headings in the text count.
    toc_end = 0
    seen = 0
    for i, raw in enumerate(lines):
        if raw.strip() in headings:
            seen += 1
            if seen == len(toc):
                toc_end = i + 1
                break

    starts: list[tuple[int, str]] = []
    for i in range(toc_end, len(lines)):
        line = lines[i].strip()
        if line in headings:
            starts.append((i, line))

    found = [h for _, h in starts]
    for heading in LIVES:
        count = found.count(heading)
        if count != 1:
            raise CorpusError(f"Heading {heading!r} found {count} times in the text")

    sections: dict[str, str] = {}
    for n, (i, heading) in enumerate(starts):
        stop = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        sections[heading] = "\n".join(lines[i + 1 : stop])
    return sections


# --- Chunking -----------------------------------------------------------------


def paragraphs(text: str) -> list[str]:
    """Paragraphs with whitespace collapsed; short ones (verse) merged backwards."""
    raw = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    merged: list[str] = []
    pending = ""  # short text before any full paragraph
    for p in filter(None, raw):
        if len(p.split()) < MIN_PARAGRAPH_WORDS:
            if merged:
                merged[-1] = f"{merged[-1]}\n{p}"
            else:
                pending = f"{pending}\n{p}".strip()
        else:
            merged.append(f"{pending}\n{p}".strip() if pending else p)
            pending = ""
    if pending:  # the whole Life was short lines
        merged.append(pending)
    return merged


def split_long(paragraph: str) -> list[str]:
    """A paragraph over MAX_CHUNK_WORDS, cut at sentence ends into ~TARGET pieces."""
    if len(paragraph.split()) <= MAX_CHUNK_WORDS:
        return [paragraph]
    pieces: list[list[str]] = [[]]
    count = 0
    for sentence in _SENTENCE_END.split(paragraph):
        if count >= TARGET_CHUNK_WORDS:
            pieces.append([])
            count = 0
        pieces[-1].append(sentence)
        count += len(sentence.split())
    texts = [" ".join(p) for p in pieces if p]
    if len(texts) > 1 and len(texts[-1].split()) < MIN_PIECE_WORDS:
        last = texts.pop()
        texts[-1] = f"{texts[-1]} {last}"
    return texts


def chunk_life(life: str, text: str) -> list[Chunk]:
    chunks = []
    for section, paragraph in enumerate(paragraphs(text), start=1):
        for index, piece in enumerate(split_long(paragraph)):
            chunks.append(
                Chunk(life=life, section=section, chunk_index=index, text=piece)
            )
    return chunks


def chunk_corpus(sections: dict[str, str]) -> list[Chunk]:
    """Chunks for the Lives in LIVES only (Comparisons and other Lives are skipped)."""
    chunks: list[Chunk] = []
    for heading, life in LIVES.items():
        chunks += chunk_life(life, sections[heading])
    return chunks


def embedding_input(chunk: Chunk) -> str:
    """What gets embedded: the Life's name first, since paragraphs often say only "he".

    Stored chunk text stays exactly the source's words, for citations.
    """
    return f"Plutarch, Life of {chunk.life}, section {chunk.section}.\n\n{chunk.text}"
