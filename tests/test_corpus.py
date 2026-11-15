"""Parsing and chunking the Gutenberg text. No network, no database."""

import pytest

from core.corpus import (
    LIVES,
    MAX_CHUNK_WORDS,
    MIN_PARAGRAPH_WORDS,
    MIN_PIECE_WORDS,
    Chunk,
    CorpusError,
    chunk_corpus,
    chunk_life,
    embedding_input,
    paragraphs,
    split_lives,
    split_long,
    strip_gutenberg,
)
from scripts.fetch_corpus import RAW_PATH, sha256


def sentence(n: int, words: int = 12) -> str:
    return " ".join(["word"] * (words - 2) + [f"s{n}", "end."]).capitalize()


def paragraph(words: int, tag: str = "") -> str:
    """A paragraph of about `words` words, as 12-word sentences, wrapped like Gutenberg."""
    sentences = [sentence(i) for i in range(max(1, words // 12))]
    text = f"{tag} " + " ".join(sentences) if tag else " ".join(sentences)
    out, line = [], []
    for w in text.split():  # wrap at ~70 characters, as the source does
        line.append(w)
        if len(" ".join(line)) > 65:
            out.append(" ".join(line))
            line = []
    out.append(" ".join(line))
    return "\n".join(out)


OTHER_HEADINGS = ["NICIAS", "COMPARISON OF CRASSUS WITH NICIAS", "OTHO"]


def book(
    headings: list[str] | None = None,
    body_headings: list[str] | None = None,
    bodies: dict[str, str] | None = None,
) -> str:
    """A small text laid out like eBook #674."""
    toc = headings or [*LIVES, *OTHER_HEADINGS]
    parts = [
        "The Project Gutenberg eBook of a test",
        "",
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***",
        "Plutarch's Lives",
        "",
        "The following are the names of the chapters.",
        "",
        "",
        *toc,
        "",
        "Transcriber",
        "*****",
        "",
    ]
    for heading in body_headings or toc:
        text = (bodies or {}).get(heading) or (
            paragraph(120, tag=f"About {heading.title()}.") + "\n\n" + paragraph(90)
        )
        parts += [heading, "", text, "", "", ""]
    parts += ["*** END OF THE PROJECT GUTENBERG EBOOK TEST ***", "license text"]
    return "\r\n".join(parts)  # the source uses Windows line ends


# --- strip_gutenberg / split_lives --------------------------------------------


class TestParsing:
    def test_strips_header_footer_and_line_ends(self):
        body = strip_gutenberg(book())
        assert "license text" not in body
        assert "Project Gutenberg eBook of" not in body
        assert "\r" not in body

    def test_missing_markers(self):
        with pytest.raises(CorpusError, match="START/END"):
            strip_gutenberg("no markers here")

    def test_every_heading_becomes_a_section(self):
        sections = split_lives(strip_gutenberg(book()))
        assert list(sections) == [*LIVES, *OTHER_HEADINGS]
        assert sections["CRASSUS"].strip().startswith("About Crassus.")

    def test_a_life_stops_at_the_next_heading(self):
        sections = split_lives(strip_gutenberg(book()))
        assert "About Nicias" not in sections["CRASSUS"]
        assert "About Comparison" not in sections["NICIAS"]

    def test_missing_life_is_an_error(self):
        body_headings = [h for h in [*LIVES, *OTHER_HEADINGS] if h != "CAESAR"]
        with pytest.raises(CorpusError, match="'CAESAR' found 0 times"):
            split_lives(strip_gutenberg(book(body_headings=body_headings)))

    def test_duplicate_heading_is_an_error(self):
        body_headings = [*LIVES, "POMPEY", *OTHER_HEADINGS]
        with pytest.raises(CorpusError, match="'POMPEY' found 2 times"):
            split_lives(strip_gutenberg(book(body_headings=body_headings)))

    def test_no_contents(self):
        with pytest.raises(CorpusError, match="Table of contents"):
            split_lives(strip_gutenberg(book(headings=["CAESAR", "POMPEY"])))


# --- paragraphs / split_long ----------------------------------------------------


class TestParagraphs:
    def test_whitespace_collapsed(self):
        assert paragraphs(paragraph(60)) == [" ".join(paragraph(60).split())]

    def test_verse_merges_into_the_paragraph_before(self):
        text = (
            f"{paragraph(60)}\n\n    In revelings and Bacchic play,\n\n{paragraph(60)}"
        )
        result = paragraphs(text)
        assert len(result) == 2
        assert result[0].endswith("\nIn revelings and Bacchic play,")

    def test_leading_verse_merges_into_the_next(self):
        result = paragraphs(f"For not alone\n\n{paragraph(60)}")
        assert len(result) == 1 and result[0].startswith("For not alone\n")

    def test_every_paragraph_is_long_enough(self):
        text = "\n\n".join(paragraph(n) for n in (60, 12, 80, 24, 70))
        assert all(len(p.split()) >= MIN_PARAGRAPH_WORDS for p in paragraphs(text))


class TestSplitLong:
    def test_short_paragraph_is_kept_whole(self):
        p = " ".join(paragraph(MAX_CHUNK_WORDS).split())
        assert split_long(p) == [p]

    def test_long_paragraph_splits_at_sentence_ends(self):
        p = " ".join(paragraph(1000).split())
        pieces = split_long(p)
        assert len(pieces) > 1
        assert all(piece.endswith("end.") for piece in pieces)
        assert all(len(piece.split()) >= MIN_PIECE_WORDS for piece in pieces)
        assert " ".join(pieces) == p  # nothing lost, nothing repeated

    def test_small_last_piece_joins_the_one_before(self):
        p = " ".join(paragraph(660).split())  # 300 + 300 + 60 words
        pieces = split_long(p)
        assert len(pieces) == 2
        assert len(pieces[-1].split()) >= MIN_PIECE_WORDS


# --- chunk_life / chunk_corpus ---------------------------------------------------


class TestChunks:
    def test_sections_from_one_and_indexes_from_zero(self):
        text = "\n\n".join([paragraph(100), paragraph(1000), paragraph(100)])
        chunks = chunk_life("Crassus", text)
        assert [(c.section, c.chunk_index) for c in chunks][:2] == [(1, 0), (2, 0)]
        assert {c.section for c in chunks} == {1, 2, 3}
        second = [c.chunk_index for c in chunks if c.section == 2]
        assert second == list(range(len(second))) and len(second) > 1

    def test_only_the_seven_lives_are_chunked(self):
        chunks = chunk_corpus(split_lives(strip_gutenberg(book())))
        assert {c.life for c in chunks} == set(LIVES.values())
        assert not any("Nicias" in c.text or "Otho" in c.text for c in chunks)

    def test_same_input_same_chunks(self):
        sections = split_lives(strip_gutenberg(book()))
        assert chunk_corpus(sections) == chunk_corpus(sections)

    def test_embedding_input_names_the_life(self):
        chunk = Chunk(
            life="Tiberius Gracchus", section=4, chunk_index=0, text="He spoke."
        )
        assert embedding_input(chunk) == (
            "Plutarch, Life of Tiberius Gracchus, section 4.\n\nHe spoke."
        )


# --- The real source file (only where it has been downloaded) ------------------


@pytest.mark.skipif(not RAW_PATH.exists(), reason="run scripts.fetch_corpus first")
def test_real_source_chunks_as_expected():
    from core.corpus import SOURCE_SHA256
    from scripts.ingest import load_chunks

    assert sha256(RAW_PATH) == SOURCE_SHA256
    chunks = load_chunks()
    assert len(chunks) == 470
    assert min(c.words for c in chunks) >= MIN_PARAGRAPH_WORDS
    caius = [c for c in chunks if c.life == "Caius Gracchus"]
    assert caius[-1].text.endswith("prevent our bearing them reasonably.")
