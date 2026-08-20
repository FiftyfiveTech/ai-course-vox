"""VOX-029. The chunker's geometry and the loader's reporting, without needing the PDF corpus.

`sources/` is gitignored (internal HR policies), so a test that read it would pass on this machine
and fail on a clean clone. Everything here runs on synthetic text and a fake tokenizer, except one
test that builds a real one-page PDF with pypdf so the extract path is exercised too.

The fake tokenizer is a word splitter with the same interface transformers gives us. That is the
point: it makes the *window arithmetic* assertable without pinning the assertions to one model's
vocabulary. Whether the real tokenizer is loaded and counted with is config, and tested where it
belongs — in the numbers `make index` prints.
"""
import json

import pytest

from src import sources


class WordTokenizer:
    """The two methods src.sources.chunk_text uses, over whitespace words."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


@pytest.fixture
def tok():
    return WordTokenizer()


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_short_text_is_one_chunk(tok):
    out = sources.chunk_text(words(10), tok, size=300, overlap=50)
    assert out == [words(10)]


def test_chunks_are_size_tokens_and_step_by_size_minus_overlap(tok):
    out = sources.chunk_text(words(700), tok, size=300, overlap=50)
    # 700 words, step 250: windows at 0, 250, 500 — the one at 750 would not exist.
    assert len(out) == 3
    assert out[0].split()[0] == "w0" and len(out[0].split()) == 300
    assert out[1].split()[0] == "w250" and len(out[1].split()) == 300
    assert out[2].split()[0] == "w500" and len(out[2].split()) == 200


def test_consecutive_chunks_overlap_by_exactly_the_overlap(tok):
    out = sources.chunk_text(words(700), tok, size=300, overlap=50)
    for a, b in zip(out, out[1:]):
        assert a.split()[-50:] == b.split()[:50]


def test_no_token_is_lost(tok):
    """Every word appears in some chunk — an off-by-one in the step would drop a band."""
    out = sources.chunk_text(words(700), tok, size=300, overlap=50)
    seen = {w for chunk in out for w in chunk.split()}
    assert seen == set(words(700).split())


def test_trailing_window_that_is_all_overlap_is_dropped(tok):
    """550 words, step 250: the window at 500 holds 50 words the chunk before it already had."""
    out = sources.chunk_text(words(550), tok, size=300, overlap=50)
    assert len(out) == 2
    assert out[-1].split()[-1] == "w549"


def test_empty_text_is_no_chunks(tok):
    assert sources.chunk_text("", tok) == []
    assert sources.chunk_text("   ", tok, size=10, overlap=2) == []


def test_overlap_must_be_smaller_than_size(tok):
    """Otherwise the step is <= 0 and the window never advances — a hang, not a bad result."""
    with pytest.raises(ValueError):
        sources.chunk_text(words(10), tok, size=50, overlap=50)


def test_clean_collapses_extraction_whitespace_and_keeps_real_characters():
    assert sources.clean("Leave  Policy \n \n  1.  Purpose \n") == "Leave Policy 1. Purpose"
    # pypdf gives us these from the real corpus; they are content, not damage.
    assert sources.clean("• twelve days ’ leave") == "• twelve days ’ leave"
    assert sources.clean(None) == ""


def test_missing_source_folder_says_so(tmp_path):
    with pytest.raises(RuntimeError, match="no source folder"):
        sources.pdf_paths(tmp_path / "nope")


def test_build_index_writes_the_four_named_fields(tmp_path, tok, monkeypatch):
    """The acceptance criterion names the fields, so the JSONL is asserted field by field."""
    monkeypatch.setattr(sources, "extract_pages",
                        lambda path: [(1, "alpha beta"), (2, ""), (3, "gamma")])
    (tmp_path / "leave-policy.pdf").write_bytes(b"%PDF-1.4 stub")
    out = tmp_path / "chunks.jsonl"

    report = sources.build_index(root=tmp_path, out=out, tokenizer=tok)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [sorted(r) for r in rows] == [["chunk_idx", "doc_id", "page", "text"]] * 2
    assert rows[0] == {"doc_id": "leave-policy", "page": 1, "chunk_idx": 0, "text": "alpha beta"}
    # chunk_idx runs across the document rather than restarting per page, so (doc_id, chunk_idx)
    # is unique on its own.
    assert rows[1] == {"doc_id": "leave-policy", "page": 3, "chunk_idx": 1, "text": "gamma"}
    assert (report.files, report.pages, report.chunks) == (1, 3, 2)


def test_a_page_with_no_text_is_named_not_skipped_silently(tmp_path, tok, monkeypatch):
    """The scanned-PDF tripwire: an image-only page must be reported by name at load time."""
    monkeypatch.setattr(sources, "extract_pages", lambda path: [(1, ""), (2, "text"), (3, "  ")])
    (tmp_path / "scanned.pdf").write_bytes(b"%PDF-1.4 stub")

    report = sources.build_index(root=tmp_path, out=tmp_path / "c.jsonl", tokenizer=tok)

    assert report.empty_pages == ["scanned p1", "scanned p3"]
    assert report.chunks == 1


def test_load_chunks_round_trips_and_complains_when_absent(tmp_path, tok, monkeypatch):
    monkeypatch.setattr(sources, "extract_pages", lambda path: [(1, "alpha beta")])
    (tmp_path / "d.pdf").write_bytes(b"%PDF-1.4 stub")
    out = tmp_path / "chunks.jsonl"

    sources.build_index(root=tmp_path, out=out, tokenizer=tok)
    assert sources.load_chunks(out) == [
        {"doc_id": "d", "page": 1, "chunk_idx": 0, "text": "alpha beta"}]

    with pytest.raises(RuntimeError, match="no chunk file"):
        sources.load_chunks(tmp_path / "gone.jsonl")


def test_extract_pages_reads_a_real_pdf(tmp_path):
    """One page in, one page of text out — the pypdf path itself, no mocking."""
    pypdf = pytest.importorskip("pypdf")
    # Written here rather than checked in as a binary fixture: what is being tested is that
    # extract_pages numbers pages from 1 and reports no-text as '', not what pypdf can parse.
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    pdf = tmp_path / "blank.pdf"
    with pdf.open("wb") as f:
        writer.write(f)

    pages = sources.extract_pages(pdf)
    assert len(pages) == 1
    assert pages[0][0] == 1              # 1-based page numbers, as the provenance field needs
    assert pages[0][1] == ""             # a blank page yields no text, and says so rather than raising
