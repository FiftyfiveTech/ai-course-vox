"""PDF source folder to text chunks (VOX-029).

The first half of the POC: a folder of PDFs in, one JSON lines file of chunks out. No model call,
no network, no key — `pypdf` is a parser, the tokenizer is counted with and never run, so nothing
here touches the cost logger. What it does need is a *latency* stamp eventually (VOX-032 adds
`t_retrieval_ms` to the turn record); indexing itself happens once at startup, not per turn.

Three decisions worth knowing before reading the code:

**Chunks never span a page.** Each page is extracted and chunked on its own, so a chunk's
`(doc_id, page)` is exact rather than approximate — which is the whole point of carrying
provenance into a spoken answer. The cost is that a sentence continuing across a page break is
split, and on this corpus that happens: `make index` measured 250 tokens per page with text, and
52 of the 163 such pages were long enough to split. `build_index` prints that count, so the size
of the trade stays a measured number rather than a comment.

**The overlap is 50 tokens of the *encoded* window, not of the re-encoded chunk.** The window
slides over token ids and each window is decoded back to text, so consecutive chunks on a page
share exactly 50 ids. Re-tokenizing the stored text does not reproduce those ids exactly, because
a chunk that starts mid-word merges differently — measured over the real corpus, the shared text
between consecutive chunks re-tokenizes to 49-52 tokens (mean 50, never 0). Chunk text is verbatim
either way: all 215 chunks are exact substrings of the page they came from.

**A page that yields no text is reported by name, not skipped.** A scanned (image-only) PDF is
the most likely reason this POC would return nothing at query time, and it has to be visible at
load. On the real corpus these are cover and section-divider pages, which is why the loader
reports them rather than refusing.

**Extraction is faithful.** The only transformation is whitespace collapsing — pypdf emits ' \\n \\n'
runs that would otherwise eat the token budget. Bullets, curly quotes and the rupee sign are left
alone: they are real characters in these documents, not extraction damage (checked, U+2022 x723,
U+2019 x117, no replacement characters anywhere in the 184 pages).
"""
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from src.config import (CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, CHUNKS_FILE, SOURCES_DIR,
                        TOKENIZER_REPO)

_WHITESPACE = re.compile(r"\s+")


def pdf_paths(root=None):
    """-> every PDF under `root`, sorted so a rebuild produces the same chunk_idx values."""
    root = Path(root or SOURCES_DIR)
    if not root.is_dir():
        raise RuntimeError(
            f"no source folder at {root}. Put the PDFs there (the folder is gitignored — these "
            f"are internal documents), or point VOX_SOURCES_DIR somewhere else."
        )
    return sorted(root.glob("*.pdf"))


def clean(text):
    """-> the page's text with whitespace collapsed, or '' if there was none.

    NFKC first, so a ligature or a full-width digit counts as the characters it reads as rather
    than as its own token. Nothing else is rewritten — see the module docstring.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text or "")).strip()


def extract_pages(path):
    """-> [(page_number, cleaned_text)] for one PDF, 1-based, empty pages included as ''.

    Empty pages stay in the list because the caller reports them by name; dropping them here
    would also renumber nothing (page numbers are read off the reader, not counted), but it would
    make the count of what produced no text impossible to recover.
    """
    reader = PdfReader(str(path))
    return [(n, clean(page.extract_text())) for n, page in enumerate(reader.pages, start=1)]


def load_tokenizer(repo=None):
    """-> the tokenizer named by config.TOKENIZER_REPO, from the local cache only.

    `local_files_only` is the "no network calls" half of the acceptance criterion: with it, an
    index build either uses files already on this machine or fails saying so. Without it the first
    build on a clean clone would quietly download ~9 MB mid-run, which is exactly the kind of
    hidden network call the criterion is about.
    """
    from transformers import AutoTokenizer  # ~2 s to import; only indexing needs it

    repo = repo or TOKENIZER_REPO
    try:
        # clean_up_tokenization_spaces is a WordPiece-era fixup that deletes spaces before
        # punctuation; transformers warns that it corrupts byte-level BPE output and ignores it
        # anyway. Set explicitly so the chunk text is decoded exactly as encoded, and so the
        # warning is not printed over every index build.
        return AutoTokenizer.from_pretrained(repo, local_files_only=True,
                                             clean_up_tokenization_spaces=False)
    except Exception as e:
        raise RuntimeError(
            f"{repo} is not in the local HF cache, and indexing does not go to the network. "
            f"Run `make tokenizer` once to fetch it, then re-run. ({type(e).__name__})"
        ) from e


def fetch_tokenizer(repo=None):
    """Download the tokenizer files once, so index builds can stay offline. `make tokenizer`."""
    from transformers import AutoTokenizer

    repo = repo or TOKENIZER_REPO
    tok = AutoTokenizer.from_pretrained(repo)
    print(f"cached {repo} — vocab {tok.vocab_size}")
    return tok


def chunk_text(text, tokenizer, size=None, overlap=None):
    """-> the text as overlapping chunks of `size` tokens, stepping `size - overlap`.

    Sliding window over the token ids, decoded back to text. Byte-level BPE round-trips, so a
    chunk is the same string the window covered (bar the leading space the decoder drops).

    A final window that is *entirely* overlap is dropped: it would carry no text the previous
    chunk did not already have, and in retrieval that is a duplicate hit with worse context.
    """
    size = size or CHUNK_TOKENS
    overlap = overlap or CHUNK_OVERLAP_TOKENS
    if overlap >= size:
        raise ValueError(f"overlap {overlap} must be smaller than chunk size {size}")
    if not text:
        return []

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    step = size - overlap
    out = []
    for start in range(0, len(ids), step):
        window = ids[start:start + size]
        if start and len(window) <= overlap:
            break
        out.append(tokenizer.decode(window, skip_special_tokens=True).strip())
        if start + size >= len(ids):
            break
    return out


@dataclass
class Report:
    """What a build measured — printed by scripts/build_index.py, asserted by the smoke test."""

    out: Path = None
    files: int = 0
    pages: int = 0
    chunks: int = 0
    tokens: int = 0
    # "doc_id p12" for every page that produced no text — the scanned-PDF tripwire.
    empty_pages: list = field(default_factory=list)
    # Pages that produced more than one chunk, i.e. the pages where the overlap did anything.
    multi_chunk_pages: int = 0
    per_doc: list = field(default_factory=list)   # (doc_id, pages, chunks) per file, in order


def build_index(root=None, out=None, tokenizer=None):
    """Extract, chunk and write every PDF under `root` to a JSON lines file. -> Report.

    One line per chunk, with exactly the fields the ticket names: doc_id, page, chunk_idx, text.
    `doc_id` is the filename stem, so a chunk points at a file that exists on disk; `chunk_idx`
    counts across the whole document rather than restarting per page, which makes (doc_id,
    chunk_idx) a unique id without needing the page as part of the key.

    The file is written whole, at the end: a half-written index that looks complete is worse than
    no index, and this corpus takes seconds.
    """
    tokenizer = tokenizer or load_tokenizer()
    out = Path(out or CHUNKS_FILE)
    report = Report(out=out)
    lines = []

    for path in pdf_paths(root):
        doc_id = path.stem
        pages = extract_pages(path)
        doc_chunks = 0
        for page_no, text in pages:
            # `extract_pages` already collapses whitespace, so this strip is only ever a no-op on
            # the real path. It is here so "produced no text" is decided in one place: a page of
            # spaces yields no chunks either way, and it must be *reported* rather than vanish.
            text = (text or "").strip()
            if not text:
                report.empty_pages.append(f"{doc_id} p{page_no}")
                continue
            chunks = chunk_text(text, tokenizer)
            if len(chunks) > 1:
                report.multi_chunk_pages += 1
            for body in chunks:
                lines.append({"doc_id": doc_id, "page": page_no,
                              "chunk_idx": doc_chunks, "text": body})
                doc_chunks += 1
            report.tokens += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        report.files += 1
        report.pages += len(pages)
        report.chunks += doc_chunks
        report.per_doc.append((doc_id, len(pages), doc_chunks))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return report


def load_chunks(path=None):
    """-> the chunk records written by `build_index`. What VOX-030 will retrieve over."""
    path = Path(path or CHUNKS_FILE)
    if not path.exists():
        raise RuntimeError(f"no chunk file at {path} — run `make index` first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
