"""Build the chunk file and print what it contains (VOX-029's verification).

    make index                                  build from config.SOURCES_DIR
    uv run python scripts/build_index.py --dry  count only, write nothing

This is the ticket's evidence, not a smoke test bolted on afterwards: the criterion is "the chunk
file exists and a smoke-test prints chunk count", so the script builds, then re-reads the file it
just wrote and reports the counts from *disk* rather than from the in-memory report. What you see
is what a retriever will load.

Empty pages are named individually. A page with no extractable text means either a cover page or
an image-only scan, and the second one would make the whole POC return nothing at query time —
that has to be visible here, at load, not three tickets later.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console is cp1252

from src.config import (CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, CHUNKS_FILE, SOURCES_DIR,
                        TOKENIZER_REPO)                                       # noqa: E402
from src.sources import build_index, load_chunks                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="source folder (default: config.SOURCES_DIR)")
    ap.add_argument("--out", default=None, help="chunk file (default: config.CHUNKS_FILE)")
    ap.add_argument("--dry", action="store_true", help="write to a temp path, report, delete")
    args = ap.parse_args()

    root = Path(args.root or SOURCES_DIR)
    out = Path(args.out or CHUNKS_FILE)
    if args.dry:
        out = out.with_suffix(".dry.jsonl")

    print(f"sources    {root}")
    print(f"chunks     {out}")
    print(f"geometry   {CHUNK_TOKENS} tokens, {CHUNK_OVERLAP_TOKENS} overlap, counted with "
          f"{TOKENIZER_REPO}")
    print()

    report = build_index(root=root, out=out)

    width = max((len(d) for d, _, _ in report.per_doc), default=8)
    print(f"  {'doc_id':<{width}}  pages  chunks")
    for doc_id, pages, chunks in report.per_doc:
        print(f"  {doc_id:<{width}}  {pages:>5}  {chunks:>6}")
    print()

    if report.empty_pages:
        print(f"pages with no extractable text ({len(report.empty_pages)}) — cover pages, or a "
              f"scanned PDF this POC cannot read:")
        for name in report.empty_pages:
            print(f"    {name}")
        print()

    # Read back what reached disk. The count that matters is the one a retriever will see.
    on_disk = load_chunks(out)
    pages_with_text = report.pages - len(report.empty_pages)
    print(f"files      {report.files}")
    print(f"pages      {report.pages}  ({pages_with_text} with text, "
          f"{len(report.empty_pages)} empty)")
    print(f"chunks     {len(on_disk)}")
    print(f"tokens     {report.tokens}  ({report.tokens // max(pages_with_text, 1)} per page with "
          f"text)")
    # The overlap only does something on a page long enough to split. Printed because it is the
    # honest scope of "~300 tokens with 50-token overlap" on a corpus of short pages.
    print(f"split pages {report.multi_chunk_pages}  (pages that produced more than one chunk, "
          f"i.e. where the {CHUNK_OVERLAP_TOKENS}-token overlap applies)")

    assert len(on_disk) == report.chunks, "chunk file disagrees with the build report"
    if args.dry:
        out.unlink()
        print("\n--dry: chunk file deleted")


if __name__ == "__main__":
    main()
