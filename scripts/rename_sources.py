"""Rename the source PDFs from export hashes to their own titles (VOX-029).

    uv run python scripts/rename_sources.py            what it would do, no changes
    uv run python scripts/rename_sources.py --apply     do it

The corpus arrives with names like `8f0a7775e8b149cf8de3528d379c9a1e.pdf`. `doc_id` is the
filename stem, and provenance is meant to be *spoken* by VOX-031 — "holiday-policy p3" is an
answer, "8f0a7775e8b149cf8de3528d379c9a1e p3" is a hash. So the files are renamed once, here,
rather than every consumer carrying a lookup table.

The new name comes from the document's own first title-like line, so this is repeatable when the
corpus is re-exported with fresh hashes. Two titles need help and are overridden by the text they
match, not by filename — see OVERRIDES.
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console is cp1252

from src.sources import clean, pdf_paths                                     # noqa: E402
from pypdf import PdfReader                                                  # noqa: E402

# Keyed by the detected title, because that survives a re-export while a filename does not.
#   the holiday list is a bare table, so its first prose line is the column header row
#   the POSH policy's real title is accurate and 55 characters long
OVERRIDES = {
    "No Holiday Month Date Day Remarks": "holiday-list-2026",
    "Prevention of Sexual Harassment in The Workplace Policy":
        "prevention-of-sexual-harassment-policy",
}

# A title line has letters, is not a page number, and is not a whole paragraph.
TITLE_MIN_CHARS = 5
TITLE_MAX_CHARS = 80


def detect_title(path):
    """-> the first line that looks like a document title, or '' if no page had one.

    Scans pages in order rather than trusting page 1: three of these PDFs have a cover page whose
    extracted text is `1` or a truncated fragment, so page 1 is often not where the title is.
    """
    for page in PdfReader(str(path)).pages:
        for line in (page.extract_text() or "").splitlines():
            s = clean(line)
            if (TITLE_MIN_CHARS <= len(s) <= TITLE_MAX_CHARS
                    and not s.isdigit() and re.search(r"[A-Za-z]{3}", s)):
                return s
    return ""


def slugify(title):
    """-> 'Code of Ethics & Business Conduct' as 'code-of-ethics-business-conduct'."""
    ascii_only = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", ascii_only.lower())).strip("-")


def plan(root=None):
    """-> [(path, new_name, why)] for every PDF, new_name None when it cannot be renamed."""
    out, taken = [], set()
    for path in pdf_paths(root):
        title = detect_title(path)
        if not title:
            out.append((path, None, "no title-like line on any page — extraction may be empty"))
            continue
        slug = OVERRIDES.get(title) or slugify(title)
        why = f"title {title!r}" + (" (override)" if title in OVERRIDES else "")
        # Two files with the same title are two copies of the same policy; they get distinct
        # names rather than one silently overwriting the other.
        name, n = f"{slug}.pdf", 1
        while name in taken or (path.parent / name).exists() and (path.parent / name) != path:
            n += 1
            name = f"{slug}-{n}.pdf"
        taken.add(name)
        out.append((path, name, why))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="rename the files (default: dry run)")
    ap.add_argument("--root", default=None, help="source folder (default: config.SOURCES_DIR)")
    args = ap.parse_args()

    rows = plan(args.root)
    width = max((len(p.name) for p, _, _ in rows), default=0)
    renamed = skipped = 0
    for path, name, why in rows:
        if name is None:
            print(f"  SKIP  {path.name:<{width}}  {why}")
            skipped += 1
            continue
        if name == path.name:
            print(f"  ok    {path.name:<{width}}  already named for its {why}")
            continue
        print(f"  {'mv  ' if args.apply else 'plan'}  {path.name:<{width}}  ->  {name}   {why}")
        if args.apply:
            path.rename(path.parent / name)
            renamed += 1

    if args.apply:
        print(f"\nrenamed {renamed} file(s), skipped {skipped}. Re-run `make index`: doc_id is "
              f"the filename stem, so every chunk id has changed.")
    else:
        print(f"\ndry run — nothing changed. Re-run with --apply to rename {len(rows) - skipped} "
              f"file(s).")


if __name__ == "__main__":
    main()
