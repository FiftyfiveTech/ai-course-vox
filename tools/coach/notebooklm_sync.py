#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["notebooklm-py>=0.8"]
# ///
"""Sync this repo's learning docs into the shared NotebookLM notebook, and pull
Video Overviews back down for the coach pages.

Why this exists: the coach pages embed short Video Overviews, and getting one by hand
means opening the notebook, uploading whichever docs changed, clicking generate, and
saving the file under exactly the name the page probes for. Four manual steps that
silently rot. This is the same work, reproducible, using the unofficial NotebookLM
client (https://github.com/teng-lin/notebooklm-py).

Auth is *not* handled here and cannot be: NotebookLM has no API keys, so the client
reuses a Google browser session. Run once, interactively, on your own machine::

    uv tool install "notebooklm-py[browser]"
    notebooklm login          # opens a browser; saves ~/.notebooklm/profiles/default

Then this script is repeatable::

    uv run --script tools/coach/notebooklm_sync.py --dry-run   # show the plan
    uv run --script tools/coach/notebooklm_sync.py             # upload docs + build videos

The notebook id is not hardcoded: pass ``--notebook <id>`` or set ``VOX_NOTEBOOK_ID``.
Creating and sharing the notebook is a one-time human step — see docs/learning/README.md.

Nothing here writes to the repo except ``docs/learning/coach/videos/`` (git-ignored).

Note on what leaves the machine: every path in :data:`SOURCE_DOCS` is uploaded to a
Google product. They are all tracked design/learning docs written for exactly that
purpose. Recordings, transcripts and eval labels must never become a notebook source —
that is how a held-out case leaks out of the repo that seals it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from notebooklm import NotebookLMClient, VideoStyle

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DOCS = (
    ("ARCHITECTURE.md", "vox-architecture.md"),
    ("evals/ENTITY_SPEC.md", "vox-entity-spec.md"),
    ("docs/learning/vox-day1-concepts.md", "vox-day1-concepts.md"),
    ("docs/learning/retros.md", "vox-retros.md"),
)
"""``(repo-relative path, notebook source title)`` pairs making up the source set.

The title is explicit rather than derived from the filename so a re-run matches what is
already in the notebook instead of adding a second copy of the same doc to compete in
retrieval. Add a row when a ticket adds a primer; keep the title stable afterwards.

Missing files are skipped with a warning rather than fatal: primers live on ticket
branches that are not merged yet, so the useful behaviour on ``dev`` is "upload what
exists".
"""

VIDEOS = (
    {
        "filename": "vox-day1-pipeline-latency.mp4",
        "style": VideoStyle.WHITEBOARD,
        "instructions": (
            "A tight technical explainer for the two engineers who built this, not an "
            "audience of beginners. Cover: the turn loop VAD -> STT -> NLU -> TTS and "
            "why each stage sits local or remote (VAD runs per 32 ms frame, TTS needs no "
            "key, the two expensive stages run on a free tier); arms as a data table "
            "named by Hugging Face repo id rather than if/else on provider; then the "
            "latency split — five fields joined to the call log by turn_id, "
            "time_to_first_audio measured from the last speech frame so the VAD hangover "
            "is inside the number — and why an n=4 range with 2-4x variance per stage is "
            "a measurement problem before it is an optimisation problem. "
            "Be concrete and name the pitfalls. Under 6 minutes."
        ),
    },
    {
        "filename": "vox-day1-fallback-bargein.mp4",
        "style": VideoStyle.WHITEBOARD,
        "instructions": (
            "A tight technical explainer for the two engineers who built this. Cover: "
            "which failures fall back to a local arm (429, timeout, 5xx) and which "
            "deliberately do not (401 and other 4xx, missing credential, empty reply "
            "from a reasoning arm) — a fallback is a way to make a failure invisible, so "
            "the exclusions carry the design; both attempts logged, the local line "
            "tagged fallback_for, the turn marked fell_back so a fallback turn is never "
            "compared to a clean one; the Retry-After cooldown that parks a rate-limited "
            "arm instead of paying a doomed round-trip every turn. Then barge-in: "
            "nothing is killed and no second listener exists — the same endpointer spans "
            "the reply, Playback.abort() stops the device being handed more samples, and "
            "stop latency is measured from the first speech frame with the device's own "
            "buffer logged beside it. Finish on why there is no echo cancellation and "
            "what that means for the demo machine. Under 6 minutes."
        ),
    },
)
"""Video Overviews to build, keyed by the filename the coach pages already expect.

``style`` must be a :class:`~notebooklm.VideoStyle` member, not the string the CLI
accepts — the library reads ``.value`` off it and an ``AttributeError`` is all you get
otherwise.

``docs/learning/coach/vox-day1.html`` HEAD-probes these paths and reveals its video
cards only when the files exist, so a failed or skipped generation degrades to a page
without videos rather than a broken one.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with ``notebook``, ``videos_dir``, ``timeout``, ``dry_run``,
        ``skip_sources`` and ``skip_videos``.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--notebook",
        default=os.environ.get("VOX_NOTEBOOK_ID", ""),
        help="notebook id (or a unique prefix); defaults to $VOX_NOTEBOOK_ID",
    )
    ap.add_argument(
        "--videos-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "learning" / "coach" / "videos",
        help="where downloaded Video Overviews land (git-ignored)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="seconds to wait per video generation (default: 1800)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    ap.add_argument("--skip-sources", action="store_true", help="do not upload docs")
    ap.add_argument("--skip-videos", action="store_true", help="do not generate videos")
    return ap.parse_args(argv)


def resolve_sources() -> tuple[list[tuple[Path, str]], list[str]]:
    """Split :data:`SOURCE_DOCS` into files that exist and paths that do not.

    Returns:
        ``(present, missing)`` — ``(absolute path, notebook title)`` pairs, and
        repo-relative strings for the docs that are not on this branch.
    """
    present: list[tuple[Path, str]] = []
    missing: list[str] = []
    for rel, title in SOURCE_DOCS:
        path = REPO_ROOT / rel
        if path.is_file():
            present.append((path, title))
        else:
            missing.append(rel)
    return present, missing


async def sync(args: argparse.Namespace) -> int:
    """Upload sources and build videos.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code — 0 on success, 1 if any video failed to generate.
    """
    present, missing = resolve_sources()
    for rel in missing:
        print(f"  skip (not on this branch): {rel}", file=sys.stderr)

    if args.dry_run:
        print(f"notebook: {args.notebook or '(unset — pass --notebook or set VOX_NOTEBOOK_ID)'}")
        print("would upload:")
        for path, title in present:
            print(f"  + {path.relative_to(REPO_ROOT).as_posix()} -> {title}")
        print("would generate:")
        for spec in VIDEOS:
            print(f"  + {spec['filename']} (style={spec['style'].name.lower()})")
        return 0

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    async with NotebookLMClient.from_storage() as client:
        if not args.skip_sources:
            existing = {s.title for s in await client.sources.list(args.notebook) if s.title}
            for path, title in present:
                # NotebookLM has no "replace source" — re-uploading a same-titled doc
                # would leave two copies competing in retrieval. Skipping titles that are
                # already there keeps re-runs idempotent; delete the source in the UI (or
                # `notebooklm source delete`) when a doc has changed materially.
                if title in existing:
                    print(f"  = already a source: {title}")
                    continue
                await client.sources.add_file(args.notebook, path, wait=True, title=title)
                print(f"  + uploaded: {title}")

        if not args.skip_videos:
            for spec in VIDEOS:
                out = args.videos_dir / spec["filename"]
                if out.exists():
                    print(f"  = video already downloaded: {out.name}")
                    continue
                print(f"  ~ generating {out.name} (this takes minutes)")
                status = await client.artifacts.generate_video(
                    args.notebook,
                    instructions=spec["instructions"],
                    video_style=spec["style"],
                )
                status = await client.artifacts.wait_for_completion(
                    args.notebook, status.task_id, timeout=args.timeout
                )
                if not status.is_complete:
                    print(f"  ! failed ({status.status}): {status.error}", file=sys.stderr)
                    failures += 1
                    continue
                await client.artifacts.download_video(
                    args.notebook, str(out), artifact_id=status.task_id
                )
                print(f"  > saved {out.relative_to(REPO_ROOT).as_posix()}")

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit code. A missing notebook id exits 2, and so do authentication
        problems, because both are fixed by a command the human runs themselves.
    """
    args = parse_args(argv)
    if not args.notebook:
        print(
            "no notebook id: pass --notebook <id> or set VOX_NOTEBOOK_ID.\n"
            "The shared notebook is created once by hand — see docs/learning/README.md.",
            file=sys.stderr,
        )
        return 2
    try:
        return asyncio.run(sync(args))
    except Exception as exc:  # noqa: BLE001 — surface the fix, not a traceback
        blob = f"{type(exc).__name__}: {exc}"
        if any(word in blob.lower() for word in ("auth", "login", "credential", "storage")):
            print(f"{blob}\n\nRun `notebooklm login` once, then retry.", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
