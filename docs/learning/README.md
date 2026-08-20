# Learning loop

Every ticket in this repo doubles as course material. This folder holds the artifacts, and
the two tools that make them usable by a second person: the **web coach** (an interactive
lesson page wired to Claude Code) and a shared **NotebookLM** notebook (Q&A and short video
overviews over the same docs).

Both are developer-agnostic on purpose. Nothing here depends on one machine, one account, or
one person's notes — clone the repo, run one command, and the session works.

## What lives here

| File | What it is | When it's written |
|---|---|---|
| `vox-<nnn>-concepts.md` | Concept primer for a ticket: each concept it exercises, why it matters here, the pitfall | At ticket start, before implementation |
| `retros.md` | Per-ticket retro ledger: what was executed, deviations + why, numbers, lessons | At ticket close, after the board comment |
| `coach/` | Lesson pages served by the coach bridge — one per ticket or topic | With the primer |
| `coach/videos/` | Downloaded NotebookLM Video Overviews (**git-ignored**, mp4s stay out of the repo) | When a video is generated |

---

## 1 · Web coach — an interactive session with Claude

The coach is a ~55-line stdlib HTTP server (`tools/coach/server.py`) plus a static page. The
page shows the ticket's concept cards and a quiz; **DONE** bundles every answer into one
message; Claude grades it, argues with weak reasoning, and replies into the page's chat
panel. The chat panel also works on its own — type a doubt mid-lesson and Claude answers
there.

```bash
make coach                                  # serves docs/learning/coach/ on 127.0.0.1:8765
# open http://127.0.0.1:8765/vox-day1.html
# then, in a Claude Code session in this repo: "start web coach session"
```

That phrase matters: it triggers `.claude/skills/web-coach/SKILL.md`, which is the Claude
half of the protocol (reset state → serve → seed a greeting → arm a file watcher → grade →
reply → re-arm). You do not have to know the protocol; Claude reads the skill.

**Why a page rather than plain chat.** The quiz answers are graded against a key in the page,
so Claude spends its turn on the *reasoning* answers and the wrong picks instead of
administering a quiz. And the session leaves an artifact: what you got wrong lands in
`retros.md`, not in a scrollback nobody reads again.

Rules worth knowing:

- `coach/chat.jsonl` is session state. Git-ignored, reset per session
  (`: > docs/learning/coach/chat.jsonl`), never committed.
- The server binds `127.0.0.1` — local only, nothing exposed.
- New topic → write the primer first, then derive the page from it. A page that says
  something the primer does not is the failure mode this ordering prevents.

Full mechanics and the page contract: [tools/coach/README.md](../../tools/coach/README.md).

---

## 2 · Shared NotebookLM notebook

**Notebook:** _not created yet — first developer to need it does the one-time setup below,
then replaces this line with the link and tells the other person._

The notebook's sources are this repo's tracked docs: `ARCHITECTURE.md`, `evals/ENTITY_SPEC.md`
and the concept primers in this folder. With those loaded it is good for two things a repo is
bad at — asking "why is TTS local?" in plain English and getting a cited answer, and
generating short **Video Overviews** per topic that the coach pages embed.

### One-time setup (a human has to do this — there is no API key)

```bash
uv tool install "notebooklm-py[browser]"
notebooklm login                       # opens a browser, uses your Google session
notebooklm create "VOX — Course Concepts"
notebooklm list                        # copy the notebook id
```

Then, in the NotebookLM UI, **share the notebook with the other developer (editor access)** —
sharing is not scriptable — and record the id here plus in your shell:

```bash
export VOX_NOTEBOOK_ID=<id>            # the sync script reads this, nothing is hardcoded
```

### Syncing sources and pulling videos

[tools/coach/notebooklm_sync.py](../../tools/coach/notebooklm_sync.py) uploads the tracked
docs as sources and downloads Video Overviews straight into `coach/videos/`, using the
unofficial [notebooklm-py](https://github.com/teng-lin/notebooklm-py) client:

```bash
uv run --script tools/coach/notebooklm_sync.py --dry-run   # print the plan, touch nothing
uv run --script tools/coach/notebooklm_sync.py             # upload docs + build videos
uv run --script tools/coach/notebooklm_sync.py --skip-videos   # sources only, seconds not minutes
```

Re-running is idempotent: a source whose title is already in the notebook is skipped, because
NotebookLM has no "replace source" and a second copy of the same doc just competes in
retrieval. When a doc changes materially, delete the old source in the UI and re-run.

Adding a primer to the sync = one row in `SOURCE_DOCS`. Adding a video = one entry in
`VIDEOS`, keyed by the filename the page already probes for.

**Source hygiene — the one hard rule.** Only tracked docs go up. Recordings, transcripts,
`runs/`, and anything under `evals/heldout/` must never become a notebook source: that is how
a sealed case leaks out of the repo that seals it, and a leaked held-out case cannot be
un-leaked. The script's `SOURCE_DOCS` is an allow-list for exactly this reason — do not
replace it with a glob.

### Video index

| Topic | Primer | Video file |
|---|---|---|
| Turn loop, arms, telemetry, latency split | [vox-day1-concepts.md](vox-day1-concepts.md) | `coach/videos/vox-day1-pipeline-latency.mp4` |
| Fallback, cooldown, barge-in | [vox-day1-concepts.md](vox-day1-concepts.md) | `coach/videos/vox-day1-fallback-bargein.mp4` |

Update this table when a new video lands. Videos are optional — every page works without
them.

---

## 3 · Per-ticket loop

1. **Ticket start — primer.** Write `vox-<nnn>-concepts.md` before implementation: the
   concepts the ticket exercises, why each matters *here*, the pitfall. Grounded in the
   ticket spec and `ARCHITECTURE.md`.
2. **Offer a coach session** on that primer before coding — page if there is one, plain
   terminal Q&A if not. Skippable, never silent.
3. **Ticket close — retro.** Append to `retros.md`: executed, deviations + why, numbers with
   the command that produced them, lessons.
4. **Keep the notebook current.** New primer → add it to `SOURCE_DOCS` and re-run the sync;
   new video → add a row to the index above.
5. Primers and retros are committed on the ticket's branch. They are deliverables, not
   scratch — and they go through the same review as the code
   ([docs/CONTRIBUTING.md](../CONTRIBUTING.md)).
