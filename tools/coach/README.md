# Coach — interactive learning sessions

A tiny stdlib-only bridge (no deps) between a lesson HTML page and Claude Code. The page
serves concept cards and a quiz; your answers POST to a chat file; Claude grades them and
replies into the page's chat panel. Nothing is exposed to the network — the server binds
`127.0.0.1`.

## Run a session (any machine, Python 3)

```bash
# 1. serve the lesson pages, from the repo root
make coach                  # or: python3 tools/coach/server.py --dir docs/learning/coach --port 8765

# 2. open a page
#    http://127.0.0.1:8765/vox-day1.html      (turn loop, telemetry, fallback, barge-in)

# 3. in a Claude Code session in this repo, say: "start web coach session"
#    Claude arms a watcher on docs/learning/coach/chat.jsonl, grades your DONE
#    submission, and coaches in the page's chat panel.
```

Answer the quiz, then hit **DONE** — everything is bundled into one submission. The chat
panel on the right works at any time for free-form questions; Enter sends, Shift+Enter is a
newline.

`chat.jsonl` is per-session state (git-ignored). Reset it between sessions:
`: > docs/learning/coach/chat.jsonl`.

## How it works

| Piece | Job |
|---|---|
| `server.py` | Serves `--dir`, plus `GET /chat.jsonl` and `POST /send`. Appends `{role:"user", text, tag}` lines. |
| `say.py` | Appends a `{role:"coach", …}` line — how Claude replies into the page. |
| the page | Polls `/chat.jsonl` every 3 s. Quiz answers stay local until DONE bundles them into one `tag:"session-done"` POST. |
| `.claude/skills/web-coach/SKILL.md` | The Claude side of the protocol: watcher, grading, where results get written. |

## Claude side

- Watcher (Claude runs it as a background task; it exits when you send something):

```bash
CHAT=docs/learning/coach/chat.jsonl; LAST=$(grep -c '"role": "user"' "$CHAT" || true)
while :; do sleep 3; NOW=$(grep -c '"role": "user"' "$CHAT" || true)
  if [ "$NOW" -gt "$LAST" ]; then grep '"role": "user"' "$CHAT" | tail -n $((NOW-LAST)); exit 0; fi
done
```

- Reply into the page: `python3 tools/coach/say.py docs/learning/coach/chat.jsonl "text"`

## Topic videos

Lesson pages embed short NotebookLM Video Overviews from `docs/learning/coach/videos/`
(git-ignored — mp4s do not belong in the repo). Scripted path, which uploads the primers as
notebook sources and downloads the videos under the exact filenames the pages probe for:

```bash
uv tool install "notebooklm-py[browser]"
notebooklm login                                              # once; opens a browser
export VOX_NOTEBOOK_ID=<id>                                   # see docs/learning/README.md
uv run --script tools/coach/notebooklm_sync.py --dry-run
uv run --script tools/coach/notebooklm_sync.py
```

Manual path (same result): open the shared notebook (link in
[docs/learning/README.md](../../docs/learning/README.md)) → generate/download the Video
Overview → save it as the filename the page expects (day 1:
`vox-day1-pipeline-latency.mp4`, `vox-day1-fallback-bargein.mp4`). The video cards appear
on their own once the files exist; the pages work fine without them.

## Adding a session page

Copy `docs/learning/coach/vox-day1.html` as the template, and write the ticket's primer
(`docs/learning/vox-<nnn>-concepts.md`) **first** — the page derives from the primer, not the
other way round. Contract to keep:

- Answers stay local until the DONE button bundles them into one `tag:'session-done'` POST
  that ends with explicit TASKS for Claude.
- Chat panel polls `/chat.jsonl` every 3 s.
- Dark mode always: CSS-var palette, `html.dark` override, 🌙/☀️ toggle, `localStorage` +
  `prefers-color-scheme`.
- Video cards are `fetch(src,{method:'HEAD'})`-gated so the page works before videos exist.
