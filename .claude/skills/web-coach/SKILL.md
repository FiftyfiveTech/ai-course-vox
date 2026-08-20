---
name: web-coach
description: Run an interactive coach learning session for a VOX ticket — serve the lesson page, watch the chat file, grade DONE submissions, reply into the page. Use when the user says "start web coach session", "coach session", "learning session", or at ticket start per the learning loop in docs/learning/README.md.
---

# web-coach — lesson page ↔ Claude bridge (VOX)

Static lesson page + stdlib server (`tools/coach/server.py`). The page POSTs answers and
questions to `chat.jsonl`; Claude watches the file, grades, replies via `tools/coach/say.py`;
the page polls `chat.jsonl` every 3s.

## Start a session

1. Serve dir = `docs/learning/coach/`. Reset state at session start only:
   `: > docs/learning/coach/chat.jsonl`
2. Start the server (background task; check `lsof -i :8765` first):
   `python3 tools/coach/server.py --dir docs/learning/coach --port 8765`
3. Smoke test: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/<page>.html`
   → 200, then POST `{"text":"ping"}` to `/send` → 204 (then reset `chat.jsonl` again).
4. Seed a greeting: `python3 tools/coach/say.py docs/learning/coach/chat.jsonl "<greeting>"`
5. Arm the watcher (background task — one-shot, exits on a new user message; **re-arm after
   every wake**):

```bash
CHAT=docs/learning/coach/chat.jsonl; LAST=$(grep -c '"role": "user"' "$CHAT" || true)
while :; do sleep 3; NOW=$(grep -c '"role": "user"' "$CHAT" || true)
  if [ "$NOW" -gt "$LAST" ]; then echo "NEW:"; grep '"role": "user"' "$CHAT" | tail -n $((NOW-LAST)); exit 0; fi
done
```

6. Tell the user the URL: `http://127.0.0.1:8765/<page>.html`

## On messages

- `tag:"chat"` — answer in the page chat via `say.py`. Short, concrete, coach tone.
- `tag:"session-done"` — grade the free-text answers, coach the wrong MCQ picks, challenge
  weak reasoning; append what was learned to `docs/learning/retros.md`; say what the next
  session covers. Reply into the page chat.

Ground every answer in this repo: `ARCHITECTURE.md` and the ticket primer are the source of
truth, and a measured number is quoted only with the command that produced it.

## Lesson pages

- Live in `docs/learning/coach/`; one page per ticket or topic. Template:
  `docs/learning/coach/vox-day1.html`.
- Contract: concept cards come from the ticket's primer
  (`docs/learning/vox-<nnn>-concepts.md`); quiz answers stay local until the DONE button
  bundles everything into ONE `tag:'session-done'` POST ending with explicit TASKS for
  Claude; the chat panel polls `/chat.jsonl` every 3s (Enter=send, Shift+Enter=newline);
  **dark mode always** (CSS-var palette, `html.dark` override, 🌙/☀️ toggle, `localStorage` +
  `prefers-color-scheme`); video cards are `fetch(src,{method:'HEAD'})`-gated so pages work
  before the videos exist.
- New topic page → write or update the ticket primer **first**; the page derives from it, not
  the other way round.

## Videos

Short NotebookLM Video Overviews from the shared notebook, downloaded into
`docs/learning/coach/videos/` (git-ignored) by
`uv run --script tools/coach/notebooklm_sync.py`. Notebook id comes from `--notebook` or
`$VOX_NOTEBOOK_ID` — never hardcode it. Setup and source-hygiene rules:
`docs/learning/README.md`.

## Notes

- `chat.jsonl` is append-only session state; never commit it.
- The server binds 127.0.0.1 — local only, nothing exposed.
- Primers, retros and pages are committed on the ticket's branch and reviewed like code
  (`docs/CONTRIBUTING.md`).
