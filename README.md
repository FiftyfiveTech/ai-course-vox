# VOX — an internal voice assistant, built by hand

**Track:** FiftyFive AI engineering course, Week 3 · **Status:** MVP freeze (`mvp-v1`)

VOX listens on a microphone, answers out loud, and answers *from the company's own PDF policies*
when the question is covered by them — citing the `document:page` it read. Employee-facing and
internal by design: no customer PII, no recording without a visible consent notice, and a
confirmation read-back before anything that would write.

```
Mic → [VAD] silero → [STT] whisper-large-v3-turbo → [retrieval] BM25 + bge-small
                                                          ↓
        Speaker ← [TTS] Kokoro-82M ← [LLM] gpt-oss-120b ← grounded prompt, or plain reply
```

Everything is a swappable **arm** named by its Hugging Face repo id, every model call is logged with
its cost and latency, and every phase is closed by a gate that prints a number. The whole thing runs
on free tiers and local weights: **`cost_usd` is 0.0 on every line of `runs/calls.jsonl`.**

---

## Developer note — read this before you clone

Four things about this repo will surprise you, and all four are deliberate:

1. **The document corpus is not in git.** `sources/` is gitignored — those are internal FiftyFive HR
   policy PDFs, and the text extracted from them is the same disclosure as the PDFs themselves. A
   clean clone therefore has **no knowledge base**. Everything else works; every grounded command
   (`make index`, `ask`, `answer`, `ground`, `gate-poc`) needs you to put your own PDFs in
   `sources/` first. `make demo` announces that at startup rather than quietly answering from the
   model's own memory.
2. **`uv.lock` is not tracked either**, so `uv sync` re-resolves on your machine. That is a known
   gap in the reproducibility claim — see [Known gaps](#known-gaps-and-gate-debt).
3. **This is a hand-built course repo.** Nothing here was generated wholesale; every module is small
   enough to explain, and `ARCHITECTURE.md` records *why* each decision was made and what it
   measured. When this README and `ARCHITECTURE.md` disagree, `ARCHITECTURE.md` wins — the numbers
   below are copies, and each one names the section it was copied from.
4. **Latency varies 2–4× run to run on identical input.** Every number here is a range or a median
   with its command attached. If you get a different number, you are probably not looking at a
   regression. Read [What to expect](#what-to-expect) before you tune anything.

Tasks come from the Odoo board (project *AI Dev Learning*) via the `odoo-board` MCP server, not from
this README. The working contract — blind labelling, no self-merges, measured gates, zero spend — is
`CLAUDE.md`.

---

## What to expect

**It is not fast, and the repo does not pretend otherwise.** The signed-off budget was 2 s end to
end. Measured, it is not met, by a lot:

| | measured | notes |
|---|---|---|
| `time_to_first_audio`, fixture-driven | **5.6 – 20.0 s** | n=4, `scripts/turn_from_fixture.py`, 2026-08-17 |
| best single turn of six, whole-architecture comparison | **6.2 s** | `make compare`, n=3 per arm, 2026-08-20 |
| add for a live microphone | **~+1.1 s** | the VAD hangover a fixture run collapses to ~4 ms |

The largest stage is TTS in every single turn measured so far, and the three model calls are not the
problem. `make compare` found the biggest available lever: `rhasspy/piper-voices` is **9–13× cheaper
per character** than `hexgrad/Kokoro-82M` (7.5–9.5 ms/char against 87–103 ms/char) — a config change,
not a rewrite. The full argument and every number: § *Latency budget* and § *Architecture comparison*
in `ARCHITECTURE.md`.

**Wear headphones.** There is no acoustic echo cancellation and none is in scope. On open speakers
silero hears Kokoro and the reply interrupts itself, every time — speaker bleed is real speech as far
as a VAD is concerned.

**The free tiers will refuse you sooner or later.** When Groq or NVIDIA NIM rate-limits, times out or
5xxs, that stage runs its **local** arm instead of losing the turn — loudly, with both attempts in
`runs/calls.jsonl` and the arm that actually ran named on the turn record. A 429 also parks that arm
for its `Retry-After` window, so the next turn does not pay another doomed round-trip. A **bad key or
a missing credential deliberately does not fall back**: that is a config bug, and hiding it behind a
worse transcript is worse than stopping.

**A fallback is not free and not identical.** Measured, n=1, 2026-08-19: the local STT arm returns
`'So this is testing.'` where the Groq arm returns `'Hello. So this is testing.'` — it drops the word
that starts the turn. And on the grounded path the local 3B took **72.6 s** for a 1381-token prompt
against 0.94 s remote. A rate limit does not cost the grounded turn's quality, it costs the turn.

**What it does well:** it stays up. Both free tiers unreachable still produced a spoken answer; the
session survives silence and Ctrl-C; you can talk over a reply and the words that interrupted it
become the next turn's input; and a question the corpus does not cover is **refused** rather than
invented — including the case where retrieval returns confident-looking chunks that do not contain
the answer.

---

## System requirements

| | |
|---|---|
| OS | Linux, macOS or Windows. Developed on Windows 11 + Git Bash and on Linux. The Makefile is POSIX `sh`, so on Windows use Git Bash or WSL, or run the `uv run …` lines directly |
| Python | **3.12** — pinned in `.python-version`. `>=3.10,<3.13`: torch has no 3.14 wheels, and both `silero-vad` and `kokoro` need torch |
| CPU | Any x86-64 or Apple Silicon. **No GPU needed** — every local arm is CPU-int8 or small enough not to care |
| RAM | ~4 GB free for the default arms. ~8 GB if the local LLM fallback (a 3B at Q4_K_M) runs alongside them |
| Disk | ~6 GB: venv ~2.5 GB (torch dominates) plus ~2.8 GB of model weights if you pull everything |
| Audio | A microphone **and** a speaker for `make demo` / `make barge` / `make compare`. `time_to_first_audio` is not measurable without an output device actually pulling samples, so `--silent` deliberately fails the five-stage criterion. `make turn` and `make ground` need no mic — they drive the same loop from a recording |
| Network | Needed for the two remote stages and for the first-run weight downloads. After that the local arms, indexing and retrieval run offline |

**First-run downloads**, pulled by `make setup` or by an explicit target so none of them ever lands
inside something being timed:

| what | size | pulled by | required? |
|---|---|---|---|
| `hexgrad/Kokoro-82M` + `en_core_web_sm` | ~350 MB | first `make demo` | yes — the default TTS arm |
| Llama-3.1 tokenizer mirror (`config.TOKENIZER_REPO`) | ~9 MB | `make tokenizer` | yes, for `make index` |
| `BAAI/bge-small-en-v1.5` | ~130 MB | `make encoder` | degrades retrieval if absent |
| `Systran/faster-whisper-base` | ~150 MB | first fallback or `make arms` | needed for the STT fallback |
| `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M` | ~2 GB | `make fallback-model` (ollama) | degrades the LLM stage if absent |
| `rhasspy/piper-voices` voice | ~65 MB | first `make compare` | only for `make compare` |
| `openai/whisper-base`, `microsoft/speecht5_tts` | ~1.1 GB | first `make arms` | only for `make arms` |

---

## Software needed

**Required:**

- [`uv`](https://astral.sh/uv) — the only package manager used here. `make setup` refuses without it.
- Python 3.12 — uv will fetch it if it is not already installed.
- `make` and `git`. On Windows: Git Bash or WSL.
- A working PortAudio device pair (mic + speaker) for anything end to end. `sounddevice` ships
  PortAudio on all three platforms; on a bare Linux box you may also need `libportaudio2`.

**Optional — the system runs without it, one stage gets worse, and it says so:**

| | what you lose | how to get it |
|---|---|---|
| [`ollama`](https://ollama.com/download) plus the 3B pull | **no local LLM fallback.** A 429 or a timeout then costs the turn instead of costing its quality. `make setup` warns rather than fails | `make fallback-model` |
| `BAAI/bge-small-en-v1.5` weights | the **dense half of retrieval**. BM25 answers alone — a worse ranking, not a wrong one, and exactly VOX-030's measured behaviour | `make encoder` |
| `GROQ_API_KEY` / `NVIDIA_API_KEY` | the remote arms. Note this is *not* a graceful degrade: a **missing** credential is a STOP-and-ask, not a fallback | see [Sample env](#sample-env) |

**Optional — blocks exactly one command:**

| | needed for | note |
|---|---|---|
| `libespeak-ng` | `scripts/gen_utterances.py`, which regenerates the 45 dev / held-out utterance WAVs | the WAVs are gitignored by `*.wav`, so **`gate_phase0`, `gate_phase1` and `gate_phase1b` cannot run on a clean clone until they are regenerated**. `evals/dev/manifest.json` is tracked, so the corpus itself is not lost |
| your own PDFs in `sources/` | `make index`, `ask`, `answer`, `ground`, `gate-poc`, and any grounded reply in `make demo` | gitignored on purpose — see the developer note |

---

## Getting started

```bash
git clone https://github.com/FiftyfiveTech/ai-course-vox
cd ai-course-vox
make setup          # uv sync, write .env from .env.example, pull the tokenizer, encoder and 3B
# fill in .env — see Sample env below
make test           # 337 passed in 9.09s (2026-08-21). No network, no key, no mic
make turn           # one instrumented turn from a recording. Needs a speaker, no mic
make dry-run        # the scripted ten-turn session, asserted turn by turn. Needs a speaker, no mic
make demo           # talk to it. Needs a mic and headphones. Ends on the clock or on Ctrl-C
```

`make setup` **fails** without `uv` and without the tokenizer, and **warns** without ollama and
without the sentence encoder — that is the required / optional split above, encoded in the recipe.

To get the grounded path as well, add a knowledge base:

```bash
cp /path/to/your/policies/*.pdf sources/
make index          # sources/*.pdf -> runs/chunks.jsonl + runs/embeddings.npz, prints the counts
make ask    Q="how many casual leaves am I entitled to in a year"   # retrieval only, no model
make answer Q="how many casual leaves am I entitled to in a year"   # the same chunks, answered
make ground                                                         # a grounded turn, no mic
```

`make index` is offline: `pypdf` parses, the tokenizer only counts, and the encoder runs locally.
Pages that yield **no** text are printed by name — on a scanned PDF that is the whole corpus, and it
has to be visible at load rather than as an empty answer three tickets later.

`make help` lists every target. The ones worth knowing:

| | |
|---|---|
| `make demo` | a real conversation, `VOX_SESSION_MINUTES` long (default 3), grounded where the corpus covers it |
| `make dry-run` | the scripted ten-turn demo session end to end, no mic: a barge-in, two confirmations, a PASS/FAIL per turn and a timing table. ~2 min, because the frames are paced at real time |
| `make barge` | three turns with interruptible replies — talk over it, it prints the stop latency |
| `make turn` / `make ground` | one turn from a recording — the plain and the grounded path, no mic |
| `make arms` | call every registered arm once, print the `calls.jsonl` lines. Fallback off, so a refusal shows on its own row |
| `make compare` | two whole architectures as real turns, five-stage split for both |
| `make floors` | re-measure both retrieval floors on `evals/dev` and print the gaps |
| `make gate-poc` | the POC gate: correct-source@3, refusal rate, grounded-answer rate |
| `make coach` | serve the interactive lesson pages on `127.0.0.1:8765` |

---

## Sample env

Real keys live in `~/.config/`, **never in the repo**. `.env` is gitignored; `.env.example` carries
names only, and `make setup` copies it for you. The free tiers are free — if anything ever asks for
a card, that is a STOP-and-ask, not a judgement call.

Below is every knob, with the default `src/config.py` uses. In `.env.example` the tunables are
**commented out on purpose**: `config.py` is the single source of defaults, and an uncommented copy
in `.env` would silently override a default that later changes. Uncomment a line only when you mean
to change it.

```bash
# --- credentials (no defaults; a MISSING one stops the turn, it does not fall back) ------------
HF_TOKEN=                       # only for gated repos; the pinned tokenizer mirror needs none
GROQ_API_KEY=                   # STT default + the LLM default and 2 more  https://console.groq.com
NVIDIA_API_KEY=                 # the gpt-oss-nim arm only               https://build.nvidia.com

# --- arm selection: alias, HF repo id, or repo/id@provider. A CLI flag beats the env ----------
VOX_STT_MODEL=                  # default openai/whisper-large-v3-turbo     (alias: turbo)
VOX_LLM_MODEL=                  # default openai/gpt-oss-120b@groq          (alias: gpt-oss)
VOX_TTS_MODEL=                  # default hexgrad/Kokoro-82M                (alias: kokoro)
VOX_EMBED_MODEL=                # default BAAI/bge-small-en-v1.5            (alias: bge-small)

# --- session and audio ------------------------------------------------------------------------
VOX_SESSION_MINUTES=3           # how long `make demo` listens
VOX_SESSION_QUIET_LIMIT=2       # consecutive silent listens before the session ends itself
VOX_TTS_VOICE=af_heart          # Kokoro voice
VOX_STT_LANGUAGE=en
VOX_STT_BIAS=0                  # 1 enables Whisper vocabulary biasing via initial_prompt (VOX-021)

# --- timeouts ---------------------------------------------------------------------------------
VOX_REMOTE_TIMEOUT_S=10         # a hosted arm's budget before the turn falls back to local
VOX_LOCAL_TIMEOUT_S=120
VOX_OLLAMA_HOST=                # default http://localhost:11434 — no key, nobody is billed

# --- knowledge base: chunking (re-run `make index` after changing any of these) ---------------
VOX_SOURCES_DIR=                # default ./sources
VOX_CHUNKS_FILE=                # default ./runs/chunks.jsonl
VOX_EMBEDDINGS_FILE=            # default ./runs/embeddings.npz
VOX_CHUNK_TOKENS=300
VOX_CHUNK_OVERLAP_TOKENS=50
VOX_TOKENIZER_REPO=NousResearch/Meta-Llama-3.1-8B-Instruct   # mirror of the gated 3.1 tokenizer
VOX_EMBED_BATCH=16

# --- retrieval: the two measured floors and the fusion. Re-run `make floors` if you touch them -
VOX_HYBRID_RETRIEVAL=1          # 0 = BM25 only, i.e. VOX-030's behaviour
VOX_RETRIEVAL_TOP_K=5           # k=5 is ~5.1x the plain prompt; this is the knob that moves first
VOX_RETRIEVAL_SCORE_FLOOR=0.28  # calibrated: answerable min 0.320 vs absent max 0.234
VOX_DENSE_SCORE_FLOOR=0.65      # chosen, NOT derived — the dense column does not separate
VOX_RRF_K=60
VOX_FUSION_CANDIDATES=20
VOX_BM25_K1=1.5
VOX_BM25_B=0.75
VOX_BM25_EPSILON=0.25
```

Every tunable is an env var read in `src/config.py`, never a Makefile variable — so a value can be
changed for one run without editing a recipe. VAD and barge-in thresholds are the exception and live
in `config.yaml` (VOX-012), because `scripts/tune_vad.py` measures their effect on the dev set.

---

## Models — HF repo ids only

The provider is only *where it runs*. The mapping from repo id to each provider's own model string
lives in `src/config.py` and nowhere else. Two arms sharing a `backend` share an adapter, so a new
arm on an existing runtime is a table row and no new code.

| Stage | HF repo id | Runs on | Backend | Alias |
|---|---|---|---|---|
| VAD | `snakers4/silero-vad` | local | — | not an arm |
| STT | `openai/whisper-large-v3-turbo` | Groq free tier | `openai-audio` | `turbo` **(default)** |
| STT | `openai/whisper-large-v3` | Groq free tier | `openai-audio` | `large-v3` |
| STT | `openai/whisper-base` | local | `transformers-whisper` | `whisper-base` |
| STT | `Systran/faster-whisper-base` | local | `faster-whisper` | `faster-base` **(fallback)** |
| LLM | `openai/gpt-oss-120b` | Groq free tier | `openai-chat` | `gpt-oss` **(default)** |
| LLM | `openai/gpt-oss-120b` | NVIDIA NIM free tier | `openai-chat` | `gpt-oss-nim` |
| LLM | `Qwen/Qwen3.8-27B` | Groq free tier | `openai-chat` | `qwen3.8` |
| LLM | `openai/gpt-oss-20b` | Groq free tier | `openai-chat` | `gpt-oss-20b` |
| LLM | `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF` | local, via ollama | `ollama-chat` | `llama-3.2-3b` **(fallback)** |
| TTS | `hexgrad/Kokoro-82M` | local | `kokoro` | `kokoro` **(default)** |
| TTS | `microsoft/speecht5_tts` | local | `speecht5` | `speecht5` **(fallback)** |
| TTS | `rhasspy/piper-voices` | local | `piper` | `piper` |
| Embed | `BAAI/bge-small-en-v1.5` | local | `transformers-embed` | `bge-small` **(default)** |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` | local | `transformers-embed` | `minilm` |

The two `meta-llama/Llama-3.1-*-Instruct` arms that used to head this stage are gone: NVIDIA NIM
retired both on **2026-08-26** and now answers `410 Gone` with that date in the body. Latency tables
elsewhere in this file that name them were measured before that and are kept as the record of what
was measured, not as a claim about arms you can still call. `make preflight` is what checks this
table against the two catalogues — run it before a demo, not after one fails.

`ollama` is a provider of its own rather than `local` because the weights are here and it still
speaks HTTP, to a daemon on `localhost:11434`. `Arm.local` is the attribute that answers "are the
weights here", and it is what `config.PIPELINE` and `config.FALLBACKS` are checked against.

```bash
make arms                                     # one real call per arm, then the log lines
uv run python scripts/check_arms.py --list    # the table, no calls
uv run python -m src.loop --stt openai/whisper-base --tts microsoft/speecht5_tts
uv run python scripts/turn_from_fixture.py --llm gpt-oss    # alias, repo id, or repo/id@provider
VOX_STT_MODEL=faster-base make turn                        # env sets the default; the flag wins
```

`stt_model` / `llm_model` / `tts_model` are on every `runs/turns.jsonl` line, so two runs with
different arms cannot be quietly averaged. The measured cost of each arm, and what the local ones
get wrong: § *Models* in `ARCHITECTURE.md`.

### Where each stage runs, and why

`config.PIPELINE` declares the placement and `tests/unit/test_fallback.py` asserts each stage's
default arm against it — so reordering an arm table can no longer move a stage across the network
boundary unnoticed.

| Stage | Placement | Why |
|---|---|---|
| VAD | local | runs per 32 ms frame; a network hop per frame is not a design |
| STT | remote | both local `base` arms drop the first word of the fixture — measured |
| LLM | remote | the widest quality gap of the five, and the least tolerable to lose |
| TTS | local | no key, no quota, and Kokoro is already good enough to ship |
| Embed | local | once per chunk at index time, once per query — and the corpus is internal |

---

## The gating table

A phase is done when its gate **prints a number**. Every gate asserts `cost_usd == 0.0` and a HF
repo id on every call it made.

| Gate | Command | Prints | Floor | Status |
|---|---|---|---|---|
| Phase 0 (VOX-010) | `uv run python tests/gates/gate_phase0.py` | the 5-way latency split for one fixture turn | `t_stt`, `t_llm`, `t_tts` all non-null and positive | **passed** — needs the dev WAVs regenerated on a clean clone |
| Phase 1 (VOX-017) | `uv run python tests/gates/gate_phase1.py` | 10 scripted turns; per-stage latency for every arm; barge-in stop latency | 10/10 turns, every arm logged | **passed** — same WAV caveat |
| Phase 1B (VOX-023 / VOX-024) | `uv run python tests/gates/gate_phase1b.py` | entity capture rate, confirmation rate on write intents, state validity, per-category breakdown — over `evals/heldout/` | **none yet.** The script prints `PASS — numbers printed`; VOX-024 is the ticket that sets the threshold and is still open | **script written, threshold not set** — see [Known gaps](#known-gaps-and-gate-debt) |
| POC / PDF (VOX-033) | `make gate-poc` | correct-source@3, refusal rate, grounded-answer rate and its intersection with correct-source | `≥ 7/8` and `refusal = 2/2`; exits non-zero below | **passed** |
| Leakage (task 0.7) | `tests/gates/test_no_leakage.py` | — | `dev ∩ heldout = ∅` by content hash | **not written** |
| Unit suite | `make test` | pass count | all pass | **337 passed in 9.09s**, 2026-08-21 |
| Demo rehearsal (VOX-026) | `make dry-run` | per-turn PASS/FAIL against `evals/demo/session_v1.json`, and the five-field split for all ten turns | every turn meets its own expectation; exits non-zero otherwise. **No latency floor** — VOX-003's budget is missed by 4x and is recorded as missed, so a floor here would be a lie or a permanent failure | **clean twice**, 2026-08-21 |

**`make gate` currently runs nothing.** It is `pytest tests/gates`, and these gates expose `main()`
rather than `test_*` functions, so pytest collects zero tests and exits 5. Run each gate by the
command in the table. Recorded as debt below rather than papered over at freeze time.

### The POC gate's numbers

`make gate-poc` over `evals/dev/pdf_queries.json` — ten written queries, eight with an expected
`file:page`, two the corpus does not cover. Measured 2026-08-21 on `meta-llama/Llama-3.1-8B-Instruct`
@ NVIDIA NIM, identical across two consecutive runs:

| number | measured | floor | asserted? |
|---|---|---|---|
| correct-source@3 | **7/8 = 0.875** | 0.875 | yes |
| refusal rate on the 2 absent | **2/2 = 1.000** | 1.000 | yes |
| grounded-answer rate | **8/8 = 1.000** | — | no |
| …grounded *on an expected source* | **7/8 = 0.875** | — | no |
| paraphrased refusals caught | **0** | — | no |

19 model calls, `cost_usd = 0.0` on every one. **The last two rows are the finding:** `grounded` only
says the model answered from the excerpts it was handed — it is true even when those excerpts came
off the wrong pages. So the rate is printed with the intersection under it, and the intersection is
the honest reading. Dev-only by construction: `heldout-v1` is sealed and holds zero document queries,
and it was not reopened for this. Full discussion: § *The POC gate* in `ARCHITECTURE.md`.

---

## The latency budget table

Target was **< 2 s** end to end. It is not met, and the budget column is kept as written so the size
of the miss stays visible rather than being edited away.

| Stage | Budget | Measured (n=4) | Notes |
|---|---|---|---|
| VAD (end-of-speech) | ~100 ms | **~1100 ms live** (by construction) | `VAD_SILENCE_MS` — the loop holds the turn open this long to see whether you are done. Silero compute is 0.15–0.83 s on top |
| STT — `openai/whisper-large-v3-turbo` @ Groq | ~300 ms | **1.0 – 15.6 s** | free tier, wildly variable; the 15.6 s is provider time in the call log |
| Retrieval — BM25 + `bge-small` | not budgeted | **83 – 117 ms** | ~1 ms lexical; the rest is one encoder forward pass on CPU |
| Reply — `Llama-3.1-8B` @ NIM | ~500 ms | **0.53 – 1.22 s** | the grounded prompt is 5.1× the plain one and still answers inside a second |
| Confirmation TTS + user response | not counted | not in this split | VOX-020 is built |
| Action (internal API) | ~200 ms | not built | — |
| TTS — warm `hexgrad/Kokoro-82M` | ~200 ms | **3.1 – 12.4 s** | **15–60× over budget**, and the largest stage in every turn measured |
| unattributed | — | **0.25 – 0.84 s** | opening the output device before the first block. Real, felt, and belongs to no model call |
| **Total (no confirmation)** | **~1.3 s** | **5.6 – 20.0 s** | `time_to_first_audio`, fixture-driven |

Measured 2026-08-17: four turns from `tests/fixtures/hello_testing_voice.mp3` via
`scripts/turn_from_fixture.py`, same clip and same machine. Two warnings about reading it:

- **n=4 is a range, not a distribution.** Every stage varies 2–4× across four runs of *identical*
  input. The variance is the finding; no single number here is a target to tune against.
- **Fixture runs understate the live figure.** Frames are pushed as fast as the CPU allows, so the
  ~1.1 s VAD hangover a person waits out collapses to 3–26 ms. Read a live turn as roughly
  *measured + 1.1 s*. `source` on each turn record says which kind of run produced it.

Where the 2 s would have to come from, on this evidence: TTS, then the VAD hangover. `make compare`
says the cheapest available win is the piper arm (9–13× cheaper per character than Kokoro) rather
than streaming Kokoro's first chunk. Full tables: § *Latency budget* and § *Architecture comparison*
in `ARCHITECTURE.md`.

### Telemetry

Two logs, joined by `turn_id`, because they answer different questions:

| | one line per | answers |
|---|---|---|
| `runs/calls.jsonl` | model call | what it cost, how long the provider took |
| `runs/turns.jsonl` | turn | where the wall clock went — `t_vad`, `t_stt`, `t_llm`, `t_tts`, `t_retrieval_ms`, `time_to_first_audio`, `grounded`, `sources` |

`time_to_first_audio` is measured from the **last frame silero called speech** through to the moment
the output device pulls its first block — the user has been waiting since they stopped talking, so
the VAD hangover is inside the number.

---

## The knowledge base, in one paragraph each

**Index** (`make index`, VOX-029). Every PDF page by page, cut into 300-token windows with 50 tokens
of overlap, each carrying its `doc_id` and `page`. **Chunks never span a page**, so the provenance is
exact rather than approximate. `doc_id` is the filename stem, and the filenames are the documents'
own titles — "leave-policy page 4" is an answer, a hash is not. Tokens are counted with the tokenizer
of the model that will read the chunks. Measured on the internal corpus, 2026-08-20: 15 files, 184
pages (163 with text, **21 with none, named individually**), 215 chunks, 40,750 tokens.

**Retrieve** (`make ask`, VOX-030 + hybrid). BM25 over those chunks, fused **by rank** with a cosine
from `BAAI/bge-small-en-v1.5` — never by score, because adding a normalised fraction to a cosine
means inventing an exchange rate and then tuning it. A chunk is kept if **either** half vouches for
it; a refusal needs both to miss. When BM25's own best chunk is under the floor it **abstains from
the ordering** rather than voting with noise. The lexical floor is calibrated (answerable min 0.320
against absent max 0.234); the dense one is *chosen and says so*, because no dense signal separates
answerable from absent.

**Answer** (`make answer`, VOX-031/032). The chunks go to `arms.llm()` — same cost logger, same
`--llm` flag, same cooldown, same local fallback — with `prompts/answer_from_source_v2.md`. Three
ways it declines, and the output says which: **no chunk cleared the floor** → refused with *no model
call at all*; **chunks cleared it but do not contain the answer** → the model refuses; **the reply
states a figure that appears in no excerpt** → `answer.ungrounded_numbers()` suppresses it and the
same refusal is spoken. All three say the *same* sentence out loud, so a listener cannot hear which
path ran. The grounded path samples at **temperature 0** — reading five policy excerpts is not a task
where variety is a feature.

Citations are the provenance of the **context**, not a token the model emitted, so there is no format
for it to get wrong. That also means `sources` says what the answer was *grounded in*, up to five
chunks — not which sentence it came from.

---

## Layout

| Path | Holds |
|---|---|
| `src/` | the system — small modules, one job each. `arms.py` is the only model interface |
| `prompts/` | versioned prompt files (`answer_from_source_v1.md`, `_v2.md`, …). Never inlined in code |
| `schemas/` | Pydantic models. Structured output is validated, not parsed by hand |
| `scripts/` | one-command drivers: `turn_from_fixture`, `check_arms`, `compare_arms`, `build_index`, `ask`, `tune_vad`, `measure_biasing` |
| `evals/dev/` | **Builder** tunes here — 15 utterance cases plus the retrieval and PDF query sets |
| `evals/heldout/` | **Evaluator** only. Sealed as `heldout-v1`; the Builder never reads it |
| `tests/unit/` | the failure modes, as tests. `make test` |
| `tests/gates/` | one script per phase gate. It prints the number; the number decides |
| `sources/` | the PDF corpus. **Gitignored** — internal documents |
| `runs/` | `calls.jsonl`, `turns.jsonl`, `chunks.jsonl`, `embeddings.npz`. **Gitignored** |
| `config.yaml` | VAD and barge-in thresholds (VOX-012), measurable via `scripts/tune_vad.py` |
| `ARCHITECTURE.md` | why every decision was made, with the measurement that decided it |
| `STANDUP.md` | daily log, append-only, two minutes |
| `docs/learning/` | a concept primer per ticket, the retro ledger, and the coach lesson pages |
| `notes/build-log/VOX/` | the week report and ticket-planning notes |
| `docs/CONTRIBUTING.md` | branches, PRs, review, merge — read before your first PR |

### Held-out seal

`evals/heldout/labels.json` — 30 gold labels, sealed as tag `heldout-v1`.
SHA-256 `030ca138283223f8d004071c7c92ed4343ff66b0ddffb0497c2eb59faa9438f9`.

---

## Known gaps and gate debt

Recorded here at freeze rather than discovered later. Full accounting, with what each one would have
caught: `notes/build-log/VOX/week-report.md`.

| gap | consequence | fix |
|---|---|---|
| `tests/gates/test_no_leakage.py` was never written (task 0.7) | `dev ∩ heldout = ∅` is asserted **nowhere**. Blind labelling holds because two people were careful — a different guarantee from a hash comparison | write it; it needs no key, no audio and no corpus |
| `gate_phase1b.py` prints numbers but asserts **no threshold** (VOX-024 still `inProgress`) | it exits PASS unconditionally, so Phase 1B has a report and not a gate. It also reads `evals/heldout/`, so it is Evaluator-run only | VOX-024: set the floor from the printed baseline |
| `make gate` collects nothing (exit 5) | the four gates that *do* exist have no single command, so nobody re-runs them | give the gates `test_*` wrappers, or make the target invoke the four scripts |
| `uv.lock` is gitignored | `uv sync` re-resolves per machine, so "a clean clone reproduces the demo" is true of the code and not of the dependency graph | track the lock file |
| `evals/dev/*.wav` and `evals/heldout/*.wav` gitignored by `*.wav` | `gate_phase0`, `gate_phase1` and `gate_phase1b` all fail on a clean clone until `scripts/gen_utterances.py` runs, and that needs `libespeak-ng` | add the regeneration step to `make setup`, or track the WAVs |
| 17 of 23 merged PRs were **self-merged** (`gh pr list --state merged --json author,mergedBy`) | the review rule was followed for the first week and then stopped being followed. `main` is protected; `dev` is not | branch protection on `dev` requiring one approving review — a rule a retro has to check by hand is a rule that decays |
| `sources/` gitignored (correctly) | no grounded path on a clean clone, and none of the POC numbers is reproducible by a stranger | unavoidable as it stands; a redistributable sample corpus would make the POC gate portable |
| no acoustic echo cancellation | on open speakers the reply interrupts itself, every time | out of scope: a webrtc/speexdsp dependency and its own ticket. **Demo on headphones** |
| the latency target is unmet | 5.6–20.0 s against a 2 s budget | the piper arm is the measured cheapest lever, then the VAD hangover |

---

## Learning: web coach and NotebookLM

Every ticket ships a concept primer next to the code, and two tools make them usable by someone who
did not write the ticket:

```bash
make coach     # serve the lesson pages on 127.0.0.1:8765
# open http://127.0.0.1:8765/vox-day1.html, then in Claude Code: "start web coach session"
```

The page shows the concept cards and a quiz; **DONE** sends every answer to Claude in one message,
and Claude grades, argues back, and replies into the page's chat panel. The same page embeds short
NotebookLM Video Overviews when they have been downloaded, and the shared notebook answers "why is
TTS local?" in plain English with citations back to these docs. Setup, the sync script, and the rule
about which files may never become a notebook source:
[docs/learning/README.md](docs/learning/README.md).

---

## Contributing

Branch off `dev`, PR against `dev`, **the other developer reviews and merges** — never yourself.
Full procedure, PR template and recovery steps: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Rules that live in this repo

`CLAUDE.md` carries the contract. The short version:

- Models and datasets are named by **Hugging Face repo id**. The provider is only where it runs.
- **Zero spend.** A paid call is a STOP-and-ask, never a judgement call.
- A phase is done when its gate **prints the number**, not when the code looks right.
- Report numbers with the command that produced them. If a number is not in this session's output,
  say so instead of quoting it.
- Every PR is reviewed by the other person. `main` is protected; self-merges are the one thing the
  Friday retro always checks.
