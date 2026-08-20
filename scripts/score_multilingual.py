"""Transcribe and entity-score the 5 multilingual (Hindi) utterances.

Runs each wav through openai/whisper-large-v3-turbo (Groq free tier) with
language=hi, then scores intent against gold labels using src/scorer.py.

At this stage (pre-VOX-019), structured entity extraction is not yet wired in,
so scoring covers intent only. Full entity scores will be added in VOX-023.

Prints a capture-rate table per utterance and overall.

Usage:
    .venv/bin/python scripts/score_multilingual.py
"""
import io
import json
import sys
from pathlib import Path

import httpx
import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_STT, SAMPLE_RATE                  # noqa: E402
from src.scorer import score_dataset                             # noqa: E402
from src.telemetry import new_turn_id                            # noqa: E402

STT = DEFAULT_STT

MANIFEST = REPO_ROOT / "evals" / "multilingual" / "manifest.json"
LABELS   = REPO_ROOT / "evals" / "multilingual" / "labels.json"

INTENT_KEYWORDS = {
    "book_meeting": ["meeting", "book", "schedule", "call", "reunion", "réunion", "reserve", "réserve"],
    "log_hours": ["log", "hours", "time", "record", "enregistre", "heures", "heure"],
    "set_reminder": ["remind", "reminder", "rappelle", "rappel"],
    "query_calendar": ["calendar", "schedule", "meetings", "available", "slot",
                       "planning", "reunions", "réunions", "quelles"],
    "escalate": ["escalate", "human", "helpdesk"],
    "refuse": ["cannot", "won't", "unable", "refuse"],
    "greet": ["hello", "hi", "hey", "good morning", "bonjour"],
}


def load_16k_mono(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sr, SAMPLE_RATE)
    return mono.numpy()


def transcribe_hi(segment, turn_id, timeout=30):
    """Transcribe with language forced to Hindi."""
    buf = io.BytesIO()
    sf.write(buf, segment, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    audio = buf.getvalue()

    r = httpx.post(
        f"{STT.api_base}/audio/transcriptions",
        headers={"Authorization": f"Bearer {STT.key()}"},
        files={"file": ("turn.wav", audio, "audio/wav")},
        data={"model": STT.provider_model, "response_format": "json",
              "temperature": "0", "language": "fr"},
        timeout=timeout,
    )
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def infer_intent(transcript: str) -> str:
    """Heuristic intent inference from transcript (pre-VOX-019 structured extraction)."""
    lower = transcript.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    return "unknown"


def main():
    manifest = json.loads(MANIFEST.read_text())
    labels   = json.loads(LABELS.read_text())

    print("Multilingual STT + entity score — French (fr)")
    print(f"Model: {STT.repo_id} via {STT.provider}")
    print("=" * 65)

    extracted_records = []

    for utt in manifest:
        uid = utt["id"]
        wav_path = REPO_ROOT / utt["file"]
        turn_id = new_turn_id()

        clip = load_16k_mono(wav_path)
        transcript = transcribe_hi(clip, turn_id)
        intent = infer_intent(transcript)

        print(f"\n[{uid}]")
        print(f"  source    : {utt['text']}")
        print(f"  gloss     : {utt['english_gloss']}")
        print(f"  transcript: {transcript!r}")
        print(f"  intent    : {intent}")

        extracted_records.append({"id": uid, "extracted": {"intent": intent}})

    print("\n" + "=" * 65)

    # Score intent only — the only field we can extract pre-VOX-019
    gold_records = [
        {"id": l["id"], "category": "multilingual",
         "gold": {"intent": l["gold"]["intent"]}}
        for l in labels
    ]

    result = score_dataset(gold_records, extracted_records)

    print(f"\nmultilingual intent capture rate: {result['overall_capture_rate']:.4f}  "
          f"({result['total_correct']}/{result['total_gold']} slots)")
    print()
    for ur in result["utterance_results"]:
        outcome = ur["slot_results"].get("intent", "miss")
        mark = "PASS" if outcome == "correct" else "FAIL"
        print(f"  [{ur['id']}] intent={mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
