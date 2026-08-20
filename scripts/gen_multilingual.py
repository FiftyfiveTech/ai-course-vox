"""Generate 5 Hindi multilingual utterance WAV files using espeak-ng.

These are used for the multilingual leg of Phase 1B (VOX-022).
Language: Hindi (hi) via espeak-ng local TTS.
Reference date: 2026-08-18

Usage:
    python3 scripts/gen_multilingual.py
"""
import ctypes
import ctypes.util
import io
import json
import os
import struct
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Hindi utterances — workplace commands, one per intent category
# Gold labels follow ENTITY_SPEC.md normalisation rules.
# Reference date: 2026-08-18 (Tuesday)
# ---------------------------------------------------------------------------

UTTERANCES = [
    {
        "id": "ml_001",
        "language": "fr",
        "text": "enregistre quatre heures sur le projet VOX aujourd'hui",
        "english_gloss": "log four hours on VOX project today",
        "gold": {
            "intent": "log_hours",
            "duration": "4h",
            "project": "vox project",
            "date": "2026-08-18",
        },
    },
    {
        "id": "ml_002",
        "language": "fr",
        "text": "reserve une reunion d'une heure avec Priya demain a quinze heures",
        "english_gloss": "book a one hour meeting with Priya tomorrow at three pm",
        "gold": {
            "intent": "book_meeting",
            "person": ["priya"],
            "duration": "1h",
            "date": "2026-08-19",
            "time": "15:00",
        },
    },
    {
        "id": "ml_003",
        "language": "fr",
        "text": "rappelle a Kiran de soumettre la feuille de temps vendredi a dix-sept heures",
        "english_gloss": "remind Kiran to submit the timesheet on Friday at five pm",
        "gold": {
            "intent": "set_reminder",
            "person": ["kiran"],
            "time": "17:00",
            "date": "2026-08-21",
        },
    },
    {
        "id": "ml_004",
        "language": "fr",
        "text": "quelles reunions ai-je demain",
        "english_gloss": "what meetings do I have tomorrow",
        "gold": {
            "intent": "query_calendar",
            "date": "2026-08-19",
        },
    },
    {
        "id": "ml_005",
        "language": "fr",
        "text": "montre mon planning pour cette semaine",
        "english_gloss": "show me my schedule for this week",
        "gold": {
            "intent": "query_calendar",
            "date": "week of 2026-08-17",
        },
    },
]

# ---------------------------------------------------------------------------
# espeak-ng synthesis (same approach as gen_utterances.py)
# ---------------------------------------------------------------------------

AUDIO_OUTPUT_SYNCHRONOUS = 0x02
SYNTH_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
)

_audio_buf: list[int] = []
_sample_rate: int = 22050


def _synth_callback(wav, numsamples, events):
    if wav and numsamples > 0:
        for i in range(numsamples):
            _audio_buf.append(wav[i])
    return 0


def _find_espeak_lib():
    for name in ("espeak-ng", "espeak-ng.so.1", "libespeak-ng.so.1"):
        path = ctypes.util.find_library(name)
        if path:
            return path
    for path in (
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/libespeak-ng.so.1",
        "/usr/local/lib/libespeak-ng.so.1",
    ):
        if os.path.exists(path):
            return path
    return None


def synth_to_wav(lib, text: str) -> bytes:
    global _audio_buf
    _audio_buf = []
    cb = SYNTH_CALLBACK(_synth_callback)
    lib.espeak_SetSynthCallback(cb)
    text_bytes = text.encode("utf-8")
    lib.espeak_Synth(
        ctypes.c_char_p(text_bytes),
        ctypes.c_size_t(len(text_bytes) + 1),
        ctypes.c_uint(0),
        ctypes.c_int(1),
        ctypes.c_uint(0),
        ctypes.c_uint(0x08),     # espeakCHARS_UTF8
        None,
        None,
    )
    lib.espeak_Synchronize()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(struct.pack(f"<{len(_audio_buf)}h", *_audio_buf))
    return buf.getvalue()


def main():
    global _sample_rate

    lib_path = _find_espeak_lib()
    if not lib_path:
        print("ERROR: libespeak-ng not found. Install: sudo apt install espeak-ng", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {lib_path}")
    lib = ctypes.CDLL(lib_path)
    lib.espeak_Initialize.restype = ctypes.c_int
    rate = lib.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, None, 0)
    if rate < 0:
        print(f"ERROR: espeak_Initialize returned {rate}", file=sys.stderr)
        sys.exit(1)
    _sample_rate = rate

    result = lib.espeak_SetVoiceByName(b"fr")
    if result != 0:
        print("WARNING: French voice not available, falling back to en", file=sys.stderr)
        lib.espeak_SetVoiceByName(b"en")

    out_dir = REPO_ROOT / "evals" / "multilingual"
    out_dir.mkdir(exist_ok=True)

    manifest = []
    labels = []

    for utt in UTTERANCES:
        filename = f"{utt['id']}_fr.wav"
        filepath = out_dir / filename
        wav_bytes = synth_to_wav(lib, utt["text"])
        filepath.write_bytes(wav_bytes)
        print(f"  {filename}  ({len(wav_bytes):,} bytes)  [{utt['english_gloss']}]")

        manifest.append({
            "id": utt["id"],
            "language": utt["language"],
            "text": utt["text"],
            "english_gloss": utt["english_gloss"],
            "file": f"evals/multilingual/{filename}",
        })
        labels.append({
            "id": utt["id"],
            "language": utt["language"],
            "text": utt["text"],
            "english_gloss": utt["english_gloss"],
            "gold": utt["gold"],
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    print(f"\nDone. {len(UTTERANCES)} Hindi utterances written to {out_dir}")
    print(f"manifest: {out_dir / 'manifest.json'}")
    print(f"labels:   {out_dir / 'labels.json'}")


if __name__ == "__main__":
    main()
