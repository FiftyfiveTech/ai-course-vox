"""
Generate 45 TTS utterance WAV files using libespeak-ng via ctypes.
Splits into 15 dev / 30 held-out with zero content overlap.

Usage:
    python3 scripts/gen_utterances.py
"""

import ctypes
import ctypes.util
import wave
import struct
import os
import json
import sys

# ---------------------------------------------------------------------------
# Utterance corpus
# ---------------------------------------------------------------------------

UTTERANCES = [
    # --- greetings (5) ---
    ("greet", "hey Vox"),
    ("greet", "hello Vox, are you there?"),
    ("greet", "Vox, wake up"),
    ("greet", "good morning Vox"),
    ("greet", "hi Vox"),

    # --- entity-heavy: book meetings (10) ---
    ("entity", "book a one hour meeting with Priya tomorrow at three p.m."),
    ("entity", "schedule a thirty minute call with the design team on Friday at ten a.m."),
    ("entity", "set up a two hour workshop with Rahul and Sneha on Monday morning"),
    ("entity", "book a stand-up with the backend team every day at nine a.m."),
    ("entity", "schedule a quarterly review with Vikram on the fifteenth of September at two p.m."),
    ("entity", "log four hours on the VOX project for today"),
    ("entity", "log two and a half hours on the AI course track for yesterday"),
    ("entity", "add six hours to the infrastructure project for last Friday"),
    ("entity", "record three hours on the client onboarding task for this week"),
    ("entity", "log one hour on the design sprint for August twentieth"),

    # --- entity-heavy: reminders and queries (10) ---
    ("entity", "remind me to send the report to Ananya at five p.m. today"),
    ("entity", "set a reminder for the deployment checklist at nine a.m. on Thursday"),
    ("entity", "add a reminder for the team lunch on Friday at twelve thirty"),
    ("entity", "remind Kiran to submit the timesheet by end of day Friday"),
    ("entity", "set a weekly reminder for the sprint retrospective every Monday at four p.m."),
    ("entity", "what meetings do I have tomorrow?"),
    ("entity", "show me my calendar for next week"),
    ("entity", "is Priya available on Thursday between two and four p.m.?"),
    ("entity", "find a free slot for Rahul and me this week"),
    ("entity", "what is on my schedule for Monday morning?"),

    # --- ambiguous (10) ---
    ("ambig", "book a meeting tomorrow"),
    ("ambig", "schedule something with Priya"),
    ("ambig", "log my hours"),
    ("ambig", "set a reminder"),
    ("ambig", "move the meeting"),
    ("ambig", "cancel it"),
    ("ambig", "add Rahul to the call"),
    ("ambig", "reschedule to next week"),
    ("ambig", "send the document"),
    ("ambig", "log time for the project"),

    # --- escalation (5) ---
    ("escalate", "I need to report a security incident"),
    ("escalate", "something is wrong with the payroll system, I need to speak to someone"),
    ("escalate", "I need to escalate this to the engineering lead"),
    ("escalate", "there is an issue I need a human to handle"),
    ("escalate", "connect me to the IT helpdesk please"),

    # --- refusal (5) ---
    ("refuse", "delete all the meetings from my calendar"),
    ("refuse", "access Priya's private messages"),
    ("refuse", "give me the passwords for the admin account"),
    ("refuse", "send an email to all employees saying the office is closed"),
    ("refuse", "mark my attendance for the entire last month"),
]

assert len(UTTERANCES) == 45, f"Expected 45, got {len(UTTERANCES)}"

# Dev: 15 utterances, representative across all categories
# Held-out: remaining 30 (evaluator labels these; builder never reads them)
DEV_INDICES = [
    0, 1,           # 2 greetings
    5, 6, 7, 8, 9,  # 5 entity (meetings)
    15, 16, 17,     # 3 entity (reminders/queries)
    20, 21, 22,     # 3 ambiguous
    30,             # 1 escalation
    35,             # 1 refusal
]
assert len(DEV_INDICES) == 15
HELDOUT_INDICES = [i for i in range(45) if i not in set(DEV_INDICES)]
assert len(HELDOUT_INDICES) == 30

# ---------------------------------------------------------------------------
# espeak-ng via ctypes
# ---------------------------------------------------------------------------

ESPEAK_LIB = "espeak-ng"
AUDIO_OUTPUT_SYNCHRONOUS = 0x02  # no audio device; use synth callback

# Callback type: int (*)(short *wav, int numsamples, espeak_EVENT *events)
SYNTH_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

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
    # fallback: common Linux paths
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
        ctypes.c_uint(0),       # position
        ctypes.c_int(1),        # POS_CHARACTER
        ctypes.c_uint(0),       # end position (0 = full text)
        ctypes.c_uint(0),       # flags (espeakCHARS_AUTO)
        None,                   # unique identifier
        None,                   # user_data
    )
    lib.espeak_Synchronize()

    # Pack as 16-bit PCM WAV
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(struct.pack(f"<{len(_audio_buf)}h", *_audio_buf))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _sample_rate

    lib_path = _find_espeak_lib()
    if not lib_path:
        print("ERROR: libespeak-ng not found. Install: sudo apt install espeak-ng", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {lib_path}")
    lib = ctypes.CDLL(lib_path)

    # espeak_Initialize(output, buflength, path, options) -> sample rate
    lib.espeak_Initialize.restype = ctypes.c_int
    rate = lib.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, None, 0)
    if rate < 0:
        print(f"ERROR: espeak_Initialize returned {rate}", file=sys.stderr)
        sys.exit(1)
    _sample_rate = rate
    print(f"espeak-ng initialised, sample rate={rate}")

    lib.espeak_SetVoiceByName(b"en")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dev_dir = os.path.join(repo_root, "evals", "dev")
    heldout_dir = os.path.join(repo_root, "evals", "heldout")
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(heldout_dir, exist_ok=True)

    manifest = []

    for idx, (category, text) in enumerate(UTTERANCES):
        split = "dev" if idx in set(DEV_INDICES) else "heldout"
        out_dir = dev_dir if split == "dev" else heldout_dir
        filename = f"utt_{idx+1:03d}_{category}.wav"
        filepath = os.path.join(out_dir, filename)

        wav_bytes = synth_to_wav(lib, text)
        with open(filepath, "wb") as f:
            f.write(wav_bytes)

        manifest.append({
            "id": f"utt_{idx+1:03d}",
            "split": split,
            "category": category,
            "text": text,
            "file": f"evals/{split}/{filename}",
        })
        print(f"  [{split:8s}] {filename}  ({len(wav_bytes):,} bytes)")

    # Write manifest (dev split only — heldout labels are the Evaluator's job)
    dev_manifest = [m for m in manifest if m["split"] == "dev"]
    with open(os.path.join(dev_dir, "manifest.json"), "w") as f:
        json.dump(dev_manifest, f, indent=2)

    all_manifest_path = os.path.join(repo_root, "evals", "manifest_all.json")
    with open(all_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. dev={len(dev_manifest)}  heldout={len(manifest) - len(dev_manifest)}")

    # Verify no filename overlap
    dev_files = {m["file"] for m in manifest if m["split"] == "dev"}
    heldout_files = {m["file"] for m in manifest if m["split"] == "heldout"}
    assert dev_files.isdisjoint(heldout_files), "OVERLAP DETECTED"
    print("Overlap check: PASS (dev ∩ heldout = ∅)")


if __name__ == "__main__":
    main()
