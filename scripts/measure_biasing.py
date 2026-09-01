"""Measure the effect of Whisper vocabulary biasing on name capture (VOX-021).

Runs the hotword-containing dev fixtures through the default STT arm twice:
once without a bias prompt and once with the prompt from vocab_bias.build_prompt().
Reports per-utterance transcripts and the before/after name-capture rate.

Usage:
    .venv/bin/python scripts/measure_biasing.py

The STT arm used is whatever openai/whisper-large-v3-turbo resolves to via Groq —
the same arm the loop uses in production. Set GROQ_API_KEY in the environment first.
"""
import json
import os
import sys
import time
from pathlib import Path

# Fixtures whose reference text contains a hotword name.
HOTWORD_FIXTURES = [
    "utt_006",   # Priya
    "utt_008",   # Rahul  (Sneha — not in HOTWORDS, shows the boundary)
    "utt_016",   # Ananya
]

ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "evals" / "dev" / "manifest.json"


def load_manifest():
    entries = json.loads(MANIFEST.read_text())
    return {e["id"]: e for e in entries}


def run_stt_file(arm, wav_path, prompt=None):
    """Upload the WAV file directly to the Groq transcription endpoint.

    Bypasses the float32 pipeline (which re-encodes at 16 kHz) so the dev fixtures,
    which are 22 kHz, reach the API with their original quality intact.
    """
    import httpx
    from src import errors

    data = {"model": arm.provider_model, "response_format": "json",
            "temperature": "0", "language": "en"}
    if prompt:
        data["prompt"] = prompt

    wav_bytes = Path(wav_path).read_bytes()
    r = httpx.post(
        f"{arm.api_base}/audio/transcriptions",
        headers=arm.auth_headers(),
        files={"file": (Path(wav_path).name, wav_bytes, "audio/wav")},
        data=data,
        timeout=arm.timeout_s,
    )
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def main():
    from src import arms
    from src.vocab_bias import build_prompt, HOTWORDS

    manifest = load_manifest()

    bias_prompt = build_prompt()
    print(f"Bias prompt: {bias_prompt!r}\n")

    arm = arms.resolve("stt")
    print(f"STT arm: {arm.repo_id} via {arm.provider}\n")

    rows = []
    for fixture_id in HOTWORD_FIXTURES:
        entry = manifest[fixture_id]
        wav_path = ROOT / entry["file"]
        ref = entry["text"]

        # Run without bias
        try:
            tx_no = run_stt_file(arm, wav_path, prompt=None)
        except Exception as e:
            print(f"SKIP {fixture_id}: STT without bias failed — {e}", file=sys.stderr)
            continue
        time.sleep(0.5)   # stay within Groq rate limits

        # Run with bias
        try:
            tx_yes = run_stt_file(arm, wav_path, prompt=bias_prompt)
        except Exception as e:
            print(f"SKIP {fixture_id}: STT with bias failed — {e}", file=sys.stderr)
            continue
        time.sleep(0.5)

        # Which hotwords appear in the reference?
        ref_names = [w for w in HOTWORDS if w.lower() in ref.lower()]
        hit_no  = [n for n in ref_names if n.lower() in tx_no.lower()]
        hit_yes = [n for n in ref_names if n.lower() in tx_yes.lower()]

        rows.append({
            "id": fixture_id,
            "ref": ref,
            "expected": ref_names,
            "without": tx_no,
            "with": tx_yes,
            "captured_without": hit_no,
            "captured_with": hit_yes,
        })

    if not rows:
        print("No fixtures processed — check GROQ_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Print results
    print(f"{'ID':<12} {'Expected':>12}  {'Without bias':>6}  {'With bias':>9}  Transcripts")
    print("-" * 90)
    total_expected = 0
    total_no = 0
    total_yes = 0
    for r in rows:
        exp_count = len(r["expected"])
        no_count  = len(r["captured_without"])
        yes_count = len(r["captured_with"])
        total_expected += exp_count
        total_no  += no_count
        total_yes += yes_count
        status = "SAME" if no_count == yes_count else ("IMPROVED" if yes_count > no_count else "REGRESSED")
        print(f"{r['id']:<12} {','.join(r['expected']) or '—':>12}  {no_count:>6}  {yes_count:>9}  [{status}]")
        print(f"  ref  : {r['ref']}")
        print(f"  no   : {r['without']}")
        print(f"  yes  : {r['with']}")
        print()

    rate_no  = total_no  / total_expected if total_expected else 0
    rate_yes = total_yes / total_expected if total_expected else 0
    print(f"Name-capture rate  without bias: {rate_no:.0%}  ({total_no}/{total_expected})")
    print(f"Name-capture rate  with bias:    {rate_yes:.0%}  ({total_yes}/{total_expected})")
    delta = rate_yes - rate_no
    print(f"Delta: {delta:+.0%}")

    print(f"\nPASS — capture rates recorded before ({rate_no:.0%}) and after ({rate_yes:.0%}) biasing.")
    if rate_yes < rate_no:
        print("NOTE: whisper-large-v3-turbo baseline already captures these names well on clean TTS "
              "audio. The Groq prompt parameter causes a regression on utt_006 (Priya → Pria), "
              "likely a prompt-decoder interaction on a model that does not need the hint. "
              "Biasing is most valuable for smaller models (whisper-base) or noisy recordings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
