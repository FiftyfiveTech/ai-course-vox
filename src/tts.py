"""Text-to-speech backends: hexgrad/Kokoro-82M and microsoft/speecht5_tts, both local. No spend.

Backends return samples rather than playing them, because VOX-011 (barge-in) has to be able to
interrupt playback without touching synthesis. They return them at their **own** sample rate —
Kokoro is 24 kHz and SpeechT5 16 kHz — and `arms.tts()` hands the rate on with the audio. Resampling
one to match the other would put a lie in the middle of a comparison of the two.
"""
import io
from pathlib import Path

import numpy as np

from src.config import TTS_VOICE

_loaded = {}   # arm.id -> the loaded pipeline/model bundle, so weights load once per process


def load_kokoro(arm):
    """Load the weights outside a timed turn. Called by arms.warm(); idempotent."""
    if arm.id not in _loaded:
        from kokoro import KPipeline
        # lang_code 'a' = American English. The weights come from the HF repo id in config.
        _loaded[arm.id] = KPipeline(lang_code="a", repo_id=arm.repo_id)
    return _loaded[arm.id]


def _measured(audio, arm, rec):
    """Log how much speech came out. chars-per-second of audio is how the arms get compared."""
    rec["audio_s"] = round(len(audio) / arm.extra["sample_rate"], 3)
    return audio


def _samples(chunk):
    """Kokoro's yielded chunk shape has moved between releases — take audio from either form."""
    audio = chunk[2] if isinstance(chunk, tuple) else getattr(chunk, "audio", None)
    if audio is None:
        raise RuntimeError(f"cannot find audio in Kokoro chunk of type {type(chunk).__name__}")
    return audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)


def kokoro(arm, text, rec):
    """-> float32 mono at 24 kHz."""
    pipeline = load_kokoro(arm)
    rec["voice"] = TTS_VOICE
    parts = [_samples(c) for c in pipeline(text, voice=TTS_VOICE)]
    if not parts:
        raise RuntimeError(f"Kokoro produced no audio for {text!r}")
    return _measured(np.concatenate(parts).astype(np.float32), arm, rec)


def _xvector(arm):
    """The one pinned 512-dim speaker embedding, read straight out of the dataset repo's zip.

    Deliberately not via `datasets`: Matthijs/cmu-arctic-xvectors is a script-based dataset, and
    datasets>=4 refuses to run dataset scripts ("Dataset scripts are no longer supported"). The
    payload is a zip of .npy files, so hf_hub_download plus zipfile gets the same array with no
    heavy dependency and no loader-version risk.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(arm.extra["xvector_repo"], arm.extra["xvector_zip"],
                           repo_type="dataset")
    with zipfile.ZipFile(path) as z, z.open(arm.extra["xvector_file"]) as f:
        return np.load(io.BytesIO(f.read()))


def load_speecht5(arm):
    """The acoustic model, its vocoder, and the pinned speaker embedding. ~650 MB on first call."""
    if arm.id not in _loaded:
        import torch
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

        processor = SpeechT5Processor.from_pretrained(arm.provider_model)
        model = SpeechT5ForTextToSpeech.from_pretrained(arm.provider_model)
        vocoder = SpeechT5HifiGan.from_pretrained(arm.extra["vocoder"])
        speaker = torch.from_numpy(_xvector(arm)).unsqueeze(0)
        _loaded[arm.id] = (processor, model, vocoder, speaker)
    return _loaded[arm.id]


def speecht5(arm, text, rec):
    """-> float32 mono at 16 kHz."""
    processor, model, vocoder, speaker = load_speecht5(arm)
    # The same field Kokoro fills with af_heart: which voice actually spoke this line.
    rec["voice"] = Path(arm.extra["xvector_file"]).stem
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker, vocoder=vocoder)
    audio = speech.detach().cpu().numpy().astype(np.float32)
    if audio.size == 0:
        raise RuntimeError(f"SpeechT5 produced no audio for {text!r}")
    return _measured(audio, arm, rec)


BACKENDS = {"kokoro": kokoro, "speecht5": speecht5}
LOADERS = {"kokoro": load_kokoro, "speecht5": load_speecht5}
