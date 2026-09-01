"""Text-to-speech backends: Kokoro-82M, speecht5_tts and piper-voices, all local. No spend.

Backends return samples rather than playing them, because VOX-011 (barge-in) has to be able to
interrupt playback without touching synthesis. They return them at their **own** sample rate —
Kokoro 24 kHz, SpeechT5 16 kHz, piper 22.05 kHz — and `arms.tts()` hands the rate on with the audio.
Resampling one to match another would put a lie in the middle of a comparison of the two.
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


def load_piper(arm):
    """The pinned voice's .onnx and its .onnx.json, straight out of the HF repo. ~65 MB once.

    Both files are named in config.py rather than derived from a voice string here, because
    `rhasspy/piper-voices` holds ~120 voices and the repo id alone therefore does not say what will
    speak. Same reason the SpeechT5 speaker embedding is pinned by filename.

    The sample rate is read back off the .onnx.json and checked against the arm's declared one. The
    arm's number is what `arms.tts()` hands the speaker and what a comparison against Kokoro is
    normalised on, so a silent disagreement would not be a wrong log line — it would play every
    piper reply at the wrong pitch.
    """
    if arm.id not in _loaded:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        onnx = hf_hub_download(arm.repo_id, arm.extra["onnx"])
        config = hf_hub_download(arm.repo_id, arm.extra["onnx_config"])
        voice = PiperVoice.load(onnx, config_path=config)
        declared, actual = arm.extra["sample_rate"], voice.config.sample_rate
        if actual != declared:
            raise RuntimeError(
                f"{arm.id} declares sample_rate={declared} but {arm.extra['onnx_config']} says "
                f"{actual}. Fix the config.py row — playback and every arm comparison read the "
                f"declared number."
            )
        _loaded[arm.id] = voice
    return _loaded[arm.id]


def piper(arm, text, rec):
    """-> float32 mono at 22.05 kHz. One chunk per sentence, concatenated."""
    voice = load_piper(arm)
    # The same field Kokoro fills with af_heart and SpeechT5 with its xvector name: who spoke.
    rec["voice"] = Path(arm.extra["onnx"]).stem
    parts = [chunk.audio_float_array for chunk in voice.synthesize(text)]
    if not parts:
        raise RuntimeError(f"piper produced no audio for {text!r}")
    return _measured(np.concatenate(parts).astype(np.float32), arm, rec)


BACKENDS = {"kokoro": kokoro, "speecht5": speecht5, "piper": piper}
LOADERS = {"kokoro": load_kokoro, "speecht5": load_speecht5, "piper": load_piper}
