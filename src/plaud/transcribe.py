import os
import platform
import sys
from pathlib import Path

from plaud.models import Word

# large-v3 beat turbo clearly on a real French meeting (acronyms, overlapping speech).
# Same weights on both backends: mlx-whisper (Apple GPU) and faster-whisper (CUDA/CPU).
DEFAULT_MODELS = {
    "mlx": "mlx-community/whisper-large-v3-mlx",
    "faster-whisper": "large-v3",
}

# Shared decoding options: limit repetition loops and invented text on silences.
_DECODE = {"condition_on_previous_text": False, "hallucination_silence_threshold": 2.0}


def backend(system: str | None = None, machine: str | None = None) -> str:
    """mlx-whisper on Apple Silicon, faster-whisper elsewhere; PLAUD_ASR_BACKEND overrides."""
    forced = os.environ.get("PLAUD_ASR_BACKEND")
    if forced:
        return forced
    system = system or sys.platform
    machine = machine or platform.machine()
    return "mlx" if system == "darwin" and machine == "arm64" else "faster-whisper"


def default_model(name: str | None = None) -> str:
    return DEFAULT_MODELS[name or backend()]


def _likely_hallucination(seg: dict) -> bool:
    # Whisper's own no-speech rule: probably silence and decoded with low confidence.
    return seg.get("no_speech_prob", 0.0) > 0.6 and seg.get("avg_logprob", 0.0) < -1.0


def words_from_result(result: dict) -> list[Word]:
    words = []
    for seg in result.get("segments", []):
        if _likely_hallucination(seg):
            continue
        for w in seg.get("words", []):
            # Keep Whisper's leading space: it marks word boundaries ("l" + "'arrêté").
            if w["word"].strip():
                words.append(Word(w["word"], float(w["start"]), float(w["end"])))
    return words


def _transcribe_mlx(wav: Path, model: str, language: str | None) -> dict:
    import mlx_whisper

    return mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=model, word_timestamps=True, language=language, **_DECODE)


def _transcribe_faster_whisper(wav: Path, model: str, language: str | None) -> dict:
    import ctranslate2
    import soundfile as sf
    from faster_whisper import WhisperModel

    gpu = ctranslate2.get_cuda_device_count() > 0
    whisper = WhisperModel(model, device="cuda" if gpu else "cpu",
                           compute_type="float16" if gpu else "int8")
    # Pass samples, not the path: faster-whisper's own decoder (PyAV) breaks with PyAV 19.
    # The wav is already 16 kHz mono, which is what Whisper expects.
    audio, _ = sf.read(str(wav), dtype="float32")
    segments, info = whisper.transcribe(
        audio, language=language, word_timestamps=True, vad_filter=True, **_DECODE)
    # Same shape as mlx-whisper's result, so both backends share words_from_result.
    return {
        "language": info.language,
        "segments": [
            {"no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
             "words": [{"word": w.word, "start": w.start, "end": w.end} for w in s.words or []]}
            for s in segments
        ],
    }


def transcribe(wav: Path, model: str | None = None,
               language: str | None = None) -> tuple[list[Word], str]:
    name = backend()
    run = _transcribe_mlx if name == "mlx" else _transcribe_faster_whisper
    result = run(wav, model or default_model(name), language)
    return words_from_result(result), result.get("language") or language or "unknown"
