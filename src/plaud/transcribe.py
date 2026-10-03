import ctypes
import importlib.util
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


class TranscribeError(RuntimeError):
    pass


def backend(system: str | None = None, machine: str | None = None) -> str:
    """mlx-whisper on Apple Silicon, faster-whisper on Linux; PLAUD_ASR_BACKEND overrides."""
    forced = os.environ.get("PLAUD_ASR_BACKEND")
    if forced:
        if forced not in DEFAULT_MODELS:
            raise TranscribeError(f"PLAUD_ASR_BACKEND must be one of: {', '.join(DEFAULT_MODELS)}"
                                  f" (got '{forced}')")
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


# CTranslate2 opens these by name; dependencies first. pip's nvidia wheels put them in
# site-packages/nvidia/*/lib, which is not on the loader path.
_CUDA_LIBS = ["libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9", "libcudnn_*.so.9"]


def preload_cuda_libs(lib_dirs: list[Path], loader=None) -> None:
    loader = loader or (lambda path: ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))
    done = set()
    for pattern in _CUDA_LIBS:
        for d in lib_dirs:
            for lib in sorted(d.glob(pattern)):
                if lib.name in done:
                    continue
                done.add(lib.name)
                try:
                    loader(str(lib))
                except OSError:
                    pass  # CTranslate2 reports what is really missing; we then fall back to CPU.


def _preload_cuda_libs() -> None:
    dirs = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            spec = None
        if spec and spec.submodule_search_locations:
            dirs += [Path(p) / "lib" for p in spec.submodule_search_locations]
    preload_cuda_libs(dirs)


def _run_faster_whisper(audio, model: str, language: str | None, device: str) -> dict:
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model, device=device,
                           compute_type="auto" if device == "cuda" else "int8")
    segments, info = whisper.transcribe(
        audio, language=language, word_timestamps=True, vad_filter=True, **_DECODE)
    # Same shape as mlx-whisper's result, so both backends share words_from_result.
    # Iterating here runs the decoding (segments is lazy), so GPU errors surface inside.
    return {
        "language": info.language,
        "segments": [
            {"no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
             "words": [{"word": w.word, "start": w.start, "end": w.end} for w in s.words or []]}
            for s in segments
        ],
    }


def _transcribe_faster_whisper(wav: Path, model: str, language: str | None) -> dict:
    import ctranslate2
    import soundfile as sf

    # Pass samples, not the path: faster-whisper's own decoder (PyAV) breaks with PyAV 19.
    # The wav is already 16 kHz mono, which is what Whisper expects.
    audio, _ = sf.read(str(wav), dtype="float32")
    if ctranslate2.get_cuda_device_count() > 0:
        _preload_cuda_libs()
        try:
            return _run_faster_whisper(audio, model, language, "cuda")
        except (RuntimeError, ValueError) as e:
            print(f"plaud: GPU transcription failed ({e}); falling back to CPU.",
                  file=sys.stderr)
    return _run_faster_whisper(audio, model, language, "cpu")


def transcribe(wav: Path, model: str | None = None,
               language: str | None = None) -> tuple[list[Word], str]:
    name = backend()
    run = _transcribe_mlx if name == "mlx" else _transcribe_faster_whisper
    result = run(wav, model or default_model(name), language)
    return words_from_result(result), result.get("language") or language or "unknown"
