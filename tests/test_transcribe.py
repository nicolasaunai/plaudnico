import pytest

from plaud.models import Word
from plaud.transcribe import words_from_result

RESULT = {
    "language": "fr",
    "segments": [
        {"words": [
            {"word": " Bonjour", "start": 0.0, "end": 0.4, "probability": 0.9},
            {"word": " à", "start": 0.4, "end": 0.5, "probability": 0.9},
            {"word": " ", "start": 0.5, "end": 0.5, "probability": 0.1},
        ]},
        {"words": [{"word": " tous.", "start": 0.6, "end": 1.0, "probability": 0.9}]},
        {"text": "segment without words"},
    ],
}


def test_words_from_result():
    # Whisper's spacing is kept: it is what tells "l'arrêté" from "l arrêté".
    assert words_from_result(RESULT) == [
        Word(" Bonjour", 0.0, 0.4),
        Word(" à", 0.4, 0.5),
        Word(" tous.", 0.6, 1.0),
    ]


def test_words_from_empty_result():
    assert words_from_result({"segments": []}) == []
    assert words_from_result({}) == []


@pytest.mark.slow
def test_transcribe_real_model(tmp_path, tts):
    from plaud.audio import to_wav
    from plaud.transcribe import transcribe

    audio = tts("The meeting starts now.", 1, tmp_path / "s")
    words, lang = transcribe(to_wav(audio, tmp_path / "s16k.wav"))
    assert lang == "en"
    assert "meeting" in "".join(w.text.lower() for w in words)


@pytest.mark.slow
def test_faster_whisper_real_model(tmp_path, tts, monkeypatch):
    from plaud.audio import to_wav
    from plaud.transcribe import transcribe

    monkeypatch.setenv("PLAUD_ASR_BACKEND", "faster-whisper")
    audio = tts("The meeting starts now.", 1, tmp_path / "s")
    words, lang = transcribe(to_wav(audio, tmp_path / "s16k.wav"), model="tiny")
    assert lang == "en"
    assert "meeting" in "".join(w.text.lower() for w in words)


def test_words_from_result_drops_likely_hallucinated_segments():
    result = {"segments": [
        {"no_speech_prob": 0.9, "avg_logprob": -1.5,
         "words": [{"word": " Sous-titres", "start": 0.0, "end": 1.0}]},
        {"no_speech_prob": 0.9, "avg_logprob": -0.2,
         "words": [{"word": " Oui.", "start": 1.0, "end": 1.2}]},
    ]}
    assert words_from_result(result) == [Word(" Oui.", 1.0, 1.2)]


def test_transcribe_passes_anti_hallucination_options(monkeypatch, tmp_path):
    import sys
    import types

    from plaud.transcribe import transcribe

    seen = {}

    def fake(audio, **kw):
        seen.update(kw)
        return {"language": "fr", "segments": []}

    monkeypatch.setenv("PLAUD_ASR_BACKEND", "mlx")
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake))
    assert transcribe(tmp_path / "x.wav") == ([], "fr")
    assert seen["condition_on_previous_text"] is False
    assert seen["hallucination_silence_threshold"] > 0


def test_backend_selection(monkeypatch):
    from plaud.transcribe import backend

    monkeypatch.delenv("PLAUD_ASR_BACKEND", raising=False)
    assert backend("darwin", "arm64") == "mlx"
    assert backend("linux", "x86_64") == "faster-whisper"
    assert backend("linux", "aarch64") == "faster-whisper"
    assert backend("darwin", "x86_64") == "faster-whisper"  # Intel Mac
    monkeypatch.setenv("PLAUD_ASR_BACKEND", "faster-whisper")
    assert backend("darwin", "arm64") == "faster-whisper"


def test_default_model_per_backend():
    from plaud.transcribe import default_model

    assert default_model("mlx") == "mlx-community/whisper-large-v3-mlx"
    assert default_model("faster-whisper") == "large-v3"


def _fake_faster_whisper(monkeypatch, cuda_devices):
    import sys
    import types

    seen = {}

    class WhisperModel:
        def __init__(self, model, **kw):
            seen["model"], seen["init"] = model, kw

        def transcribe(self, audio, **kw):
            seen["audio"], seen["transcribe"] = audio, kw
            ns = types.SimpleNamespace
            segments = [
                ns(no_speech_prob=0.1, avg_logprob=-0.2,
                   words=[ns(word=" Bonjour", start=0.0, end=0.4),
                          ns(word=" l", start=0.5, end=0.6),
                          ns(word="'équipe.", start=0.6, end=1.0)]),
                ns(no_speech_prob=0.9, avg_logprob=-1.5,
                   words=[ns(word=" Sous-titres", start=2.0, end=3.0)]),
                ns(no_speech_prob=0.0, avg_logprob=-0.1, words=None),
            ]
            return iter(segments), ns(language="fr")

    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=WhisperModel))
    monkeypatch.setitem(sys.modules, "ctranslate2",
                        types.SimpleNamespace(get_cuda_device_count=lambda: cuda_devices))
    monkeypatch.setenv("PLAUD_ASR_BACKEND", "faster-whisper")
    return seen


def _silent_wav(path):
    import numpy as np
    import soundfile as sf

    sf.write(str(path), np.zeros(1600, dtype="float32"), 16000)
    return path


def test_faster_whisper_backend_cpu(monkeypatch, tmp_path):
    from plaud.transcribe import transcribe

    seen = _fake_faster_whisper(monkeypatch, cuda_devices=0)
    words, lang = transcribe(_silent_wav(tmp_path / "x.wav"), model="large-v3")
    assert lang == "fr"
    assert words == [Word(" Bonjour", 0.0, 0.4), Word(" l", 0.5, 0.6),
                     Word("'équipe.", 0.6, 1.0)]
    assert seen["model"] == "large-v3"
    assert seen["init"]["device"] == "cpu"
    assert seen["init"]["compute_type"] == "int8"
    kw = seen["transcribe"]
    assert kw["word_timestamps"] is True
    assert kw["condition_on_previous_text"] is False
    assert kw["vad_filter"] is True
    assert kw["hallucination_silence_threshold"] > 0
    assert kw["language"] is None
    assert seen["audio"].dtype.name == "float32"  # samples, not a path (PyAV bypass)


def test_faster_whisper_backend_uses_gpu_when_present(monkeypatch, tmp_path):
    from plaud.transcribe import transcribe

    seen = _fake_faster_whisper(monkeypatch, cuda_devices=1)
    transcribe(_silent_wav(tmp_path / "x.wav"), model="large-v3", language="fr")
    assert seen["init"]["device"] == "cuda"
    assert seen["init"]["compute_type"] == "float16"
    assert seen["transcribe"]["language"] == "fr"


def test_transcribe_default_model_follows_backend(monkeypatch, tmp_path):
    from plaud.transcribe import transcribe

    seen = _fake_faster_whisper(monkeypatch, cuda_devices=0)
    transcribe(_silent_wav(tmp_path / "x.wav"))
    assert seen["model"] == "large-v3"
