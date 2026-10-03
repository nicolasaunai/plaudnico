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
def test_transcribe_real_model(tmp_path):
    import subprocess

    from plaud.audio import to_wav
    from plaud.transcribe import transcribe

    aiff = tmp_path / "s.aiff"
    # Explicit English voice: the default voice follows the system language.
    try:
        subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), "The meeting starts now."],
                       check=True)
    except subprocess.CalledProcessError:
        pytest.skip("macOS voice Samantha not installed")
    words, lang = transcribe(to_wav(aiff, tmp_path / "s.wav"))
    assert lang == "en"
    assert "meeting" in " ".join(w.text.lower() for w in words)


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

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake))
    assert transcribe(tmp_path / "x.wav") == ([], "fr")
    assert seen["condition_on_previous_text"] is False
    assert seen["hallucination_silence_threshold"] > 0
