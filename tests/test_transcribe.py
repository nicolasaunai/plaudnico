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
    assert words_from_result(RESULT) == [
        Word("Bonjour", 0.0, 0.4),
        Word("à", 0.4, 0.5),
        Word("tous.", 0.6, 1.0),
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
