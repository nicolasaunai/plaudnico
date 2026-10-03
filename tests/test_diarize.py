import pytest

from plaud.diarize import DiarizeError, hf_token, label_speakers
from plaud.models import Turn


def test_label_speakers_in_order_of_first_appearance():
    raw = [(5.0, 6.0, "SPEAKER_00"), (0.0, 2.0, "SPEAKER_03"), (2.5, 4.0, "SPEAKER_00")]
    assert label_speakers(raw) == [
        Turn(0.0, 2.0, "Speaker 1"),
        Turn(2.5, 4.0, "Speaker 2"),
        Turn(5.0, 6.0, "Speaker 2"),
    ]


def test_label_speakers_empty():
    assert label_speakers([]) == []


def test_hf_token_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(DiarizeError, match="HF_TOKEN"):
        hf_token()


def test_hf_token_present(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    assert hf_token() == "hf_abc"
