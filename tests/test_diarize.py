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


def test_pyannote_telemetry_forced_off(monkeypatch):
    import importlib

    import plaud.diarize

    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    importlib.reload(plaud.diarize)
    import os
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def _torch(cuda, mps):
    from types import SimpleNamespace as ns
    return ns(cuda=ns(is_available=lambda: cuda),
              backends=ns(mps=ns(is_available=lambda: mps)))


def test_resolve_device_auto():
    from plaud.diarize import resolve_device

    assert resolve_device("auto", _torch(cuda=True, mps=False)) == "cuda"
    assert resolve_device("auto", _torch(cuda=False, mps=True)) == "mps"
    assert resolve_device("auto", _torch(cuda=False, mps=False)) == "cpu"


def test_resolve_device_explicit():
    from plaud.diarize import resolve_device

    assert resolve_device("cpu", _torch(cuda=True, mps=True)) == "cpu"
