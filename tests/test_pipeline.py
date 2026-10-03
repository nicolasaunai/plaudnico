import json

import pytest

from plaud import pipeline
from plaud.models import Turn, Word
from plaud.report import ReportError


@pytest.fixture
def meeting(tmp_path, monkeypatch):
    d = tmp_path / "2026" / "2026-10-03_1015_test"
    d.mkdir(parents=True)
    (d / "audio.m4a").write_bytes(b"x")
    (d / "meeting.json").write_text(json.dumps(
        {"id": "h", "source": "s", "start": "2026-10-03T10:15", "title": "Test"}))
    calls = {"transcribe": 0, "diarize": 0, "claude": 0}

    def fake_to_wav(src, dst):
        dst.write_bytes(b"wav")
        return dst

    def fake_transcribe(wav, model, language):
        calls["transcribe"] += 1
        return [Word("Bonjour.", 0.0, 0.5), Word("Salut.", 1.0, 1.5)], "fr"

    def fake_diarize(wav, num_speakers, device):
        calls["diarize"] += 1
        return [Turn(0.0, 0.8, "Speaker 1"), Turn(0.9, 2.0, "Speaker 2")]

    def fake_claude(prompt):
        calls["claude"] += 1
        return "# Compte-rendu\n"

    monkeypatch.setattr(pipeline, "to_wav", fake_to_wav)
    monkeypatch.setattr(pipeline, "duration", lambda wav: 2.0)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "diarize", fake_diarize)
    monkeypatch.setattr(pipeline, "run_claude", fake_claude)
    return d, calls


def run(d, **kw):
    args = dict(templates=["minutes"], model="m", language=None,
                num_speakers=None, device="cpu", log=lambda *a: None)
    args.update(kw)
    return pipeline.process(d, **args)


def test_full_run(meeting):
    d, calls = meeting
    reports = run(d)
    assert reports == [d / "report-minutes.md"]
    assert (d / "report-minutes.md").read_text() == "# Compte-rendu\n"
    t = (d / "transcript.md").read_text()
    assert "**[00:00:00] Speaker 1:** Bonjour." in t
    assert "**[00:00:01] Speaker 2:** Salut." in t
    assert calls == {"transcribe": 1, "diarize": 1, "claude": 1}


def test_rerun_uses_cache(meeting):
    d, calls = meeting
    run(d)
    run(d)
    assert calls["transcribe"] == 1 and calls["diarize"] == 1
    assert calls["claude"] == 2


def test_force_recomputes(meeting):
    d, calls = meeting
    run(d)
    run(d, force=True)
    assert calls["transcribe"] == 2 and calls["diarize"] == 2


def test_no_speech_skips_report(meeting, monkeypatch):
    d, calls = meeting
    monkeypatch.setattr(pipeline, "transcribe", lambda wav, model, language: ([], "unknown"))
    assert run(d) == []
    assert "No speech detected" in (d / "transcript.md").read_text()
    assert calls["claude"] == 0


def test_claude_failure_keeps_transcript(meeting, monkeypatch):
    d, _ = meeting

    def boom(prompt):
        raise ReportError("claude -p failed: auth error")

    monkeypatch.setattr(pipeline, "run_claude", boom)
    with pytest.raises(ReportError):
        run(d)
    assert (d / "transcript.md").exists()


def test_no_templates(meeting):
    d, calls = meeting
    assert run(d, templates=[]) == []
    assert calls["claude"] == 0


def test_changed_language_recomputes_transcription(meeting):
    d, calls = meeting
    run(d)
    run(d, language="fr")
    assert calls["transcribe"] == 2
    assert calls["diarize"] == 1


def test_changed_model_recomputes_transcription(meeting):
    d, calls = meeting
    run(d)
    run(d, model="other")
    assert calls["transcribe"] == 2


def test_changed_speaker_count_recomputes_diarization(meeting):
    d, calls = meeting
    run(d)
    run(d, num_speakers=3)
    assert calls["diarize"] == 2
    assert calls["transcribe"] == 1


def test_diarize_runs_before_transcribe(meeting, monkeypatch):
    d, _ = meeting
    order = []
    monkeypatch.setattr(pipeline, "diarize",
                        lambda wav, num_speakers, device: order.append("d") or [])
    monkeypatch.setattr(pipeline, "transcribe",
                        lambda wav, model, language: order.append("t") or ([], "fr"))
    run(d)
    assert order == ["d", "t"]


def test_stage_failure_leaves_no_cache_file(meeting, monkeypatch):
    d, _ = meeting

    def boom(wav, model, language):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    with pytest.raises(RuntimeError):
        run(d)
    assert not (d / "asr.json").exists()
    assert not list(d.glob("*.tmp"))


def test_changed_backend_recomputes_transcription(meeting, monkeypatch):
    d, calls = meeting
    monkeypatch.setenv("PLAUD_ASR_BACKEND", "mlx")
    run(d)
    monkeypatch.setenv("PLAUD_ASR_BACKEND", "faster-whisper")
    run(d)
    assert calls["transcribe"] == 2
