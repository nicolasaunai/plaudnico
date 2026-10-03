from plaud.models import Utterance
from plaud.render import NO_SPEECH, fmt_ts, render_transcript


def test_fmt_ts():
    assert fmt_ts(0) == "00:00:00"
    assert fmt_ts(3725.9) == "01:02:05"


def test_render_transcript():
    utts = [Utterance("Speaker 1", 12.0, 14.0, "Bonjour."),
            Utterance("Speaker 2", 75.0, 80.0, "Salut.")]
    md = render_transcript(utts, "Réunion équipe", "2026-10-03T10:15", "fr", 90.0)
    assert md.startswith("# Réunion équipe\n")
    assert "- Language: fr" in md
    assert "- Duration: 00:01:30" in md
    assert "- Speakers: 2" in md
    assert "**[00:00:12] Speaker 1:** Bonjour." in md
    assert "**[00:01:15] Speaker 2:** Salut." in md


def test_render_empty_transcript():
    md = render_transcript([], "Silence", "2026-10-03T10:15", "unknown", 5.0)
    assert NO_SPEECH in md
    assert "- Speakers: 0" in md
