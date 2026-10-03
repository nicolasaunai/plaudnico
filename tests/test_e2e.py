import os
import subprocess

import pytest

from plaud import cli


@pytest.mark.slow
def test_two_voices_end_to_end(tmp_path, monkeypatch, tts):
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")
    monkeypatch.setenv("PLAUD_ARCHIVE", str(tmp_path / "arch"))

    parts = []
    lines = [
        (0, "Good morning. Today we review the simulation budget for next year."),
        (1, "Thanks. I think we need two more weeks on the cluster."),
        (0, "Agreed. Please send the request before Friday."),
        (1, "I will send it tomorrow."),
    ]
    for i, (voice, text) in enumerate(lines):
        parts.append(tts(text, voice, tmp_path / str(i)))
    listing = tmp_path / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    audio = tmp_path / "meeting.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(listing), str(audio)], check=True)

    assert cli.main(["process", str(audio), "--template", "none", "--speakers", "2"]) == 0
    [d] = list((tmp_path / "arch").glob("*/*"))
    t = (d / "transcript.md").read_text()
    assert "- Language: en" in t
    assert "Speaker 1" in t and "Speaker 2" in t
    assert "budget" in t.lower()
