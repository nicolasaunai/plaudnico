import os
import subprocess

import pytest

from plaud import cli


def _voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


@pytest.mark.slow
def test_two_voices_end_to_end(tmp_path, monkeypatch):
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")
    voices = _voices()
    if not {"Daniel", "Samantha"} <= voices:
        pytest.skip("macOS voices Daniel and Samantha not installed")
    monkeypatch.setenv("PLAUD_ARCHIVE", str(tmp_path / "arch"))

    parts = []
    lines = [
        ("Daniel", "Good morning. Today we review the simulation budget for next year."),
        ("Samantha", "Thanks. I think we need two more weeks on the cluster."),
        ("Daniel", "Agreed. Please send the request before Friday."),
        ("Samantha", "I will send it tomorrow."),
    ]
    for i, (voice, text) in enumerate(lines):
        p = tmp_path / f"{i}.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(p), text], check=True)
        parts.append(p)
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
