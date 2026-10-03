import subprocess
from pathlib import Path

import soundfile as sf


def to_wav(src: Path, dst: Path) -> Path:
    """Convert any ffmpeg-readable audio to 16 kHz mono wav (what both models expect)."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )
    return dst


def duration(wav: Path) -> float:
    return sf.info(str(wav)).duration
