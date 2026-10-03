import os
import subprocess
from pathlib import Path

import soundfile as sf


class AudioError(RuntimeError):
    pass


def to_wav(src: Path, dst: Path) -> Path:
    """Convert any ffmpeg-readable audio to 16 kHz mono wav (what both models expect)."""
    # Write to a temp name first so an interrupted conversion never looks finished.
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-i", str(src), "-ac", "1", "-ar", "16000", str(tmp)],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise AudioError("ffmpeg not found on PATH") from None
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise AudioError(f"ffmpeg could not read {src.name}: {e.stderr.strip()}") from None
    os.replace(tmp, dst)
    return dst


def duration(wav: Path) -> float:
    return sf.info(str(wav)).duration
