import subprocess

import soundfile as sf

from plaud.audio import duration, to_wav


def _make_m4a(path, seconds=1.0):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ac", "2", "-ar", "44100", str(path)],
        check=True,
    )


def test_to_wav_mono_16k(tmp_path):
    src = tmp_path / "Nouvel enregistrement.m4a"  # space in name, like Voice Memos
    _make_m4a(src)
    wav = to_wav(src, tmp_path / "audio.wav")
    info = sf.info(str(wav))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert abs(duration(wav) - 1.0) < 0.1


def test_corrupt_audio_raises_audio_error(tmp_path):
    import pytest

    from plaud.audio import AudioError

    src = tmp_path / "bad.m4a"
    src.write_bytes(b"not audio at all")
    dst = tmp_path / "audio16k.wav"
    with pytest.raises(AudioError, match="bad.m4a"):
        to_wav(src, dst)
    assert not dst.exists()
    assert list(tmp_path.iterdir()) == [src]  # no temp leftovers
