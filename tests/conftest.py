import shutil
import subprocess

import pytest

MAC_VOICES = ["Daniel", "Samantha"]
ESPEAK_VOICES = ["en-us+m3", "en-gb+f3"]


@pytest.fixture
def tts():
    """speak(text, voice, path) -> audio file, with macOS `say` or Linux `espeak-ng`."""
    if shutil.which("say"):
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        if not set(MAC_VOICES) <= {line.split()[0] for line in out.splitlines() if line.strip()}:
            pytest.skip(f"macOS voices {MAC_VOICES} not installed")

        def speak(text, voice, path):
            out = path.with_suffix(".aiff")
            subprocess.run(["say", "-v", MAC_VOICES[voice], "-o", str(out), text], check=True)
            return out
    elif shutil.which("espeak-ng"):
        def speak(text, voice, path):
            out = path.with_suffix(".wav")
            subprocess.run(["espeak-ng", "-v", ESPEAK_VOICES[voice], "-w", str(out), text],
                           check=True)
            return out
    else:
        pytest.skip("no text-to-speech tool (say or espeak-ng)")
    return speak
