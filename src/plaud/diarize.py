import os
from pathlib import Path

from plaud.models import Turn

# Nothing but transcript text may leave the Mac: pyannote sends usage metrics by default.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Open-source, runs locally. Never switch to "precision-2": it uploads audio.
PIPELINE = "pyannote/speaker-diarization-community-1"


class DiarizeError(RuntimeError):
    pass


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise DiarizeError(
            f"HF_TOKEN is not set. Accept the conditions of {PIPELINE} on "
            "huggingface.co, create a read token, then `export HF_TOKEN=...`."
        )
    return token


def label_speakers(raw: list[tuple[float, float, str]]) -> list[Turn]:
    """Rename pyannote labels to 'Speaker 1..N' in order of first appearance."""
    names: dict[str, str] = {}
    turns = []
    for start, end, spk in sorted(raw, key=lambda t: t[0]):
        if spk not in names:
            names[spk] = f"Speaker {len(names) + 1}"
        turns.append(Turn(start, end, names[spk]))
    return turns


def diarize(wav: Path, num_speakers: int | None = None, device: str = "mps") -> list[Turn]:
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(PIPELINE, token=hf_token())
    pipeline.to(torch.device(device))
    data, sr = sf.read(str(wav), dtype="float32")
    audio = {"waveform": torch.from_numpy(data).unsqueeze(0), "sample_rate": sr}
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    output = pipeline(audio, **kwargs)
    raw = [(t.start, t.end, spk) for t, spk in output.speaker_diarization]
    return label_speakers(raw)
