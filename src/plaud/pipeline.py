import json
import os
from dataclasses import asdict
from pathlib import Path

from plaud.align import assign_speakers, group_utterances
from plaud.audio import duration, to_wav
from plaud.diarize import diarize
from plaud.models import Turn, Word, dump
from plaud.render import render_transcript
from plaud.report import build_prompt, load_template, run_claude
from plaud.transcribe import backend, transcribe


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _cached(path: Path, params: dict, force: bool) -> dict | None:
    """Return the stage output if it exists and was made with the same parameters."""
    if force or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("params") != params:
        return None
    return data


def process(meeting_dir: Path, templates: list[str], model: str,
            language: str | None, num_speakers: int | None, device: str,
            force: bool = False, log=print) -> list[Path]:
    meta = json.loads((meeting_dir / "meeting.json").read_text(encoding="utf-8"))
    audio = next(meeting_dir.glob("audio.*"))

    # Separate name so an original .wav recording is never overwritten.
    wav = meeting_dir / "audio16k.wav"
    if force or not wav.exists():
        log("Converting audio…")
        to_wav(audio, wav)

    # Diarization first: a Hugging Face auth problem shows up before the long transcription.
    diar = meeting_dir / "diar.json"
    diar_params = {"num_speakers": num_speakers}
    diar_data = _cached(diar, diar_params, force)
    if diar_data is None:
        log("Diarizing…")
        turns = diarize(wav, num_speakers=num_speakers, device=device)
        diar_data = {"params": diar_params, "turns": [asdict(t) for t in turns]}
        _write_atomic(diar, json.dumps(diar_data, ensure_ascii=False, indent=1))
    turns = [Turn(**t) for t in diar_data["turns"]]

    asr = meeting_dir / "asr.json"
    asr_params = {"backend": backend(), "model": model, "language": language}
    asr_data = _cached(asr, asr_params, force)
    if asr_data is None:
        log(f"Transcribing with {model}…")
        words, lang = transcribe(wav, model=model, language=language)
        asr_data = {"params": asr_params, "language": lang,
                    "words": [asdict(w) for w in words]}
        _write_atomic(asr, json.dumps(asr_data, ensure_ascii=False, indent=1))
    words = [Word(**w) for w in asr_data["words"]]
    lang = asr_data["language"]

    words = assign_speakers(words, turns)
    dump(words, meeting_dir / "turns.json")
    utts = group_utterances(words)
    transcript = render_transcript(utts, meta["title"], meta["start"], lang, duration(wav))
    (meeting_dir / "transcript.md").write_text(transcript, encoding="utf-8")
    log(f"Transcript: {meeting_dir / 'transcript.md'}")

    if not utts:
        log("No speech detected: no report written.")
        return []

    reports = []
    for name in templates:
        log(f"Writing report '{name}' with Claude…")
        out = meeting_dir / f"report-{name}.md"
        out.write_text(run_claude(build_prompt(load_template(name), transcript)),
                       encoding="utf-8")
        reports.append(out)
        log(f"Report: {out}")
    return reports
