import json
from dataclasses import asdict
from pathlib import Path

from plaud.align import assign_speakers, group_utterances
from plaud.audio import duration, to_wav
from plaud.diarize import diarize
from plaud.models import Turn, Word, dump, load
from plaud.render import render_transcript
from plaud.report import build_prompt, load_template, run_claude
from plaud.transcribe import transcribe


def _todo(path: Path, force: bool) -> bool:
    return force or not path.exists()


def process(meeting_dir: Path, templates: list[str], model: str,
            language: str | None, num_speakers: int | None, device: str,
            force: bool = False, log=print) -> list[Path]:
    meta = json.loads((meeting_dir / "meeting.json").read_text(encoding="utf-8"))
    audio = next(meeting_dir.glob("audio.*"))

    # Separate name so an original .wav recording is never overwritten.
    wav = meeting_dir / "audio16k.wav"
    if _todo(wav, force):
        log("Converting audio…")
        to_wav(audio, wav)

    asr = meeting_dir / "asr.json"
    if _todo(asr, force):
        log(f"Transcribing with {model}…")
        words, lang = transcribe(wav, model=model, language=language)
        asr.write_text(json.dumps({"language": lang, "words": [asdict(w) for w in words]},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    asr_data = json.loads(asr.read_text(encoding="utf-8"))
    words = [Word(**w) for w in asr_data["words"]]
    lang = asr_data["language"]

    diar = meeting_dir / "diar.json"
    if _todo(diar, force):
        log("Diarizing…")
        dump(diarize(wav, num_speakers=num_speakers, device=device), diar)
    turns = load(Turn, diar)

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
