import argparse
import sys
from pathlib import Path

from plaud.audio import AudioError
from plaud.diarize import DiarizeError, hf_token
from plaud.meeting import create_or_get
from plaud.pipeline import process
from plaud.report import ReportError, load_template
from plaud.transcribe import TranscribeError, default_model


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plaud", description="Local meeting transcription and reports.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("process", help="Transcribe an audio file and write reports.")
    pp.add_argument("audio", type=Path)
    pp.add_argument("--template", default="minutes",
                    help="comma-separated template names, or 'none' (default: minutes)")
    pp.add_argument("--model", default=None,
                    help="Whisper model (default: large-v3 for this machine's backend)")
    pp.add_argument("--language", default=None, help="force language (e.g. fr, en); default: detect")
    pp.add_argument("--speakers", type=int, default=None, help="number of speakers, if known")
    pp.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                    help="diarization device (default: auto)")
    pp.add_argument("--force", action="store_true", help="recompute all stages")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.audio.is_file():
        print(f"plaud: file not found: {args.audio}", file=sys.stderr)
        return 2
    templates = [] if args.template == "none" else \
        [t.strip() for t in args.template.split(",") if t.strip()]
    try:
        # Fail fast, before creating folders or loading models.
        hf_token()
        model = args.model or default_model()
        for t in templates:
            load_template(t)
        meeting_dir = create_or_get(args.audio)
        print(meeting_dir)
        process(meeting_dir, templates=templates, model=model,
                language=args.language, num_speakers=args.speakers,
                device=args.device, force=args.force)
    except (AudioError, DiarizeError, ReportError, TranscribeError) as e:
        print(f"plaud: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
