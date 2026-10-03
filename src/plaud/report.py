import subprocess
import tempfile
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

PROMPT = """You write a meeting report from a transcript.

Rules:
- Write in the language of the transcript (see its "Language" line).
- Speakers are anonymous labels ("Speaker 1", "Speaker 2", ...). Use these labels and never guess real names. If a speaker states their own name in the transcript, you may write "Speaker 2 (introduced as X)".
- Report only what is in the transcript. Do not invent facts, dates, owners or decisions.
- The transcript comes from automatic speech recognition: some words may be wrong. Do not repeat obvious recognition errors.
- Output Markdown only, with no preamble.

## Template

{template}

## Transcript

{transcript}
"""


class ReportError(RuntimeError):
    pass


def load_template(name: str, templates_dir: Path = TEMPLATES_DIR) -> str:
    path = templates_dir / f"{name}.md"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in templates_dir.glob("*.md")))
        raise ReportError(f"Unknown template '{name}'. Available: {available}")
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, transcript: str) -> str:
    return PROMPT.format(template=template.strip(), transcript=transcript.strip())


CLAUDE_TIMEOUT = 600


def run_claude(prompt: str, runner=subprocess.run) -> str:
    # No tools, no MCP servers, no user/project settings or CLAUDE.md, neutral cwd:
    # Claude only reads the prompt text and returns Markdown.
    cmd = ["claude", "-p", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--output-format", "text"]
    try:
        with tempfile.TemporaryDirectory() as cwd:
            res = runner(cmd, input=prompt, capture_output=True, text=True,
                         encoding="utf-8", cwd=cwd, timeout=CLAUDE_TIMEOUT)
    except FileNotFoundError:
        raise ReportError("claude CLI not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise ReportError(f"claude -p timed out after {CLAUDE_TIMEOUT} s") from None
    if res.returncode != 0:
        raise ReportError(f"claude -p failed ({res.returncode}): {res.stderr.strip()}")
    if not res.stdout.strip():
        raise ReportError("claude -p returned an empty report")
    return res.stdout
