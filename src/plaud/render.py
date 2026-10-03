from plaud.models import Utterance

NO_SPEECH = "_No speech detected._"


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_transcript(utts: list[Utterance], title: str, start: str,
                      language: str, duration: float) -> str:
    n_speakers = len({u.speaker for u in utts})
    lines = [
        f"# {title}",
        "",
        f"- Start: {start}",
        f"- Duration: {fmt_ts(duration)}",
        f"- Language: {language}",
        f"- Speakers: {n_speakers}",
        "",
    ]
    if not utts:
        lines.append(NO_SPEECH)
    for u in utts:
        lines += [f"**[{fmt_ts(u.start)}] {u.speaker}:** {u.text}", ""]
    return "\n".join(lines).rstrip() + "\n"
