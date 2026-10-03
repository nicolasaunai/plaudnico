from plaud.models import Turn, Utterance, Word

UNKNOWN = "Unknown speaker"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _distance(x: float, t: Turn) -> float:
    if t.start <= x <= t.end:
        return 0.0
    return min(abs(x - t.start), abs(x - t.end))


def assign_speakers(words: list[Word], turns: list[Turn], max_gap: float = 1.0) -> list[Word]:
    out = []
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            ov = _overlap(w.start, w.end, t.start, t.end)
            if ov > best_ov:
                best, best_ov = t, ov
        if best is None and turns:
            mid = (w.start + w.end) / 2
            nearest = min(turns, key=lambda t: _distance(mid, t))
            if _distance(mid, nearest) <= max_gap:
                best = nearest
        out.append(Word(w.text, w.start, w.end, best.speaker if best else UNKNOWN))
    return out


def group_utterances(words: list[Word]) -> list[Utterance]:
    utts: list[Utterance] = []
    for w in words:
        if utts and utts[-1].speaker == w.speaker:
            utts[-1].end = w.end
            utts[-1].text += w.text
        else:
            utts.append(Utterance(w.speaker, w.start, w.end, w.text))
    for u in utts:
        u.text = u.text.strip()
    return utts
