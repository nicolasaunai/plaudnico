from pathlib import Path

from plaud.models import Word

# Chosen provisionally; Task 9 confirms or replaces it after the real-recording test.
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def _likely_hallucination(seg: dict) -> bool:
    # Whisper's own no-speech rule: probably silence and decoded with low confidence.
    return seg.get("no_speech_prob", 0.0) > 0.6 and seg.get("avg_logprob", 0.0) < -1.0


def words_from_result(result: dict) -> list[Word]:
    words = []
    for seg in result.get("segments", []):
        if _likely_hallucination(seg):
            continue
        for w in seg.get("words", []):
            text = w["word"].strip()
            if text:
                words.append(Word(text, float(w["start"]), float(w["end"])))
    return words


def transcribe(wav: Path, model: str = DEFAULT_MODEL,
               language: str | None = None) -> tuple[list[Word], str]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=model,
        word_timestamps=True,
        language=language,
        # Limit repetition loops and invented text on long recordings and silences.
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
    return words_from_result(result), result.get("language") or language or "unknown"
