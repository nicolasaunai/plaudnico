from pathlib import Path

from plaud.models import Word

# Chosen provisionally; Task 9 confirms or replaces it after the real-recording test.
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def words_from_result(result: dict) -> list[Word]:
    words = []
    for seg in result.get("segments", []):
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
    )
    return words_from_result(result), result.get("language") or language or "unknown"
