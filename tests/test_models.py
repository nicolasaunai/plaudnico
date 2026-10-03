from plaud.models import Turn, Utterance, Word, dump, load


def test_roundtrip_words(tmp_path):
    words = [Word("Bonjour", 0.0, 0.5, "Speaker 1"), Word("à", 0.5, 0.6)]
    path = tmp_path / "w.json"
    dump(words, path)
    assert load(Word, path) == words
    assert "à" in path.read_text(encoding="utf-8")  # no \u escapes


def test_roundtrip_turns_and_utterances(tmp_path):
    turns = [Turn(0.0, 1.0, "Speaker 1")]
    utts = [Utterance("Speaker 1", 0.0, 1.0, "Hello there")]
    dump(turns, tmp_path / "t.json")
    dump(utts, tmp_path / "u.json")
    assert load(Turn, tmp_path / "t.json") == turns
    assert load(Utterance, tmp_path / "u.json") == utts
