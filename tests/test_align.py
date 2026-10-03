from plaud.align import UNKNOWN, assign_speakers, group_utterances
from plaud.models import Turn, Utterance, Word

TURNS = [Turn(0.0, 2.0, "Speaker 1"), Turn(2.0, 4.0, "Speaker 2")]


def speakers(words):
    return [w.speaker for w in words]


def test_word_inside_turn():
    assert speakers(assign_speakers([Word("a", 0.5, 0.9)], TURNS)) == ["Speaker 1"]


def test_word_straddling_turns_goes_to_larger_overlap():
    assert speakers(assign_speakers([Word("a", 1.8, 2.5)], TURNS)) == ["Speaker 2"]


def test_zero_length_word_inside_turn():
    assert speakers(assign_speakers([Word("a", 1.0, 1.0)], TURNS)) == ["Speaker 1"]


def test_word_in_small_gap_goes_to_nearest_turn():
    turns = [Turn(0.0, 1.0, "Speaker 1"), Turn(5.0, 6.0, "Speaker 2")]
    assert speakers(assign_speakers([Word("a", 1.2, 1.4)], turns)) == ["Speaker 1"]


def test_word_far_from_any_turn_is_unknown():
    turns = [Turn(0.0, 1.0, "Speaker 1")]
    assert speakers(assign_speakers([Word("a", 10.0, 10.5)], turns)) == [UNKNOWN]


def test_no_turns_at_all():
    assert speakers(assign_speakers([Word("a", 0.0, 1.0)], [])) == [UNKNOWN]


def test_no_words():
    assert assign_speakers([], TURNS) == []


def test_group_utterances():
    words = [
        Word("Bonjour", 0.0, 0.4, "Speaker 1"),
        Word("à", 0.4, 0.5, "Speaker 1"),
        Word("Merci.", 2.1, 2.5, "Speaker 2"),
        Word("Bien.", 3.0, 3.4, "Speaker 1"),
    ]
    assert group_utterances(words) == [
        Utterance("Speaker 1", 0.0, 0.5, "Bonjour à"),
        Utterance("Speaker 2", 2.1, 2.5, "Merci."),
        Utterance("Speaker 1", 3.0, 3.4, "Bien."),
    ]


def test_group_utterances_empty():
    assert group_utterances([]) == []
