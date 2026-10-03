import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


def dump(objs: list, path: Path) -> None:
    path.write_text(
        json.dumps([asdict(o) for o in objs], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def load(cls: type, path: Path) -> list:
    return [cls(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
