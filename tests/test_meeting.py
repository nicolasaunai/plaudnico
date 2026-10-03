import json
from datetime import datetime

from plaud.meeting import archive_root, create_or_get, slugify


def test_slugify():
    assert slugify("Nouvel enregistrement 12") == "nouvel-enregistrement-12"
    assert slugify("Réunion d'équipe — PHARE") == "reunion-d-equipe-phare"
    assert slugify("???") == "meeting"


def test_archive_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PLAUD_ARCHIVE", str(tmp_path))
    assert archive_root() == tmp_path


def test_create_meeting(tmp_path):
    audio = tmp_path / "Réunion équipe.m4a"
    audio.write_bytes(b"fake audio")
    when = datetime(2026, 10, 3, 10, 15)
    d = create_or_get(audio, archive=tmp_path / "arch", when=when)
    assert d == tmp_path / "arch" / "2026" / "2026-10-03_1015_reunion-equipe"
    assert (d / "audio.m4a").read_bytes() == b"fake audio"
    meta = json.loads((d / "meeting.json").read_text(encoding="utf-8"))
    assert meta["title"] == "Réunion équipe"
    assert meta["start"] == "2026-10-03T10:15"
    assert len(meta["id"]) == 64


def test_same_file_reuses_folder(tmp_path):
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    arch = tmp_path / "arch"
    d1 = create_or_get(audio, archive=arch, when=datetime(2026, 10, 3, 10, 15))
    d2 = create_or_get(audio, archive=arch, when=datetime(2026, 10, 4, 9, 0))
    assert d1 == d2
    assert len(list(arch.glob("*/*/meeting.json"))) == 1


def test_different_files_same_minute_and_name(tmp_path):
    arch = tmp_path / "arch"
    when = datetime(2026, 10, 3, 10, 15)
    (tmp_path / "x").mkdir()
    a, b = tmp_path / "a.m4a", tmp_path / "x" / "a.m4a"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    d1 = create_or_get(a, archive=arch, when=when)
    d2 = create_or_get(b, archive=arch, when=when)
    assert d1 != d2
    assert d2.name == "2026-10-03_1015_a-2"


def test_interrupted_creation_leaves_no_half_folder(tmp_path, monkeypatch):
    import shutil

    import pytest

    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    arch = tmp_path / "arch"
    when = datetime(2026, 10, 3, 10, 15)

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copy2", boom)
    with pytest.raises(KeyboardInterrupt):
        create_or_get(audio, archive=arch, when=when)
    monkeypatch.undo()
    d = create_or_get(audio, archive=arch, when=when)
    assert d.name == "2026-10-03_1015_a"
    assert [p.name for p in d.parent.iterdir()] == [d.name]

