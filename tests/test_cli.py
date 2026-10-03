import pytest

from plaud import cli


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAUD_ARCHIVE", str(tmp_path / "arch"))
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    seen = {}

    def fake_process(d, **kw):
        seen["dir"], seen["kw"] = d, kw
        return []

    monkeypatch.setattr(cli, "process", fake_process)
    return tmp_path, audio, seen


def test_process_ok(env, capsys):
    tmp, audio, seen = env
    assert cli.main(["process", str(audio), "--speakers", "3", "--language", "fr"]) == 0
    assert seen["dir"].parent.parent == tmp / "arch"
    assert seen["kw"]["templates"] == ["minutes"]
    assert seen["kw"]["num_speakers"] == 3
    assert seen["kw"]["language"] == "fr"
    assert str(seen["dir"]) in capsys.readouterr().out


def test_template_none(env):
    _, audio, seen = env
    assert cli.main(["process", str(audio), "--template", "none"]) == 0
    assert seen["kw"]["templates"] == []


def test_missing_file(env, capsys):
    tmp, _, seen = env
    assert cli.main(["process", str(tmp / "nope.m4a")]) == 2
    assert "not found" in capsys.readouterr().err
    assert seen == {}


def test_missing_token_fails_before_anything(env, monkeypatch, capsys):
    tmp, audio, seen = env
    monkeypatch.delenv("HF_TOKEN")
    assert cli.main(["process", str(audio)]) == 1
    assert "HF_TOKEN" in capsys.readouterr().err
    assert seen == {}
    assert not (tmp / "arch").exists()


def test_unknown_template_fails_before_anything(env, capsys):
    tmp, audio, seen = env
    assert cli.main(["process", str(audio), "--template", "nope"]) == 1
    assert "Unknown template 'nope'" in capsys.readouterr().err
    assert not (tmp / "arch").exists()


def test_report_error_exit_1(env, monkeypatch, capsys):
    _, audio, _ = env
    from plaud.report import ReportError

    def boom(d, **kw):
        raise ReportError("claude -p failed (1): auth error")

    monkeypatch.setattr(cli, "process", boom)
    assert cli.main(["process", str(audio)]) == 1
    assert "auth error" in capsys.readouterr().err
