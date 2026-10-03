import subprocess

import pytest

from plaud.report import ReportError, build_prompt, load_template, run_claude


def test_load_template_shipped():
    assert "Actions" in load_template("minutes")


def test_load_unknown_template(tmp_path):
    (tmp_path / "minutes.md").write_text("x")
    with pytest.raises(ReportError, match="Unknown template 'nope'.*minutes"):
        load_template("nope", tmp_path)


def test_build_prompt():
    p = build_prompt("TEMPLATE BODY", "TRANSCRIPT BODY")
    assert "TEMPLATE BODY" in p and "TRANSCRIPT BODY" in p
    assert p.index("TEMPLATE BODY") < p.index("TRANSCRIPT BODY")
    assert "never guess" in p.lower()
    assert "language" in p.lower()


def _runner(returncode=0, stdout="# Minutes\n", stderr=""):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    return run, calls


def test_run_claude_no_tools_prompt_on_stdin():
    run, calls = _runner()
    assert run_claude("PROMPT", runner=run) == "# Minutes\n"
    cmd, kw = calls[0]
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert kw["input"] == "PROMPT"


def test_run_claude_failure():
    run, _ = _runner(returncode=1, stdout="", stderr="auth error")
    with pytest.raises(ReportError, match="auth error"):
        run_claude("P", runner=run)


def test_run_claude_empty_output():
    run, _ = _runner(stdout="  \n")
    with pytest.raises(ReportError, match="empty"):
        run_claude("P", runner=run)
