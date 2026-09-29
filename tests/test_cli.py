import pytest

from grantrisk import cli


def test_help_lists_every_stage(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name, code, _ in cli.STAGES:
        assert name in out
        assert code in out
    assert "run-all" in out


@pytest.mark.parametrize(
    "command", [name for name, _, _ in cli.STAGES if name not in cli.IMPLEMENTED] + ["run-all"]
)
def test_stage_stub_reports_not_implemented(command, capsys):
    assert cli.main([command]) == cli.EXIT_NOT_IMPLEMENTED
    assert "not implemented" in capsys.readouterr().err


def test_label_needs_its_input_runs():
    with pytest.raises(SystemExit) as exc:
        cli.main(["label"])
    assert exc.value.code == 2


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit) as exc:
        cli.main(["no-such-stage"])
    assert exc.value.code == 2
