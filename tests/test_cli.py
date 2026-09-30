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
    "argv",
    [["label"], ["extract"], ["extract", "--extractor", "regex", "--c1-run", "x"], ["consolidate", "--manual-run", "x"],
     ["convert"], ["validate"], ["validate", "--manual-run", "x", "--l3-run", "y"],
     ["represent"], ["train", "--m1-run", "x"], ["extract", "--extractor", "manual", "--c2-run", "x"],
     ["extract", "--extractor", "llm", "--c1-run", "x"], ["evaluate"]],
)
def test_implemented_commands_need_their_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit) as exc:
        cli.main(["no-such-stage"])
    assert exc.value.code == 2


def test_extract_takes_a_budget_ceiling():
    args = cli.build_parser().parse_args(
        ["extract", "--extractor", "llm", "--c2-run", "x", "--gold-only", "--budget-usd", "4"])
    assert args.budget_usd == 4.0
