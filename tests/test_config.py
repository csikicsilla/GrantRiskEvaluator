import pytest

from grantrisk import config


def test_default_config_loads():
    cfg = config.load()
    assert cfg.data_root == config.REPO_ROOT / "data"


def test_relative_sources_resolve_outside_the_repository():
    cfg = config.load()
    # The old code line sits next to the repository, in 30_Kód.
    assert cfg.source("gold_csv").parent.parent.name == "20_GrantRiskEstimator"


def test_absolute_path_is_kept(tmp_path):
    cfg = config.load()
    assert cfg.resolve(tmp_path) == tmp_path


def test_missing_data_root_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("sources: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data_root"):
        config.load(bad)


def test_environment_variable_selects_the_file(tmp_path, monkeypatch):
    other = tmp_path / "other.yaml"
    other.write_text(f"data_root: {tmp_path.as_posix()}\n", encoding="utf-8")
    monkeypatch.setenv("GRANTRISK_CONFIG", str(other))
    assert config.load().data_root == tmp_path
