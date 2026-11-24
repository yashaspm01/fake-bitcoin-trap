import pytest

from trap import cli, experiment
from trap.cli import build_parser, main


def test_parser_defaults():
    args = build_parser().parse_args(["noise"])
    assert (args.series, args.days) == (20, 1500)


def test_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_real_without_data_returns_error(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "REAL_PATH", tmp_path / "missing.csv")
    assert main(["real"]) == 1


def test_real_command_prints_table(saved_closes, capsys):
    assert main(["real"]) == 0
    assert "real_btcusdt" in capsys.readouterr().out


def test_holdout_command_prints_result(saved_closes, capsys):
    assert main(["holdout"]) == 0
    out = capsys.readouterr().out
    assert "best_on_train" in out and "test_sharpe" in out


def test_noise_command_small_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path)
    assert main(["noise", "--series", "2", "--days", "300"]) == 0
    assert "Looks real" in capsys.readouterr().out
    assert (tmp_path / "day4_fake_noise.csv").exists()
