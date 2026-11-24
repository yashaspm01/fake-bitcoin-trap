from trap import experiment
from trap.cli import main
from trap.report import build_report


def test_report_includes_real_section_when_data_exists(saved_closes, tmp_path):
    page = build_report(tmp_path / "site", n_series=2).read_text(encoding="utf-8")
    assert "Pure-noise" in page
    assert "Real BTCUSDT" in page
    assert "best_on_train" in page


def test_report_without_real_data_still_builds(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "REAL_PATH", tmp_path / "missing.csv")
    page = build_report(tmp_path / "site", n_series=2).read_text(encoding="utf-8")
    assert "not available" in page


def test_report_command(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "REAL_PATH", tmp_path / "missing.csv")
    out = tmp_path / "out"
    assert main(["report", "--out", str(out), "--series", "2"]) == 0
    assert (out / "index.html").exists()
