import pandas as pd
import pytest

from trap import experiment
from trap.prices import make_fake_prices


@pytest.fixture
def fake_prices() -> list[float]:
    return make_fake_prices(600, seed=1)


@pytest.fixture
def saved_closes(tmp_path, monkeypatch):
    """A CSV of fake closes, with REAL_PATH pointed at it for the test."""
    path = tmp_path / "closes.csv"
    pd.DataFrame({"close": make_fake_prices(1200, seed=7)}).to_csv(path)
    monkeypatch.setattr(experiment, "REAL_PATH", path)
    return path
