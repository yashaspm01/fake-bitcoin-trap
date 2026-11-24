import numpy as np
import pytest

from trap.backtest import ma_crossover_signals, strategy_returns
from trap.metrics import sharpe_ratio
from trap.prices import make_fake_prices
from trap.search import search_grid


def test_grid_skips_invalid_pairs():
    prices = make_fake_prices(300, seed=1)
    table, _ = search_grid(prices, [5, 10], [10, 20])
    assert len(table) == 3  # (5,10) (5,20) (10,20); (10,10) is skipped


def test_matches_single_strategy_backtest():
    prices = np.array(make_fake_prices(400, seed=3))
    table, _ = search_grid(prices, [5], [20])
    expected = sharpe_ratio(strategy_returns(prices, ma_crossover_signals(prices, 5, 20)))
    assert table.loc[0, "sharpe"] == pytest.approx(expected)


def test_returns_dict_matches_table():
    prices = make_fake_prices(300, seed=2)
    table, returns = search_grid(prices, [5, 10], [20, 30])
    assert set(table["name"]) == set(returns)
