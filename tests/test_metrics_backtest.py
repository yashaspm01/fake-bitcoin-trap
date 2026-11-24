import numpy as np
import pytest

from trap.backtest import daily_returns, ma_crossover_signals, strategy_returns
from trap.metrics import sharpe_ratio, total_return
from trap.prices import make_fake_prices
from trap.strategies import MovingAverageCrossover


def test_daily_returns():
    np.testing.assert_allclose(daily_returns([100, 110, 121]), [0.1, 0.1])


def test_signal_is_shifted_to_avoid_lookahead():
    # Signal says "buy" on day 1 (price 200). We only earn day 1 -> day 2 (a loss).
    result = strategy_returns([100, 200, 100], [0, 1, 0], fee=0.0)
    np.testing.assert_allclose(result, [0.0, -0.5])


def test_fee_charged_on_entry_and_exit():
    result = strategy_returns([100, 100, 100, 100], [0, 1, 0, 0], fee=0.01)
    np.testing.assert_allclose(result, [0.0, -0.01, -0.01])


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        strategy_returns([1.0, 2.0, 3.0], [1, 1])


def test_vectorized_signals_match_pure_python_version():
    prices = make_fake_prices(300, seed=1)
    slow_way = MovingAverageCrossover(fast=5, slow=20).signals(prices)
    fast_way = ma_crossover_signals(prices, 5, 20)
    assert fast_way.tolist() == slow_way


def test_sharpe_known_value():
    assert sharpe_ratio([0.01, 0.03], periods_per_year=1) == pytest.approx(2**0.5)


def test_sharpe_flat_returns_is_zero():
    assert sharpe_ratio([0.0] * 10) == 0.0


def test_total_return_compounds():
    assert total_return([0.1, 0.1]) == pytest.approx(0.21)
