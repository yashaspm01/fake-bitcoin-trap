import pytest

from trap.backtest import ma_crossover_signals
from trap.prices import make_fake_prices


@pytest.mark.parametrize("n", [0, -1, -100])
def test_fake_prices_reject_non_positive_n(n):
    with pytest.raises(ValueError):
        make_fake_prices(n)


@pytest.mark.parametrize("fast, slow", [(0, 5), (5, 5), (10, 5)])
def test_signals_reject_bad_windows(fast, slow):
    with pytest.raises(ValueError):
        ma_crossover_signals([1.0] * 20, fast, slow)
