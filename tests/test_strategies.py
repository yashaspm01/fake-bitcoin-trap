import pytest

from trap.strategies import (
    BuyAndHold,
    MovingAverageCrossover,
    Strategy,
    count_trades,
    moving_average,
    strategy_grid,
)


def test_moving_average_values():
    assert moving_average([1.0, 2.0, 3.0, 4.0], 2) == [None, 1.5, 2.5, 3.5]


def test_moving_average_bad_window():
    with pytest.raises(ValueError):
        moving_average([1.0, 2.0], 0)


def test_crossover_rejects_fast_not_smaller_than_slow():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast=10, slow=5)


def test_crossover_on_steady_uptrend():
    prices = [float(x) for x in range(1, 21)]
    sig = MovingAverageCrossover(fast=2, slow=5).signals(prices)
    assert sig[:4] == [0, 0, 0, 0]
    assert sig[4:] == [1] * 16


def test_buy_and_hold_is_always_in():
    assert BuyAndHold().signals([1.0, 2.0, 3.0]) == [1, 1, 1]


def test_count_trades():
    assert count_trades([0, 0, 1, 1, 0, 1]) == 3


def test_strategy_base_class_cannot_be_used_directly():
    with pytest.raises(TypeError):
        Strategy()


def test_grid_keeps_only_valid_pairs():
    names = {s.name for s in strategy_grid([5, 10], [10, 20])}
    assert names == {"ma_cross_5_10", "ma_cross_5_20", "ma_cross_10_20"}
