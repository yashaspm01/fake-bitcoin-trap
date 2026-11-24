import numpy as np
import pytest

from trap.holdout import holdout_check
from trap.prices import make_fake_prices
from trap.search import search_grid


def test_holdout_returns_expected_keys(fake_prices):
    result = holdout_check(fake_prices, [5, 10], [30, 50], train_fraction=0.5)
    assert set(result) == {
        "best_on_train",
        "train_sharpe",
        "test_sharpe",
        "buy_and_hold_test_sharpe",
    }
    assert str(result["best_on_train"]).startswith("ma_cross_")


@pytest.mark.parametrize("fraction", [0, 1, 1.5, -0.2, 0.1, 0.9])
def test_holdout_rejects_bad_splits(fraction):
    with pytest.raises(ValueError):
        holdout_check(make_fake_prices(100, seed=1), [5], [20], train_fraction=fraction)


def test_selection_uses_training_data_only(fake_prices):
    prices = np.array(fake_prices)
    result = holdout_check(prices, [5, 10], [30, 50], 0.5)
    table, _ = search_grid(prices[:300], [5, 10], [30, 50])
    assert result["best_on_train"] == table.iloc[0]["name"]


def test_changing_the_future_cannot_change_the_choice(fake_prices):
    prices = np.array(fake_prices)
    altered = prices.copy()
    altered[300:] *= 2  # rewrite the "test" half of history
    before = holdout_check(prices, [5, 10], [30, 50], 0.5)
    after = holdout_check(altered, [5, 10], [30, 50], 0.5)
    assert before["best_on_train"] == after["best_on_train"]
