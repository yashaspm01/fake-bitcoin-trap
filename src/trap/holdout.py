"""Out-of-sample check: pick the best strategy on old data, judge it on newer data."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from trap.backtest import ma_crossover_signals, strategy_returns
from trap.metrics import sharpe_ratio
from trap.search import search_grid
from trap.timing import timed

MIN_DAYS_PER_SIDE = 30


@timed
def holdout_check(
    prices: ArrayLike,
    fasts: Sequence[int],
    slows: Sequence[int],
    train_fraction: float = 0.5,
    fee: float = 0.001,
) -> dict[str, float | str]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    p = np.asarray(prices, dtype=float)
    split = int(len(p) * train_fraction)
    if split < MIN_DAYS_PER_SIDE or len(p) - split < MIN_DAYS_PER_SIDE:
        raise ValueError(f"need at least {MIN_DAYS_PER_SIDE} days on each side of the split")

    # 1) Choose the winner using ONLY the training period.
    table, _ = search_grid(p[:split], fasts, slows, fee)
    best = table.iloc[0]
    fast, slow = (int(x) for x in str(best["name"]).split("_")[2:])

    # 2) Judge it on the later period. Signals use only past prices, so
    #    computing them on the full series is safe (the moving averages are warmed up).
    signals = ma_crossover_signals(p, fast, slow)
    test_returns = strategy_returns(p, signals, fee)[split:]
    hold_returns = strategy_returns(p, np.ones(len(p)), fee)[split:]

    return {
        "best_on_train": str(best["name"]),
        "train_sharpe": float(best["sharpe"]),
        "test_sharpe": sharpe_ratio(test_returns),
        "buy_and_hold_test_sharpe": sharpe_ratio(hold_returns),
    }
