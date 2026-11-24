"""Fast grid search over moving-average crossover strategies."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from trap.backtest import strategy_returns
from trap.metrics import sharpe_ratio


def rolling_means(prices: np.ndarray, windows: Sequence[int]) -> dict[int, np.ndarray]:
    """Compute each distinct rolling mean once, not once per strategy."""
    series = pd.Series(prices)
    return {w: series.rolling(w).mean().to_numpy() for w in windows}


def search_grid(
    prices: ArrayLike,
    fasts: Sequence[int],
    slows: Sequence[int],
    fee: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Backtest every valid (fast, slow) pair.

    Returns a table sorted by per-day Sharpe ("sr", best first) and a dict
    of each strategy's daily returns. "sharpe" is the annualized version.
    """
    p = np.asarray(prices, dtype=float)
    means = rolling_means(p, sorted(set(fasts) | set(slows)))

    rows = []
    returns_by_name: dict[str, np.ndarray] = {}
    for fast in fasts:
        for slow in slows:
            if fast >= slow:
                continue
            name = f"ma_cross_{fast}_{slow}"
            signals = (means[fast] > means[slow]).astype(int)
            returns = strategy_returns(p, signals, fee)
            returns_by_name[name] = returns
            rows.append(
                {
                    "name": name,
                    "fast": fast,
                    "slow": slow,
                    "sr": sharpe_ratio(returns, periods_per_year=1),
                    "sharpe": sharpe_ratio(returns),
                }
            )
    table = pd.DataFrame(rows).sort_values("sr", ascending=False).reset_index(drop=True)
    return table, returns_by_name
