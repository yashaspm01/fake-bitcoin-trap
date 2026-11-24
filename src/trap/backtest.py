"""Vectorized backtesting with NumPy and pandas."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def daily_returns(prices: ArrayLike) -> np.ndarray:
    """Return from day t to day t+1, as a fraction. Length is n - 1."""
    p = np.asarray(prices, dtype=float)
    return p[1:] / p[:-1] - 1.0


def ma_crossover_signals(prices: ArrayLike, fast: int, slow: int) -> np.ndarray:
    """1 when the fast moving average is above the slow one, else 0."""
    if not 0 < fast < slow:
        raise ValueError("need 0 < fast < slow")
    s = pd.Series(np.asarray(prices, dtype=float))
    fast_ma = s.rolling(fast).mean().to_numpy()
    slow_ma = s.rolling(slow).mean().to_numpy()
    # Comparisons with NaN (not enough history yet) are False, so signal = 0.
    return (fast_ma > slow_ma).astype(int)


def strategy_returns(prices: ArrayLike, signals: ArrayLike, fee: float = 0.001) -> np.ndarray:
    """Daily strategy returns after trading fees.

    The signal computed at the close of day t earns the return from t to t+1,
    never the return that led to it. That one-day shift prevents lookahead bias.
    """
    p = np.asarray(prices, dtype=float)
    sig = np.asarray(signals, dtype=float)
    if len(p) != len(sig):
        raise ValueError("prices and signals must have the same length")

    returns = daily_returns(p)
    position = sig[:-1]  # position held from day t to t+1
    turnover = np.abs(np.diff(position, prepend=0.0))  # 1 on each buy or sell
    return position * returns - turnover * fee
