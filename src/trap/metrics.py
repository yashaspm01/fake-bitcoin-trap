"""Performance metrics."""

import numpy as np
from numpy.typing import ArrayLike


def sharpe_ratio(returns: ArrayLike, periods_per_year: int = 365) -> float:
    """Annualized Sharpe ratio (risk-free rate assumed 0).

    Bitcoin trades every day, so we annualize with 365, not the 252
    used for stocks.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def total_return(returns: ArrayLike) -> float:
    """Compounded return, e.g. 0.25 means +25%."""
    r = np.asarray(returns, dtype=float)
    return float(np.prod(1.0 + r) - 1.0)
