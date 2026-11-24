"""Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

Every Sharpe ratio in this file is per-period (per day), NOT annualized.
"""

from math import e, sqrt
from statistics import NormalDist
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from trap.metrics import sharpe_ratio

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()  # standard normal: .cdf(x) and .inv_cdf(p)


class DsrResult(NamedTuple):
    sr: float  # best strategy's per-day Sharpe
    sr0: float  # Sharpe that pure luck is expected to reach after N trials
    psr_vs_zero: float  # naive confidence: "ignore the search, compare to 0"
    dsr: float  # honest confidence: compare to sr0 instead


def skew_kurtosis(returns: ArrayLike) -> tuple[float, float]:
    """Skewness and (non-excess) kurtosis. A normal distribution gives 0 and 3."""
    r = np.asarray(returns, dtype=float)
    std = r.std()
    if std < 1e-12:
        return 0.0, 3.0
    z = (r - r.mean()) / std
    return float((z**3).mean()), float((z**4).mean())


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected best Sharpe among n_trials strategies that all have zero skill."""
    if n_trials < 2:
        return 0.0
    first = _NORMAL.inv_cdf(1 - 1 / n_trials)
    second = _NORMAL.inv_cdf(1 - 1 / (n_trials * e))
    return sqrt(sr_variance) * ((1 - EULER_GAMMA) * first + EULER_GAMMA * second)


def probabilistic_sharpe(
    sr: float, benchmark_sr: float, n_obs: int, skew: float, kurt: float
) -> float:
    """Probability that the true Sharpe is above benchmark_sr, given the sample."""
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0 or n_obs < 2:
        return 0.0  # variance estimate invalid: treat as not significant
    z = (sr - benchmark_sr) * sqrt(n_obs - 1) / sqrt(denom)
    return _NORMAL.cdf(z)


def deflated_sharpe(best_returns: ArrayLike, trial_srs: ArrayLike) -> DsrResult:
    """Judge the winning strategy after accounting for how many were tried."""
    r = np.asarray(best_returns, dtype=float)
    srs = np.asarray(trial_srs, dtype=float)

    sr = sharpe_ratio(r, periods_per_year=1)
    skew, kurt = skew_kurtosis(r)
    sr0 = expected_max_sharpe(len(srs), float(np.var(srs, ddof=1)))

    return DsrResult(
        sr=sr,
        sr0=sr0,
        psr_vs_zero=probabilistic_sharpe(sr, 0.0, r.size, skew, kurt),
        dsr=probabilistic_sharpe(sr, sr0, r.size, skew, kurt),
    )
