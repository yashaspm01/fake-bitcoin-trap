import numpy as np
import pytest

from trap.stats import (
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
    skew_kurtosis,
)


def test_skew_kurtosis_of_normal_data():
    data = np.random.default_rng(0).normal(size=200_000)
    skew, kurt = skew_kurtosis(data)
    assert abs(skew) < 0.05
    assert abs(kurt - 3) < 0.1


def test_expected_max_sharpe_grows_with_trials():
    few = expected_max_sharpe(10, 0.01)
    many = expected_max_sharpe(1000, 0.01)
    assert 0 < few < many


def test_one_trial_needs_no_deflation():
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_psr_is_half_when_sr_equals_benchmark():
    assert probabilistic_sharpe(0.1, 0.1, 500, 0.0, 3.0) == pytest.approx(0.5)


def test_psr_grows_with_more_data():
    short = probabilistic_sharpe(0.1, 0.0, 100, 0.0, 3.0)
    long = probabilistic_sharpe(0.1, 0.0, 1000, 0.0, 3.0)
    assert short < long


def test_deflated_is_lower_than_naive():
    rng = np.random.default_rng(0)
    best_returns = rng.normal(0.001, 0.01, 1500)
    trial_srs = rng.normal(0.0, 0.03, 500)
    result = deflated_sharpe(best_returns, trial_srs)
    assert result.sr0 > 0
    assert 0.0 <= result.dsr < result.psr_vs_zero <= 1.0
