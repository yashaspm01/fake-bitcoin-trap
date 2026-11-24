"""Trading strategies and helpers (pure Python, no NumPy yet)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


def moving_average(prices: list[float], window: int) -> list[float | None]:
    """Average of the last `window` prices. None until enough data exists."""
    if window <= 0:
        raise ValueError("window must be positive")
    return [
        None if i + 1 < window else sum(prices[i + 1 - window : i + 1]) / window
        for i in range(len(prices))
    ]


class Strategy(ABC):
    """Every strategy must say its name and turn prices into signals."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, readable identifier."""

    @abstractmethod
    def signals(self, prices: list[float]) -> list[int]:
        """Return one value per price: 1 = holding Bitcoin, 0 = in cash."""


@dataclass(frozen=True)
class BuyAndHold(Strategy):
    """Baseline: always hold. Every other strategy must beat this."""

    @property
    def name(self) -> str:
        return "buy_and_hold"

    def signals(self, prices: list[float]) -> list[int]:
        return [1] * len(prices)


@dataclass(frozen=True)
class MovingAverageCrossover(Strategy):
    """Hold when the fast average is above the slow average."""

    fast: int
    slow: int

    def __post_init__(self) -> None:
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("windows must be positive")
        if self.fast >= self.slow:
            raise ValueError("fast must be smaller than slow")

    @property
    def name(self) -> str:
        return f"ma_cross_{self.fast}_{self.slow}"

    def signals(self, prices: list[float]) -> list[int]:
        fast_ma = moving_average(prices, self.fast)
        slow_ma = moving_average(prices, self.slow)
        return [
            1 if f is not None and s is not None and f > s else 0 for f, s in zip(fast_ma, slow_ma)
        ]


def count_trades(signals: list[int]) -> int:
    """How many times the position changed (buy or sell)."""
    return sum(1 for a, b in zip(signals, signals[1:]) if a != b)


def strategy_grid(fasts: list[int], slows: list[int]) -> list[MovingAverageCrossover]:
    """Every valid (fast, slow) pair as a strategy object."""
    return [MovingAverageCrossover(fast=f, slow=s) for f in fasts for s in slows if f < s]
