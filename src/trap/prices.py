"""Fake Bitcoin price generation for the Fake Bitcoin Trap project."""

import random
from collections.abc import Iterator
from itertools import accumulate

START_PRICE = 100.0
DAILY_VOLATILITY = 0.01  # 1% standard deviation per step


def make_fake_prices(n: int, seed: int = 0) -> list[float]:
    """Basic version: build the price list with a plain loop."""
    if n <= 0:
        raise ValueError("n must be positive")

    random.seed(seed)
    prices = [START_PRICE]

    for _ in range(n - 1):
        daily_return = random.gauss(0, DAILY_VOLATILITY)
        prices.append(prices[-1] * (1 + daily_return))

    return prices


def make_fake_prices_v2(n: int, seed: int = 0) -> list[float]:
    """Intermediate version: comprehension for returns, accumulate for prices."""
    if n <= 0:
        raise ValueError("n must be positive")

    random.seed(seed)
    returns = [random.gauss(0, DAILY_VOLATILITY) for _ in range(n - 1)]
    return list(accumulate(returns, lambda price, r: price * (1 + r), initial=START_PRICE))


def fake_price_stream(seed: int = 0) -> Iterator[float]:
    """Advanced version: a generator that yields one price at a time, forever."""
    rng = random.Random(seed)
    price = START_PRICE
    while True:
        yield price
        price *= 1 + rng.gauss(0, DAILY_VOLATILITY)
