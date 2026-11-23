from itertools import islice

import pytest

from trap.prices import fake_price_stream, make_fake_prices, make_fake_prices_v2


def test_length_and_start():
    prices = make_fake_prices(50, seed=1)
    assert len(prices) == 50
    assert prices[0] == 100.0


def test_same_seed_same_prices():
    assert make_fake_prices(20, seed=7) == make_fake_prices(20, seed=7)


def test_bad_n():
    with pytest.raises(ValueError):
        make_fake_prices(0)


def test_v2_matches_basic_version():
    a = make_fake_prices(30, seed=3)
    b = make_fake_prices_v2(30, seed=3)
    assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def test_stream_is_reproducible():
    first = list(islice(fake_price_stream(seed=5), 10))
    second = list(islice(fake_price_stream(seed=5), 10))
    assert first == second
    assert first[0] == 100.0
