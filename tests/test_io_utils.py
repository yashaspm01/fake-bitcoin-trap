from trap.io_utils import load_json, load_prices, save_json, save_prices
from trap.prices import make_fake_prices


def test_prices_roundtrip(tmp_path):
    prices = make_fake_prices(25, seed=2)
    path = tmp_path / "nested" / "prices.csv"
    save_prices(prices, path)
    assert load_prices(path) == prices


def test_json_roundtrip(tmp_path):
    data = {"ma_cross_5_50": {"days_in_market": 120, "trades": 9}}
    path = tmp_path / "out" / "summary.json"
    save_json(data, path)
    assert load_json(path) == data
