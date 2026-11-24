from trap import data


def test_fetch_parses_and_drops_unfinished_day(monkeypatch):
    def fake_klines(params):
        day_ms = 86_400_000
        return [
            [1_700_000_000_000 + i * day_ms, "0", "0", "0", str(100 + i), "0"]
            for i in range(params["limit"])
        ]

    monkeypatch.setattr(data, "_get_klines", fake_klines)
    closes = data.fetch_daily_closes("BTCUSDT", days=5)

    assert len(closes) == 5
    assert closes.iloc[0] == 100.0
    assert closes.iloc[-1] == 104.0  # the 6th row (105) was dropped
