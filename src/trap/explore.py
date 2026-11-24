"""Day 3: compare strategy grids on fake and real prices."""

import time

import numpy as np
import pandas as pd

from trap.backtest import ma_crossover_signals, strategy_returns
from trap.data import fetch_daily_closes
from trap.metrics import sharpe_ratio, total_return
from trap.paths import DATA_DIR, RESULTS_DIR
from trap.prices import make_fake_prices
from trap.strategies import moving_average


def benchmark(n: int = 20_000, window: int = 200) -> None:
    prices = make_fake_prices(n, seed=1)
    t0 = time.perf_counter()
    moving_average(prices, window)
    pure = time.perf_counter() - t0
    t0 = time.perf_counter()
    pd.Series(prices).rolling(window).mean()
    fast = max(time.perf_counter() - t0, 1e-9)
    print(f"Moving average on {n:,} prices")
    print(f"  pure Python: {pure:.4f}s | pandas: {fast:.4f}s | {pure / fast:.0f}x faster\n")


def run_grid(prices: np.ndarray, fasts: list[int], slows: list[int]) -> pd.DataFrame:
    rows = []
    for fast in fasts:
        for slow in slows:
            if fast >= slow:
                continue
            returns = strategy_returns(prices, ma_crossover_signals(prices, fast, slow))
            rows.append(
                {
                    "name": f"ma_cross_{fast}_{slow}",
                    "sharpe": sharpe_ratio(returns),
                    "total_return": total_return(returns),
                }
            )
    hold = strategy_returns(prices, np.ones(len(prices)))
    rows.append(
        {
            "name": "buy_and_hold",
            "sharpe": sharpe_ratio(hold),
            "total_return": total_return(hold),
        }
    )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def show(title: str, table: pd.DataFrame) -> None:
    print(f"=== {title} (top 5 of {len(table)} by Sharpe) ===")
    print(table.head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()


def main() -> None:
    fasts, slows = [5, 10, 20, 30], [50, 100, 150, 200]

    benchmark()

    fake = np.array(make_fake_prices(1500, seed=42))
    fake_table = run_grid(fake, fasts, slows)
    show("FAKE prices (pure noise)", fake_table)

    try:
        closes = fetch_daily_closes("BTCUSDT", 1500)
    except RuntimeError as exc:
        print(f"Skipping real data: {exc}")
        return

    DATA_DIR.mkdir(exist_ok=True)
    closes.to_csv(DATA_DIR / "btcusdt_1d.csv")
    first, last = closes.index[0].date(), closes.index[-1].date()
    print(f"Downloaded {len(closes)} real daily closes: {first} to {last}\n")

    real_table = run_grid(closes.to_numpy(), fasts, slows)
    show("REAL BTCUSDT prices", real_table)

    RESULTS_DIR.mkdir(exist_ok=True)
    fake_table.to_csv(RESULTS_DIR / "day3_fake.csv", index=False)
    real_table.to_csv(RESULTS_DIR / "day3_real.csv", index=False)


if __name__ == "__main__":
    main()
