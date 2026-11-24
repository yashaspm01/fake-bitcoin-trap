"""Day 4: search pure noise, then deflate the winner. Same test on real Bitcoin."""

import numpy as np
import pandas as pd

from trap.paths import DATA_DIR, RESULTS_DIR
from trap.prices import make_fake_prices
from trap.search import search_grid
from trap.stats import deflated_sharpe

FASTS = list(range(2, 52, 2))
SLOWS = list(range(20, 500, 10))
N_SERIES = 20
N_DAYS = 1500
CONFIDENT = 0.95


def analyse(prices: np.ndarray) -> dict:
    """Search the whole grid on one price series and judge its best strategy."""
    table, returns = search_grid(prices, FASTS, SLOWS)
    best = table.iloc[0]
    result = deflated_sharpe(returns[best["name"]], table["sr"].to_numpy())
    return {
        "best_strategy": best["name"],
        "n_trials": len(table),
        "best_sharpe": best["sharpe"],
        "naive_confidence": result.psr_vs_zero,
        "luck_bar_sharpe": result.sr0 * np.sqrt(365),
        "dsr": result.dsr,
    }


def main() -> None:
    rows = []
    for seed in range(N_SERIES):
        prices = np.array(make_fake_prices(N_DAYS, seed=seed))
        rows.append({"series": f"fake_{seed}", **analyse(prices)})
    fake = pd.DataFrame(rows)

    fmt = {"float_format": lambda x: f"{x:.3f}", "index": False}
    print(f"=== {N_SERIES} pure-noise price series, {fake['n_trials'].iloc[0]} strategies each ===")
    print(fake.to_string(**fmt))

    naive_hits = int((fake["naive_confidence"] > CONFIDENT).sum())
    deflated_hits = int((fake["dsr"] > CONFIDENT).sum())
    print(f"\nAverage best Sharpe found in noise: {fake['best_sharpe'].mean():.2f}")
    print(f"Looks real (>{CONFIDENT:.0%}) if you ignore the search: {naive_hits} of {N_SERIES}")
    print(f"Looks real (>{CONFIDENT:.0%}) after deflating:          {deflated_hits} of {N_SERIES}")

    RESULTS_DIR.mkdir(exist_ok=True)
    fake.to_csv(RESULTS_DIR / "day4_fake_noise.csv", index=False)

    real_path = DATA_DIR / "btcusdt_1d.csv"
    if real_path.exists():
        closes = pd.read_csv(real_path, index_col=0)["close"].to_numpy()
        real = pd.DataFrame([{"series": "real_btcusdt", **analyse(closes)}])
        print("\n=== REAL BTCUSDT, same search and same test ===")
        print(real.to_string(**fmt))
        real.to_csv(RESULTS_DIR / "day4_real.csv", index=False)
    else:
        print("\n(Run `python -m trap.explore` first to download the real data.)")


if __name__ == "__main__":
    main()
