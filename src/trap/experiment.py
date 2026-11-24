"""Search pure noise, then deflate the winner. Same test on real Bitcoin."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trap.paths import DATA_DIR, RESULTS_DIR
from trap.prices import make_fake_prices
from trap.search import search_grid
from trap.stats import deflated_sharpe
from trap.timing import timed

log = logging.getLogger(__name__)

FASTS = list(range(2, 52, 2))
SLOWS = list(range(20, 500, 10))
N_SERIES = 20
N_DAYS = 1500
CONFIDENT = 0.95
REAL_PATH = DATA_DIR / "btcusdt_1d.csv"


def analyse(prices: np.ndarray) -> dict[str, Any]:
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


@timed
def run_noise(n_series: int = N_SERIES, n_days: int = N_DAYS) -> pd.DataFrame:
    rows = []
    for seed in range(n_series):
        row = analyse(np.array(make_fake_prices(n_days, seed=seed)))
        log.debug(
            "series %d: best=%s sharpe=%.2f dsr=%.3f",
            seed,
            row["best_strategy"],
            row["best_sharpe"],
            row["dsr"],
        )
        rows.append({"series": f"fake_{seed}", **row})
    return pd.DataFrame(rows)


def load_real_closes(path: Path | None = None) -> np.ndarray:
    path = path or REAL_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `trap download` first.")
    return pd.read_csv(path, index_col=0)["close"].to_numpy()


@timed
def run_real(path: Path | None = None) -> pd.DataFrame:
    row = analyse(load_real_closes(path))
    return pd.DataFrame([{"series": "real_btcusdt", **row}])


def format_table(table: pd.DataFrame) -> str:
    return table.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def noise_summary(fake: pd.DataFrame) -> str:
    n = len(fake)
    naive = int((fake["naive_confidence"] > CONFIDENT).sum())
    deflated = int((fake["dsr"] > CONFIDENT).sum())
    return (
        f"Average best Sharpe found in noise: {fake['best_sharpe'].mean():.2f}\n"
        f"Looks real (>{CONFIDENT:.0%}) if you ignore the search: {naive} of {n}\n"
        f"Looks real (>{CONFIDENT:.0%}) after deflating: {deflated} of {n}"
    )


def main() -> None:
    fake = run_noise()
    n_trials = fake["n_trials"].iloc[0]
    print(f"=== {len(fake)} pure-noise series, {n_trials} strategies each ===")
    print(format_table(fake))
    print()
    print(noise_summary(fake))
    RESULTS_DIR.mkdir(exist_ok=True)
    fake.to_csv(RESULTS_DIR / "day4_fake_noise.csv", index=False)

    try:
        real = run_real()
    except FileNotFoundError:
        print("\n(Run `trap download` first to get the real data.)")
        return
    print("\n=== REAL BTCUSDT, same search and same test ===")
    print(format_table(real))
    real.to_csv(RESULTS_DIR / "day4_real.csv", index=False)


if __name__ == "__main__":
    main()
