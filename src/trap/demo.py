"""Run a small strategy grid on fake prices and save the summary."""

from trap.io_utils import load_prices, save_json, save_prices
from trap.paths import DATA_DIR, RESULTS_DIR
from trap.prices import make_fake_prices
from trap.strategies import count_trades, strategy_grid


def main() -> None:
    csv_path = DATA_DIR / "fake_prices.csv"
    save_prices(make_fake_prices(500, seed=42), csv_path)
    prices = load_prices(csv_path)

    summary: dict[str, dict[str, int]] = {}
    for strategy in strategy_grid([5, 10, 20], [50, 100]):
        sig = strategy.signals(prices)
        summary[strategy.name] = {
            "days_in_market": sum(sig),
            "trades": count_trades(sig),
        }

    save_json(summary, RESULTS_DIR / "day2_summary.json")
    for name, stats in summary.items():
        print(f"{name:<16} {stats}")


if __name__ == "__main__":
    main()
