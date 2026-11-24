# The Fake Bitcoin Trap

[![CI](https://github.com/yashaspm01/fake-bitcoin-trap/actions/workflows/ci.yml/badge.svg)](https://github.com/yashaspm01/fake-bitcoin-trap/actions/workflows/ci.yml)

Search enough trading strategies and one will look brilliant, even on prices made of
pure random noise. This project demonstrates that, then measures it with the
**deflated Sharpe ratio** (Bailey & López de Prado, 2014).

**Live report (refreshed weekly):** https://yashaspm01.github.io/fake-bitcoin-trap/

*Educational research, not financial advice.*

## What it does

1. Generates fake "Bitcoin" prices as a random walk (no edge by construction).
2. Backtests about 1,200 moving-average crossover strategies on each fake series,
   with a one-day signal shift (no lookahead) and 0.1% fees.
3. Takes the best strategy of each search and tests it two ways: naively, and after
   deflating for the number of strategies tried.
4. Repeats the same search on real BTCUSDT daily closes from Binance's public API,
   and runs a holdout check (choose on the first half, judge on the second).

## Findings (run of 24 Nov 2026; the real-data rows move as the window rolls)

| Experiment | Result |
|---|---|
| Best Sharpe found in noise (20 series) | 0.55 on average, 1.01 at most |
| "Looks real" (>95% confidence) ignoring the search | 5 of 20 |
| "Looks real" after deflating | **0 of 20** |
| Real BTCUSDT, best strategy `ma_cross_2_50` | Sharpe 1.30, naive confidence 0.997, **deflated 0.938** |
| Holdout: chosen on first half, judged on second | Sharpe 1.69 in-sample, **0.84 out of sample** vs buy-and-hold 0.11 |

Reading: the naive test is fooled by luck about a quarter of the time. After deflating,
no noise strategy survived. On real Bitcoin the best strategy is suggestive, not proven
(0.938 is just under the 95% bar), and its out-of-sample Sharpe roughly halved while still
beating buy-and-hold.

## Limitations

- The deflated Sharpe assumes independent trials. Neighbouring windows are strongly
  correlated, so the effective number of trials is smaller and the test is somewhat harsh.
- One asset, one period (mostly an uptrend), one train/test split. The out-of-sample
  Sharpe has a wide error band (roughly ±0.8 over two years).
- 20 noise series is a small experiment. Fees are a flat 0.1%, with no slippage.
- The best strategy sits at the edge of the grid (`fast=2` is the smallest value tested).

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

faketrap download           # real BTCUSDT daily closes -> data/
faketrap noise --series 20  # search pure noise and deflate the winners
faketrap real               # same search and test on real prices
faketrap holdout            # choose on early data, judge on later data
faketrap report --out site  # static HTML report
```

Docker:

```bash
docker build -t faketrap .
docker run --rm -v trap-data:/data faketrap noise --series 5
```

## Development

```bash
pytest -q && ruff check . && ruff format --check . && mypy
```

CI runs those four checks on every push. A weekly workflow rebuilds the report and
deploys it to GitHub Pages, and tags like `v0.1.0` publish a Docker image to GHCR.

## Layout

| File | Purpose |
|---|---|
| `prices.py` | random-walk price generation (list, comprehension and generator versions) |
| `strategies.py` | readable pure-Python strategies (reference implementation) |
| `backtest.py`, `search.py` | vectorized NumPy/pandas backtester and grid search |
| `metrics.py`, `stats.py` | Sharpe ratio; probabilistic and deflated Sharpe |
| `holdout.py` | out-of-sample check |
| `data.py` | Binance public-API downloader |
| `experiment.py`, `report.py`, `cli.py` | orchestration, HTML report, command line |

## Design notes

- The pure-Python strategies are kept as a slow reference, and tests assert the fast
  vectorized code gives identical signals.
- Tests never touch the network: the Binance client is replaced with a fake.
- Fixed seeds make every noise result reproducible on any machine.

## License

MIT
