"""Build a small static HTML report (served by GitHub Pages)."""

import html
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trap import experiment
from trap.holdout import holdout_check

STYLE = """
body {
  font-family: system-ui, sans-serif;
  max-width: 60rem;
  margin: 2rem auto;
  padding: 0 1rem;
  line-height: 1.5;
}
table { border-collapse: collapse; font-size: 0.9rem; }
th, td { padding: 0.25rem 0.6rem; border-bottom: 1px solid #ddd; text-align: right; }
th:first-child, td:first-child { text-align: left; }
pre { background: #f5f5f5; padding: 1rem; overflow-x: auto; }
"""


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")


def build_report(out_dir: Path, n_series: int = experiment.N_SERIES) -> Path:
    """Write out_dir/index.html and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fake = experiment.run_noise(n_series)
    sections = [
        f"<h2>Pure-noise price series ({n_series} runs)</h2>",
        _table(fake),
        f"<pre>{html.escape(experiment.noise_summary(fake))}</pre>",
    ]

    if experiment.REAL_PATH.exists():
        closes = experiment.load_real_closes()
        holdout = holdout_check(closes, experiment.FASTS, experiment.SLOWS)
        sections += [
            "<h2>Real BTCUSDT</h2>",
            _table(experiment.run_real()),
            "<h2>Holdout: choose on early data, judge on later data</h2>",
            _table(pd.DataFrame([holdout])),
        ]
    else:
        sections.append("<p>Real price data is not available in this run.</p>")

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>The Fake Bitcoin Trap</title>"
        f"<style>{STYLE}</style></head><body>"
        "<h1>The Fake Bitcoin Trap</h1>"
        f"<p>Generated {stamp}. Educational research, not financial advice.</p>"
        + "".join(sections)
        + "</body></html>"
    )
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
