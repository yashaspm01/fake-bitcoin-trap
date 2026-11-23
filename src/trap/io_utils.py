"""Save and load prices (CSV) and results (JSON)."""

import csv
import json
from pathlib import Path
from typing import Any


def save_prices(prices: list[float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "price"])
        for day, price in enumerate(prices):
            writer.writerow([day, price])


def load_prices(path: str | Path) -> list[float]:
    with Path(path).open(newline="") as f:
        return [float(row["price"]) for row in csv.DictReader(f)]


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)

