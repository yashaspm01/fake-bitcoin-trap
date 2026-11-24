"""Command-line interface. Try `trap --help`."""

import argparse
import logging
from collections.abc import Callable, Sequence

from trap import experiment
from trap.data import fetch_daily_closes
from trap.holdout import holdout_check
from trap.logging_setup import setup_logging
from trap.paths import RESULTS_DIR

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trap", description="The Fake Bitcoin Trap experiments.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download real BTCUSDT daily closes")
    download.add_argument("--days", type=int, default=1500)

    noise = sub.add_parser("noise", help="search pure noise and deflate the winners")
    noise.add_argument("--series", type=int, default=experiment.N_SERIES)
    noise.add_argument("--days", type=int, default=experiment.N_DAYS)

    sub.add_parser("real", help="run the same search and deflation on real prices")

    holdout = sub.add_parser("holdout", help="choose on early data, judge on later data")
    holdout.add_argument("--train-fraction", type=float, default=0.5)
    return parser


def cmd_download(args: argparse.Namespace) -> int:
    closes = fetch_daily_closes("BTCUSDT", args.days)
    experiment.REAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    closes.to_csv(experiment.REAL_PATH)
    log.info(
        "saved %d closes (%s to %s)", len(closes), closes.index[0].date(), closes.index[-1].date()
    )
    return 0


def cmd_noise(args: argparse.Namespace) -> int:
    fake = experiment.run_noise(args.series, args.days)
    print(experiment.format_table(fake))
    print()
    print(experiment.noise_summary(fake))
    RESULTS_DIR.mkdir(exist_ok=True)
    fake.to_csv(RESULTS_DIR / "day4_fake_noise.csv", index=False)
    return 0


def cmd_real(args: argparse.Namespace) -> int:
    print(experiment.format_table(experiment.run_real()))
    return 0


def cmd_holdout(args: argparse.Namespace) -> int:
    result = holdout_check(
        experiment.load_real_closes(), experiment.FASTS, experiment.SLOWS, args.train_fraction
    )
    for key, value in result.items():
        shown = f"{value:.3f}" if isinstance(value, float) else value
        print(f"{key:<26} {shown}")
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "download": cmd_download,
    "noise": cmd_noise,
    "real": cmd_real,
    "holdout": cmd_holdout,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return COMMANDS[args.command](args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log.error("%s", exc, exc_info=args.verbose)
        return 1
