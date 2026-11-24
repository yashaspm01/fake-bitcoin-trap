"""Project folders, independent of the current working directory."""

import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("TRAP_HOME", _DEFAULT_ROOT))
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
