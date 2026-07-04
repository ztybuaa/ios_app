import sys
from pathlib import Path


def ensure_project_venv():
    project_root = Path(__file__).resolve().parents[1]
    expected_prefix = project_root / ".venv"
    actual_prefix = Path(sys.prefix).resolve()

    if actual_prefix != expected_prefix.resolve():
        raise SystemExit(
            "This project must be run with its local virtual environment.\n"
            f"Expected: {expected_prefix / 'Scripts' / 'python.exe'}\n"
            f"Actual:   {Path(sys.executable).resolve()}\n"
            "Do not run project scripts from Anaconda base or any global Python."
        )
