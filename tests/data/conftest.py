import sys
from pathlib import Path

# The pipeline scripts import each other as top-level modules (they are run as
# `python scripts/data/<name>.py`), so the tests need the same import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))
