import sys
from pathlib import Path

# Make the `gen` package importable (it lives in the project root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
