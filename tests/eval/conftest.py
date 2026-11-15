import sys
from pathlib import Path

# Make eval.py and the modules it imports importable (they live in the project root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
