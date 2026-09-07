import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# casper/ is the shared compiler and layer; tasks/mnist/ is the default task.
# tasks/warcraft/ is NOT added here -- it shares module names (data, model,
# train, minedit) with tasks/mnist/, so it is loaded through
# tests/_warcraft_import.load_warcraft(), which adds it only for the duration
# of one import and evicts the colliding names afterwards.
for sub in ("casper", "tasks/mnist", "tasks/sudoku", "examples"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
