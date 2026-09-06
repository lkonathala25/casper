import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# src/ and src_warcraft/ both define these module names. Importing a
# src_warcraft module leaves its versions in sys.modules, which then shadow
# src/ for every other test collected into the same process.
_COLLIDING = ("constraint_layer", "data", "model", "minedit", "train")


def load_warcraft(name):
    """Import a src_warcraft module without leaving it shadowing src/.

    src_warcraft is prepended only for the duration of the import, and the
    colliding module names it pulls in are evicted from sys.modules afterwards
    so a later `import constraint_layer` still resolves against src/.
    """
    path = str(ROOT / "src_warcraft")
    added = path not in sys.path
    sys.path.insert(0, path)
    before = set(sys.modules)
    try:
        return importlib.import_module(name)
    finally:
        if added:
            sys.path.remove(path)
        else:
            sys.path.pop(0)
        for mod in set(sys.modules) - before:
            if mod in _COLLIDING:
                del sys.modules[mod]
