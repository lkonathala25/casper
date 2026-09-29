from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def resolved_params(raw: str) -> dict:
    """SPL's command-line dict with every relative path made absolute against this folder."""
    params = ast.literal_eval(raw)
    cfg = (HERE / params["default_json"]).resolve()
    params["default_json"] = str(cfg)
    loader = dict(json.loads(cfg.read_text())["loader_params"])
    loader.update(params.get("loader_params", {}))
    loader["data_dir"] = str((HERE / loader["data_dir"]).resolve())
    params["loader_params"] = loader
    params["model_dir"] = str((HERE / params["model_dir"]).resolve())
    return params


def main():
    """Run SPL's own main() with the StepsFFCircuit trainer registered.

    SPL's main.py and trainers.py are used unmodified. The run happens inside
    the seed's model_dir, so SPL's relative checkpoints/ folder is per seed.
    """
    params = resolved_params(sys.argv[1])
    Path(params["model_dir"]).mkdir(parents=True, exist_ok=True)
    os.chdir(params["model_dir"])
    sys.argv = [sys.argv[0], repr(params)]

    import main as spl_main
    from steps_trainer import StepsFFCircuitTrainer

    base = spl_main.trainer_loaders["warcraft_shortest_path"]

    def get_trainer(name):
        return StepsFFCircuitTrainer if name == "StepsFFCircuit" else base(name)

    spl_main.trainer_loaders["warcraft_shortest_path"] = get_trainer
    spl_main.main()


if __name__ == "__main__":
    main()
