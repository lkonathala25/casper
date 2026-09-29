from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def predict(run_dir: Path, data_dir: Path, grid_size: int) -> Path:
    """Decode the test split with a run's selected checkpoint; save canonical-order edge predictions."""
    sys.path.insert(0, str(HERE))
    import torch
    import torch.nn.functional as F
    from models import get_model
    from warcraft_shortest_path.data_utils import load_dataset
    from steps_trainer import StepsFFCircuitTrainer, e2i_permutation  # noqa: F401 (sets up paths)
    from compute_mpe import CircuitMPE
    from layered_circuit import LayeredCircuit
    from steps_sdd import DEFAULT_OUT, load_moves
    from steps_spl import StepsSplAdapter

    settings = json.loads((run_dir / "settings.json").read_text())
    K = settings["trainer_params"]["num_reps"]
    ckpts = sorted((run_dir / "checkpoints").glob("*.pt"))
    assert len(ckpts) == 1, f"expected one selected checkpoint in {run_dir}/checkpoints, got {ckpts}"
    state = torch.load(ckpts[0], map_location="cuda")

    _, test_iter, meta = load_dataset(str(data_dir), use_test_set=True, evaluate_with_extra=False,
                                      normalize=True, use_local_path=False)
    model = get_model("ResNet18", out_features=264 * K + K, in_channels=meta["num_channels"],
                      arch_params={}).cuda()
    model.load_state_dict(state["nn_state_dict"])
    model.eval()

    g = grid_size
    cm = CircuitMPE(str(DEFAULT_OUT / f"steps_{g}.vtree"), str(DEFAULT_OUT / f"steps_{g}.sdd"))
    cm.beta.num_reps = K
    adapter = StepsSplAdapter(load_moves(g))
    engine = LayeredCircuit(cm.beta, adapter.sm.num_vars)
    perm = torch.as_tensor(e2i_permutation(g), device="cuda")

    preds = []
    with torch.no_grad():
        for batch in test_iter.get_epoch_iterator(batch_size=16, number_of_epochs=1, shuffle=False,
                                                  device="cuda", preload=False):
            x = batch["images"]
            out = model(x).view(x.size(0), 264 + 1, K)
            out, mixing = out.split((264, 1), dim=1)
            mixing = mixing.squeeze(1).log_softmax(dim=1)
            lp = F.logsigmoid(out).clamp(max=-1e-7)
            canon = torch.zeros_like(lp)
            canon[:, perm] = lp
            neg, pos = adapter.lit_tensors(canon)
            edges, _ = adapter.bits_to_edges(engine.mpe_bits(neg, pos, mixing))
            preds.append(edges.cpu().numpy().astype(np.uint8))
    pred = np.concatenate(preds)
    out_path = run_dir / "test_pred_edges.npy"
    np.save(out_path, pred)
    print(f"saved {pred.shape} predictions from {ckpts[0].name} (epoch {state.get('epoch')})")
    return out_path


def score(run_dir: Path, data_dir: Path, grid_size: int) -> dict:
    """Score saved predictions with the repo's shared paper_metrics; write metrics.json."""
    sys.path.insert(0, str(ROOT / "tasks" / "warcraft"))
    from eval_warcraft_all import paper_metrics

    pred = np.load(run_dir / "test_pred_edges.npy")
    gt = np.load(data_dir / "test_edge_labels.npy")
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    metrics = paper_metrics(pred, gt, np.load(data_dir / "test_vertex_weights.npy"),
                            np.load(data_dir / "edges.npy"), grid_size,
                            optimal_costs=np.load(data_dir / "test_costs.npy"))
    metrics["run"] = {"method": "spl_steps", "decode": "spl_mpe_loop_erased",
                      "run_dir": str(run_dir), "grid_size": grid_size, "data_dir": str(data_dir)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"spl_steps {run_dir.name}: optimal {metrics['optimal']:.1f}%  exact {metrics['exact_match']:.1f}%  "
          f"consistent {metrics['consistent']:.1f}%")
    return metrics


def main():
    """Predict the test split with a run's selected checkpoint, then score it in a clean process."""
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "warcraft" / "12x12" / "preprocessed_4conn")
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--score", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    if args.score:
        score(run_dir, args.data_dir.resolve(), args.grid_size)
        return
    predict(run_dir, args.data_dir.resolve(), args.grid_size)
    # tasks/warcraft and SPL both define data/, utils and models, so the shared
    # metrics are imported in a separate interpreter.
    subprocess.run([sys.executable, __file__, str(run_dir), "--data-dir", str(args.data_dir.resolve()),
                    "--grid-size", str(args.grid_size), "--score"], check=True)


if __name__ == "__main__":
    main()
