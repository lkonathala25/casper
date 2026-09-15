"""DSL warcraft entrypoint (grid_size in the config, default 12x12).

Mirrors deepsoftlog/experiments/mnist_addition/addition.py with one
deliberate divergence: the loss combines per-node CE with the prover NLL,
mirroring the scallop_warcraft baseline (loss = ce + lambda_path * path_nll).
The CE term is computed directly in PyTorch via a Trainer subclass
(`WarcraftTrainer`) rather than via a per-node DSL predicate; this is
Option 2 of the two CE-supervision routes discussed earlier. Without CE,
the diagnose run showed the path-loss alone can converge to a globally
consistent length-22 path that is not the ground-truth path.

Pipeline:
  - configure_dsl_warcraft injects var_ranges/edge_proj/edges/semiring into
    WARCRAFT_CONFIG and registers ResNetNode + WarcraftPathFunctor on
    SPECIAL_MODELS, BEFORE load_program is called.
  - get_train_dataloader returns Query+node_label tuples; get_val/test
    return plain Query (Trainer.eval() requires that shape).
  - WarcraftTrainer.train_epoch: per batch
        ce       = node_categorical_loss_vectorised(resnet.backbone(imgs), node_labels)
        path_nll = -mean over batch of log P(valid(Img)) via the prover
        loss     = ce + lambda_path * path_nll
        loss.backward(); step_optimizer()
  - compute_extra_metrics reports Exact / Hamming / Consistent (matching
    Scallop) plus endpoint_agreement_gt and mean -log P(valid).
"""

import argparse
import csv
from contextlib import contextmanager
import sys
from pathlib import Path
from time import time

# Allow `python deepsoftlog/experiments/warcraft/warcraft.py ...` from repo root
# to find the `deepsoftlog` package without needing -m.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepsoftlog.algebraic_prover.terms.expression import Expr
from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.embeddings import nn_models as dsl_nn
from deepsoftlog.embeddings.initialize_vector import SPECIAL_MODELS
from deepsoftlog.logic.soft_term import TensorTerm
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from deepsoftlog.training.trainer import Trainer

from deepsoftlog.experiments.warcraft import dataset as _wds
from deepsoftlog.experiments.warcraft.dataset import (
    WarcraftQueryDataset,
    get_node_setup,
)

GRID = _wds.GRID


def configure_grid(cfg):
    """Set the grid size from cfg['grid_size'] (default 12) here and in dataset.py."""
    global GRID
    GRID = int(cfg.get("grid_size", 12))
    _wds.set_grid(GRID)

# tasks/warcraft is added to sys.path by dataset.py; safe to import.
from model import (  # noqa: E402
    compute_metrics as compute_edge_metrics,
    node_categorical_loss_vectorised,
)


_EXP_ROOT = Path(__file__).parent


# =============================================================================
# Static setup: build edge_proj, edge_endpoints, and per-endpoint projections
# =============================================================================

def _build_edge_projections(node_info, node_order, edges_np, num_edges=None):
    """Build:
        edge_proj         (num_edges, total_dim) -- 0.5 * (proj @ flat) gives
                          the average of the two endpoints' opinions on edge.
        edge_endpoints    (num_edges, 2)          -- node ids (u, v) per edge.
        proj_u / proj_v   (num_edges, total_dim) -- per-endpoint marginals;
                          (proj_u @ flat)[e] = p_u(e) = node u's marginal
                          probability of selecting any pattern that uses e.
                          Used by endpoint_agreement_gt.
    """
    total_dim = sum(info["n_cats"] for info in node_info.values())
    num_edges = int(edges_np.shape[0]) if num_edges is None else num_edges
    edge_proj = torch.zeros(num_edges, total_dim)
    proj_u = torch.zeros(num_edges, total_dim)
    proj_v = torch.zeros(num_edges, total_dim)

    edge_endpoints = torch.zeros(num_edges, 2, dtype=torch.long)
    for e, (r1, c1, r2, c2) in enumerate(edges_np):
        edge_endpoints[e, 0] = int(r1) * GRID + int(c1)
        edge_endpoints[e, 1] = int(r2) * GRID + int(c2)

    offset = 0
    node_offsets = {}
    for node in node_order:
        node_offsets[node] = offset
        offset += node_info[node]["n_cats"]

    for node in node_order:
        info = node_info[node]
        i, j = node
        node_id = i * GRID + j
        off = node_offsets[node]
        for cat_idx, pattern in enumerate(info["edge_patterns"]):
            for edge_idx, value in pattern.items():
                if value != 1:
                    continue
                edge_proj[edge_idx, off + cat_idx] = 1.0
                u_id = int(edge_endpoints[edge_idx, 0].item())
                v_id = int(edge_endpoints[edge_idx, 1].item())
                if node_id == u_id:
                    proj_u[edge_idx, off + cat_idx] = 1.0
                elif node_id == v_id:
                    proj_v[edge_idx, off + cat_idx] = 1.0
                else:
                    raise AssertionError(
                        f"node {node} contributes to edge {edge_idx} but is neither endpoint")

    return edge_proj, edge_endpoints, proj_u, proj_v, total_dim


def _steps_domain_mask(cfg):
    """(L+1, n) bool mask of the compiled steps domains D_i for this grid."""
    from steps_walk import load_compiled_encoding
    art = cfg.get("steps_artifact") or (
        Path(__file__).resolve().parents[5] / "constraints" / "warcraft"
        / "steps_artifacts" / f"steps_{GRID}_trimmed_compiled.json")
    enc = load_compiled_encoding(art)
    if enc.g != GRID:
        raise ValueError(f"{art} is for a {enc.g}x{enc.g} grid, not {GRID}x{GRID}")
    mask = torch.zeros(enc.horizon + 1, GRID * GRID, dtype=torch.bool)
    for i, dom in enumerate(enc.domains):
        mask[i, list(dom)] = True
    return mask


# Per-endpoint projections used by endpoint-agreement metric (filled by configure).
_PROJ = {"u": None, "v": None}


def configure_dsl_warcraft(cfg, node_info, var_ranges, node_order, edges_np):
    edge_proj, edge_endpoints, proj_u, proj_v, total_dim = _build_edge_projections(
        node_info, node_order, edges_np
    )
    dsl_nn.set_warcraft_config(
        var_ranges=list(var_ranges),
        total_dim=int(total_dim),
        edge_proj=edge_proj,
        edge_endpoints=edge_endpoints,
        num_nodes=GRID * GRID,
        num_edges=int(edges_np.shape[0]),
        source=0,
        sink=GRID * GRID - 1,
        num_steps=int(cfg.get("num_path_steps", 2 * (GRID - 1))),
        semiring=str(cfg.get("semiring", "max")),
        steps_domains=(_steps_domain_mask(cfg)
                       if str(cfg.get("constraint", "steps")) == "steps" else None),
    )
    SPECIAL_MODELS[("resnet_node", 1)] = dsl_nn.ResNetNode
    SPECIAL_MODELS[("warcraft_path", 1)] = dsl_nn.WarcraftPathFunctor

    _PROJ["u"] = proj_u
    _PROJ["v"] = proj_v


# =============================================================================
# Dataloaders (mirror addition.get_*_dataloader)
# =============================================================================

def _data_dir(cfg):
    return Path(cfg.get("data_dir") or _wds.DATA_DIR_DEFAULT)


def get_train_dataloader(cfg):
    train_dataset = WarcraftQueryDataset("train", data_dir=_data_dir(cfg), with_labels=True)
    train_dataset = train_dataset.random_subset(cfg.get("data_subset"))
    return DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, seed=cfg["seed"])


def get_val_dataloader(cfg):
    val_dataset = WarcraftQueryDataset("val", data_dir=_data_dir(cfg))
    return DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)


def get_test_dataloader(cfg):
    test_dataset = WarcraftQueryDataset("test", data_dir=_data_dir(cfg))
    return DataLoader(test_dataset, batch_size=1, shuffle=False)


# =============================================================================
# Metrics: Exact / Hamming / Consistent + endpoint_agreement_gt
# =============================================================================

@torch.no_grad()
def compute_extra_metrics(program, cfg, split="test"):
    """Walk the split through the ResNet, compute:
        - exact / hamming / consistent (same as Scallop baseline)
        - endpoint_agreement_gt (mean of min(p_u, p_v) over GT-path edges)
        - mean -log P(valid) (one prover call per image)
    """
    store = program.get_store()
    store.eval()
    device = cfg["device"]
    data_dir = _data_dir(cfg)

    node_info, var_ranges, node_order, total_dim = get_node_setup(GRID)
    ds = WarcraftQueryDataset(split, data_dir=data_dir, node_info=node_info, node_order=node_order)
    edges_np = np.load(data_dir / "edges.npy")
    resnet = store.functor_embeddings["('resnet_node', 1)"]
    proj_u = _PROJ["u"].to(device)
    proj_v = _PROJ["v"].to(device)

    all_probs, all_edge_labels, agreements = [], [], []
    nlp_sum = 0.0
    n_images = 0
    for i in range(len(ds)):
        img, _node_label, edge_label = ds.raw_dataset[i]
        img = img.to(device)
        edge_label = edge_label.to(device)

        logits = resnet.backbone(img.unsqueeze(0)).squeeze(0)  # (total_dim,)
        flat = torch.empty_like(logits)
        offset = 0
        for vr in var_ranges:
            flat[offset:offset + vr] = F.softmax(logits[offset:offset + vr], dim=0)
            offset += vr
        all_probs.append(flat.cpu().unsqueeze(0))
        all_edge_labels.append(edge_label.cpu().unsqueeze(0))

        # Endpoint agreement on GT path: edges where edge_label > 0.5.
        p_u = (proj_u @ flat).clamp(0, 1)
        p_v = (proj_v @ flat).clamp(0, 1)
        gt_mask = edge_label > 0.5
        if gt_mask.any():
            agreements.append(float(torch.minimum(p_u[gt_mask], p_v[gt_mask]).mean()))

        # -log P(valid): prover call.
        store.clear_cache()
        log_p, _, _ = program(Expr("valid", TensorTerm(img)))
        if isinstance(log_p, torch.Tensor):
            nlp_sum += float(-log_p)
        n_images += 1

    probs = torch.cat(all_probs, dim=0)
    edge_labels = torch.cat(all_edge_labels, dim=0)
    edge_metrics = compute_edge_metrics(probs, edge_labels, edges_np, node_info,
                                        node_order, int(edges_np.shape[0]), GRID)
    metrics = {
        "exact": edge_metrics["exact"],
        "hamming": edge_metrics["hamming"],
        "consistent": edge_metrics["consistent"],
        "endpoint_agreement_gt": float(np.mean(agreements)) if agreements else float("nan"),
        "mean_neg_log_p_valid": nlp_sum / max(n_images, 1),
    }
    store.train()
    return metrics


def compute_raw_task_accuracy(program, cfg, trainer=None):
    """raw_eval_fn signature: returns (task_acc, digit_acc) in [0, 1].

    Logs the full metric dict + per-epoch peak GPU memory to stdout each
    epoch. When `trainer` is provided, also appends a row to its CSV via
    trainer.log_epoch (with epoch_time and peak GPU memory).
    """
    metrics = compute_extra_metrics(program, cfg, split="test")
    peak_mb = trainer.log_epoch(metrics) if trainer is not None else float("nan")
    print(
        f"  metrics(test): exact={metrics['exact']:5.2f}%  "
        f"hamming={metrics['hamming']:5.2f}%  "
        f"consistent={metrics['consistent']:5.2f}%  "
        f"endpoint_agreement_gt={metrics['endpoint_agreement_gt']:.4f}  "
        f"mean_neg_log_p_valid={metrics['mean_neg_log_p_valid']:.4f}  "
        f"peak_mem={peak_mb:.0f}MB"
    )
    return metrics["exact"] / 100.0, metrics["hamming"] / 100.0


# =============================================================================
# Trainer subclass: combine per-node CE with prover path NLL
# =============================================================================

@contextmanager
def frozen_bn_stats(module):
    """Keep BatchNorm running statistics unchanged inside the block (momentum 0)."""
    bns = [m for m in module.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, momentum in zip(bns, saved):
            m.momentum = momentum


class WarcraftTrainer(Trainer):
    """Override train_epoch to combine per-node CE with the prover path NLL.

    Loss per batch:
        ce       = mean over batch & nodes of cross_entropy on per-node
                   categorical, computed by running ResNet18 directly on
                   the image (the shared per-cell CE in tasks/warcraft/model.py).
        path_nll = -mean over batch of log P(valid(Img)) from the prover.
        loss     = ce + lambda_path * path_nll

    Backward+step uses the inherited Trainer.step_optimizer (which clips,
    measures grad norm, calls optimizer.step + zero_grad + clear_cache).

    Trainer.eval / raw_eval_fn / scheduler / save remain unchanged.

    Expects training items as (Query, node_label) tuples. Val/test still
    yield plain Query (compatible with the unmodified Trainer.eval path).
    """

    def __init__(self, *args, var_ranges, lambda_path=0.1, csv_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_ranges = list(var_ranges)
        self.lambda_path = float(lambda_path)
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._csv_epoch = 0

    def _ensure_csv(self):
        if self.csv_path is None or self._csv_writer is not None:
            return
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "epoch", "epoch_time_s", "gpu_mem_peak_mb",
            "exact", "hamming", "consistent",
            "endpoint_agreement_gt", "mean_neg_log_p_valid",
        ])
        self._csv_file.flush()

    def log_epoch(self, metrics):
        """Called by raw_eval_fn after each epoch.

        Returns peak GPU memory (MB) since the last call, then resets the
        peak counter so the next epoch's reading is independent.
        """
        self._ensure_csv()
        epoch_time = self.epoch_times[-1] if self.epoch_times else 0.0
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_mb = float("nan")
        if self._csv_writer is not None:
            self._csv_writer.writerow([
                self._csv_epoch, epoch_time, peak_mb,
                metrics["exact"], metrics["hamming"], metrics["consistent"],
                metrics["endpoint_agreement_gt"], metrics["mean_neg_log_p_valid"],
            ])
            self._csv_file.flush()
        self._csv_epoch += 1
        return peak_mb

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def save(self, config):
        """Override Trainer.save to NOT rmtree the results folder.

        The base Trainer wipes results/{name}/ at the start of every save,
        which deletes the per_epoch.csv we wrote earlier in the same epoch.
        We just overwrite config.yaml + store.pt in place.
        """
        save_folder = Path(f"results/{config['name']}")
        save_folder.mkdir(parents=True, exist_ok=True)
        config.save(save_folder / "config.yaml")
        torch.save(self.get_store().state_dict(), save_folder / "store.pt")

    def _split_batch(self, batch, device):
        queries = [item[0] for item in batch]
        node_labels = torch.stack([item[1] for item in batch]).to(device)
        imgs = []
        for q in queries:
            term = q.query.arguments[0]
            if not isinstance(term, TensorTerm):
                raise TypeError(f"expected TensorTerm in valid(Img), got {type(term)}")
            imgs.append(term.tensor.to(device))
        imgs = torch.stack(imgs)
        return queries, imgs, node_labels

    def train_epoch(self, verbose: bool):
        device = next(self.program.parameters()).device
        store = self.program.get_store()
        resnet = store.functor_embeddings["('resnet_node', 1)"]

        for batch in tqdm(self.train_dataset, leave=False, smoothing=0, disable=not verbose):
            t0 = time()
            queries, imgs, node_labels = self._split_batch(batch, device)

            # CE branch: direct ResNet forward on the batched images.
            logits = resnet.backbone(imgs)                                 # (B, total_dim)
            ce = node_categorical_loss_vectorised(logits, node_labels, self.var_ranges)

            # Path-NLL branch: one prover call per query (each runs its own
            # ResNet forward via the resnet_node functor; gradients
            # accumulate on the shared backbone).
            # Each prover query runs ResNetNode on a single image in train mode,
            # which would update batch-norm running statistics from single-image
            # statistics once per query -- 32 updates per step against one from the
            # batched CE forward -- and evaluation normalises with those running
            # statistics. Freeze the updates here so they come from batched forwards
            # only; this branch's normalisation and gradients are unchanged.
            log_ps = []
            ps_total = 0; nbp_total = 0
            with frozen_bn_stats(resnet):
                for q in queries:
                    res, ps, nbp = self.program(q.query, **self.search_args)
                    if not isinstance(res, torch.Tensor):
                        res = torch.tensor(float(res), device=device)
                    elif res.device != device:
                        res = res.to(device)
                    log_ps.append(res)
                    ps_total += ps; nbp_total += nbp
            path_nll = -torch.stack(log_ps).mean()

            loss = ce + self.lambda_path * path_nll
            loss.backward()
            grad_norm = self.step_optimizer()  # also calls store.clear_cache()

            if verbose:
                self.logger.log({
                    "grad_norm": grad_norm,
                    "loss": float(loss),
                    "ce": float(ce),
                    "path_nll": float(path_nll),
                    "step_time": time() - t0,
                    "proof_steps": ps_total / max(len(queries), 1),
                    "nb_proofs": nbp_total / max(len(queries), 1),
                })

        if verbose:
            self.logger.print()


# =============================================================================
# Train / eval entrypoints
# =============================================================================

def train(config_name, overrides=None):
    """Run the full warcraft 12x12 training.

    `overrides` is an optional dict whose keys (nb_epochs, seed, name,
    lambda_path, ...) replace the corresponding values in the config.yaml
    after it is loaded. Used by the CLI in __main__.
    """
    cfg = load_config(config_name)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    configure_grid(cfg)
    node_info, var_ranges, node_order, total_dim = get_node_setup(GRID)
    edges_np = np.load(_data_dir(cfg) / "edges.npy")
    configure_dsl_warcraft(cfg, node_info, var_ranges, node_order, edges_np)
    print(f"DSL warcraft: name={cfg['name']}  seed={cfg['seed']}  "
          f"nb_epochs={cfg['nb_epochs']}  batch_size={cfg['batch_size']}  "
          f"grid={GRID}x{GRID}  lambda_path={cfg.get('lambda_path', 0.1)}  semiring={cfg.get('semiring')}  "
          f"constraint={cfg.get('constraint', 'steps')}  "
          + (f"num_path_steps={cfg.get('num_path_steps', 2*(GRID-1))}  "
             if str(cfg.get('constraint', 'steps')) == 'reach' else "")
          + f"total_dim={total_dim}")

    val_dataloader = get_val_dataloader(cfg)
    test_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, val_dataloader)
    optimizer = get_optimizer(program.get_store(), cfg)
    logger = WandbLogger(cfg)

    csv_path = Path("results") / cfg["name"] / "per_epoch.csv"
    trainer_box = []

    def raw_eval_fn():
        return compute_raw_task_accuracy(
            program, cfg, trainer=trainer_box[0] if trainer_box else None
        )

    trainer = WarcraftTrainer(
        program,
        get_train_dataloader,
        val_dataloader,
        test_dataloader,
        nll_loss,
        optimizer,
        logger=logger,
        supervised_dataloader=None,
        raw_eval_fn=raw_eval_fn,
        var_ranges=var_ranges,
        lambda_path=float(cfg.get("lambda_path", 0.1)),
        csv_path=str(csv_path),
    )
    trainer_box.append(trainer)
    try:
        trainer.train(cfg, nb_workers=cfg["nb_workers"])
        avg_epoch_time = sum(trainer.epoch_times) / len(trainer.epoch_times) if trainer.epoch_times else 0.0

        final_metrics = compute_extra_metrics(program, cfg, split="test")
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory_mb = -1

        print(
            f"RESULT: name={cfg['name']} seed={cfg['seed']} "
            f"exact={final_metrics['exact']:.4f} "
            f"hamming={final_metrics['hamming']:.4f} "
            f"consistent={final_metrics['consistent']:.4f} "
            f"endpoint_agreement_gt={final_metrics['endpoint_agreement_gt']:.4f} "
            f"mean_neg_log_p_valid={final_metrics['mean_neg_log_p_valid']:.4f} "
            f"peak_memory_mb={peak_memory_mb:.1f} "
            f"epoch_time_s={avg_epoch_time:.2f} "
            f"per_epoch_csv={csv_path}"
        )
        return final_metrics
    finally:
        trainer.close()


def eval(folder: str):
    cfg = load_config(f"results/{folder}/config.yaml")
    print(f"EVALING {cfg['name']}")
    configure_grid(cfg)

    node_info, var_ranges, node_order, _ = get_node_setup(GRID)
    edges_np = np.load(_data_dir(cfg) / "edges.npy")
    configure_dsl_warcraft(cfg, node_info, var_ranges, node_order, edges_np)

    test_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, test_dataloader)
    state_dict = torch.load(f"results/{folder}/store.pt")
    program.store.load_state_dict(state_dict, strict=False)

    metrics = compute_extra_metrics(program, cfg, split="test")
    print(metrics)
    return metrics


def _parse_args():
    p = argparse.ArgumentParser(description="DSL warcraft training.")
    p.add_argument("--grid-size", type=int, default=None,
                   help="Grid size (overrides grid_size in the config).")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Data directory (overrides data_dir in the config).")
    p.add_argument("--config", type=str,
                   default="deepsoftlog/experiments/warcraft/config.yaml")
    p.add_argument("--nb-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--lambda-path", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="Override functor_learning_rate")
    p.add_argument("--data-subset", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = {
        "nb_epochs": args.nb_epochs,
        "seed": args.seed,
        "name": args.name,
        "lambda_path": args.lambda_path,
        "batch_size": args.batch_size,
        "functor_learning_rate": args.lr,
        "data_subset": args.data_subset,
        "grid_size": args.grid_size,
        "data_dir": args.data_dir,
    }
    train(args.config, overrides=overrides)
