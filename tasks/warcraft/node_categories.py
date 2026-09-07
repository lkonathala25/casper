import sys as _sys
from pathlib import Path as _Path

# Runnable entry point: put the shared compiler and this task's own directory on
# the path. Sibling task directories are deliberately not added -- they share
# module names (data, model, train, minedit).
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "casper", _Path(__file__).resolve().parent, _ROOT / "examples"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import numpy as np


# Unordered direction pairs, ordered so that the four monotone pairs come first.
# This makes the 'full' alphabet a strict prefix extension of 'monotone' at every
# node, so a monotone-trained head embeds into the full head by zero-padding.
_PAIR_ORDER = [('L', 'R'), ('L', 'D'), ('U', 'R'), ('U', 'D'), ('L', 'U'), ('R', 'D')]


def build_node_categories(grid_size=12, alphabet='monotone'):
    """
    Build the categorical variable definition for each node in the grid.

    Args:
        grid_size: N for an NxN grid.
        alphabet:  'monotone' — one incoming (left/above) and one outgoing
                   (right/below) edge per routing category. Only down/right
                   traversal is representable; this is the original encoding.
                   'full' — every unordered pair of incident edges, so all
                   4-connected simple paths are representable.

    Returns:
        node_info: dict mapping (i,j) -> {
            'categories': list of category names,
            'n_cats': int,
            'incoming': dict label -> edge index (left='R', above='D'),
            'outgoing': dict label -> edge index (right='R', below='D'),
            'edge_dirs': dict 'L'/'U'/'R'/'D' -> edge index,
            'edge_patterns': list indexed by category -> {edge_idx: 0 or 1},
            'all_edges': list of incident edge indices,
            'is_source', 'is_sink': bool
        }
        var_ranges: list of category counts per node (in row-major order)
        node_order: list of (i,j) tuples in row-major order
        total_dim: sum of all category counts
    """
    if alphabet not in ('monotone', 'full'):
        raise ValueError(f"alphabet must be 'monotone' or 'full', got {alphabet!r}")

    n_w = grid_size - 1
    n_right = grid_size * n_w  # 132 horizontal edges

    def right_idx(i, j):
        return i * n_w + j

    def down_idx(i, j):
        return n_right + i * grid_size + j

    node_info = {}
    node_order = []
    var_ranges = []

    for i in range(grid_size):
        for j in range(grid_size):
            node = (i, j)
            node_order.append(node)

            # Incident edges by compass direction.
            edge_dirs = {}
            if j > 0:
                edge_dirs['L'] = right_idx(i, j - 1)
            if i > 0:
                edge_dirs['U'] = down_idx(i - 1, j)
            if j < grid_size - 1:
                edge_dirs['R'] = right_idx(i, j)
            if i < grid_size - 1:
                edge_dirs['D'] = down_idx(i, j)

            # Legacy view: 'R'/'D' keyed by traversal role, not compass direction.
            incoming = {}
            outgoing = {}
            if 'L' in edge_dirs:
                incoming['R'] = edge_dirs['L']
            if 'U' in edge_dirs:
                incoming['D'] = edge_dirs['U']
            if 'R' in edge_dirs:
                outgoing['R'] = edge_dirs['R']
            if 'D' in edge_dirs:
                outgoing['D'] = edge_dirs['D']

            is_source = (i == 0 and j == 0)
            is_sink = (i == grid_size - 1 and j == grid_size - 1)

            all_edge_indices = list(incoming.values()) + list(outgoing.values())

            categories = []
            edge_patterns = []

            if is_source or is_sink:
                # Endpoints carry exactly one incident edge. Iterating the legacy
                # dicts preserves the original category indices; the names are
                # compass directions, so the sink reads 'L'/'U' not 'R'/'D'.
                endpoint_edges = (outgoing if is_source else incoming)
                compass = {v: k for k, v in edge_dirs.items()}
                for idx in endpoint_edges.values():
                    categories.append(compass[idx])
                    pattern = {e: 0 for e in all_edge_indices}
                    pattern[idx] = 1
                    edge_patterns.append(pattern)

            else:
                # Skip: all edges OFF
                categories.append('skip')
                edge_patterns.append({e: 0 for e in all_edge_indices})

                if alphabet == 'monotone':
                    pairs = [(a, b) for (a, b) in _PAIR_ORDER[:4]
                             if a in edge_dirs and b in edge_dirs]
                else:
                    pairs = [(a, b) for (a, b) in _PAIR_ORDER
                             if a in edge_dirs and b in edge_dirs]

                for a, b in pairs:
                    categories.append(f'{a}|{b}')
                    pattern = {e: 0 for e in all_edge_indices}
                    pattern[edge_dirs[a]] = 1
                    pattern[edge_dirs[b]] = 1
                    edge_patterns.append(pattern)

            info = {
                'categories': categories,
                'n_cats': len(categories),
                'incoming': incoming,
                'outgoing': outgoing,
                'edge_dirs': edge_dirs,
                'edge_patterns': edge_patterns,
                'all_edges': all_edge_indices,
                'is_source': is_source,
                'is_sink': is_sink,
                'alphabet': alphabet,
            }

            node_info[node] = info
            var_ranges.append(len(categories))

    total_dim = sum(var_ranges)

    return node_info, var_ranges, node_order, total_dim


class UnrepresentablePatternError(ValueError):
    """A ground-truth edge pattern has no category under the active alphabet."""


def edge_labels_to_node_labels(edge_labels, node_info, node_order, instance_ids=None):
    """
    Convert binary edge labels (batch, 264) to node categorical labels (batch, 144).

    For each node, find which category matches the observed edge pattern.

    Raises UnrepresentablePatternError if any node's observed edge pattern has no
    matching category. Silently relabelling such a node (the previous behaviour)
    corrupts the training target without warning.

    Args:
        edge_labels: (batch, 264) binary
        node_info: from build_node_categories
        node_order: list of (i,j)
        instance_ids: optional per-row identifiers used in the error message;
                      defaults to the row index within `edge_labels`.

    Returns:
        node_labels: (batch, 144) integer category indices
    """
    if isinstance(edge_labels, np.ndarray):
        edge_labels = edge_labels.astype(int)
        is_numpy = True
    else:
        edge_labels_np = edge_labels.cpu().numpy().astype(int)
        is_numpy = False
        edge_labels = edge_labels_np
    
    batch_size = edge_labels.shape[0]
    n_nodes = len(node_order)
    node_labels = np.zeros((batch_size, n_nodes), dtype=np.int64)
    
    for n_idx, node in enumerate(node_order):
        info = node_info[node]
        patterns = info['edge_patterns']
        all_edges = info['all_edges']
        
        for b in range(batch_size):
            # Get observed edge values for this node
            observed = {e: int(edge_labels[b, e]) for e in all_edges}
            
            # Find matching category
            matched = False
            for cat_idx, pattern in enumerate(patterns):
                if pattern == observed:
                    node_labels[b, n_idx] = cat_idx
                    matched = True
                    break

            if not matched:
                inst = b if instance_ids is None else instance_ids[b]
                on = sorted(e for e, v in observed.items() if v == 1)
                dirs = {v: k for k, v in info['edge_dirs'].items()}
                raise UnrepresentablePatternError(
                    f"instance {inst}: node {node} has edge pattern "
                    f"{{{', '.join(f'{dirs.get(e, e)}(e{e})' for e in on)}}} "
                    f"(on-edges {on}, all incident {sorted(all_edges)}), which has no "
                    f"category under alphabet {info['alphabet']!r}. "
                    f"Categories here: {info['categories']}. "
                    f"Rebuild with alphabet='full' to represent all 4-connected paths."
                )

    return node_labels


def node_preds_to_edge_preds(node_probs, node_info, node_order, num_edges=264):
    """
    Convert node categorical predictions back to binary edge predictions.
    
    Args:
        node_probs: (batch, total_dim) softmax probabilities
        node_info: from build_node_categories
        node_order: list of (i,j)
        num_edges: 264
    
    Returns:
        edge_preds: (batch, 264) binary predictions
    """
    import torch
    
    batch_size = node_probs.shape[0]
    device = node_probs.device
    edge_preds = torch.zeros(batch_size, num_edges, device=device)
    
    offset = 0
    for n_idx, node in enumerate(node_order):
        info = node_info[node]
        n_cats = info['n_cats']
        
        # Get argmax category for this node
        cat_probs = node_probs[:, offset:offset + n_cats]
        pred_cat = cat_probs.argmax(dim=1)  # (batch,)
        
        # For each sample, set edges according to the predicted pattern
        for b in range(batch_size):
            cat = pred_cat[b].item()
            pattern = info['edge_patterns'][cat]
            for edge_idx, value in pattern.items():
                if value == 1:
                    edge_preds[b, edge_idx] = 1.0
        
        offset += n_cats
    
    return edge_preds


if __name__ == "__main__":
    from collections import Counter

    for alphabet in ("monotone", "full"):
        node_info, var_ranges, node_order, total_dim = build_node_categories(
            12, alphabet=alphabet)
        print(f"\n=== 12x12, alphabet={alphabet!r} ===")
        print(f"  Nodes: {len(node_order)}   total_dim: {total_dim}")
        for n_cats, count in sorted(Counter(var_ranges).items()):
            print(f"    {n_cats} categories: {count} nodes")
        print(f"  Source (0,0):   {node_info[(0,0)]['categories']}")
        print(f"  Sink (11,11):   {node_info[(11,11)]['categories']}")
        print(f"  Top edge (0,5): {node_info[(0,5)]['categories']}")
        print(f"  Interior (5,5): {node_info[(5,5)]['categories']}")