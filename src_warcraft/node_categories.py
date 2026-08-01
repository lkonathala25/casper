import numpy as np


def build_node_categories(grid_size=12):
    """
    Build the categorical variable definition for each node in the grid.
    
    Returns:
        node_info: dict mapping (i,j) -> {
            'categories': list of category names,
            'n_cats': int,
            'incoming': list of edge indices,
            'outgoing': list of edge indices,
            'edge_pattern': dict mapping category_idx -> {edge_idx: 0 or 1}
        }
        var_ranges: list of category counts per node (in row-major order)
        node_order: list of (i,j) tuples in row-major order
        total_dim: sum of all category counts
    """
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
            
            # Determine available edges
            incoming = {}  # label -> edge_idx
            outgoing = {}
            
            if j > 0:
                incoming['R'] = right_idx(i, j - 1)
            if i > 0:
                incoming['D'] = down_idx(i - 1, j)
            if j < grid_size - 1:
                outgoing['R'] = right_idx(i, j)
            if i < grid_size - 1:
                outgoing['D'] = down_idx(i, j)
            
            is_source = (i == 0 and j == 0)
            is_sink = (i == grid_size - 1 and j == grid_size - 1)
            
            all_edge_indices = list(incoming.values()) + list(outgoing.values())
            
            categories = []
            edge_patterns = []
            
            if is_source:
                # Must emit flow: one outgoing edge ON, no incoming
                for out_label, out_idx in outgoing.items():
                    categories.append(out_label)
                    pattern = {e: 0 for e in all_edge_indices}
                    pattern[out_idx] = 1
                    edge_patterns.append(pattern)
                    
            elif is_sink:
                # Must absorb flow: one incoming edge ON, no outgoing
                for in_label, in_idx in incoming.items():
                    categories.append(in_label)
                    pattern = {e: 0 for e in all_edge_indices}
                    pattern[in_idx] = 1
                    edge_patterns.append(pattern)
                    
            else:
                # Skip: all edges OFF
                categories.append('skip')
                edge_patterns.append({e: 0 for e in all_edge_indices})
                
                # Routing: one incoming ON, one outgoing ON
                for in_label, in_idx in incoming.items():
                    for out_label, out_idx in outgoing.items():
                        categories.append(f'{in_label}→{out_label}')
                        pattern = {e: 0 for e in all_edge_indices}
                        pattern[in_idx] = 1
                        pattern[out_idx] = 1
                        edge_patterns.append(pattern)
            
            info = {
                'categories': categories,
                'n_cats': len(categories),
                'incoming': incoming,
                'outgoing': outgoing,
                'edge_patterns': edge_patterns,
                'all_edges': all_edge_indices,
                'is_source': is_source,
                'is_sink': is_sink,
            }
            
            node_info[node] = info
            var_ranges.append(len(categories))
    
    total_dim = sum(var_ranges)
    
    return node_info, var_ranges, node_order, total_dim


def edge_labels_to_node_labels(edge_labels, node_info, node_order):
    """
    Convert binary edge labels (batch, 264) to node categorical labels (batch, 144).
    
    For each node, find which category matches the observed edge pattern.
    
    Args:
        edge_labels: (batch, 264) binary
        node_info: from build_node_categories
        node_order: list of (i,j)
    
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
                # Shouldn't happen with valid paths, default to skip (cat 0)
                # or first category for source/sink
                node_labels[b, n_idx] = 0
    
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
    print("Building node categories for 12x12 grid...")
    node_info, var_ranges, node_order, total_dim = build_node_categories(12)
    
    print(f"\nTotal nodes: {len(node_order)}")
    print(f"Total output dim: {total_dim}")
    print(f"Variable ranges: min={min(var_ranges)}, max={max(var_ranges)}")
    
    # Count by type
    from collections import Counter
    cat_counts = Counter(var_ranges)
    print(f"\nCategory distribution:")
    for n_cats, count in sorted(cat_counts.items()):
        print(f"  {n_cats} categories: {count} nodes")
    
    # Show examples
    print(f"\nSource (0,0): {node_info[(0,0)]['categories']}")
    print(f"Sink (11,11): {node_info[(11,11)]['categories']}")
    print(f"Top edge (0,5): {node_info[(0,5)]['categories']}")
    print(f"Interior (5,5): {node_info[(5,5)]['categories']}")
    
    # Show edge patterns for interior node
    print(f"\nInterior (5,5) edge patterns:")
    info = node_info[(5, 5)]
    for cat_idx, (name, pattern) in enumerate(zip(info['categories'], info['edge_patterns'])):
        active = [f"e{e}=1" for e, v in pattern.items() if v == 1]
        print(f"  Cat {cat_idx} ({name}): {active if active else 'all OFF'}")
    
    # Test label conversion
    print(f"\nTesting edge→node label conversion...")
    dummy_edges = np.zeros((2, 264), dtype=int)
    # Sample 0: path goes right from source
    dummy_edges[0, node_info[(0,0)]['outgoing']['R']] = 1  # R[0,0] = 1
    
    node_labels = edge_labels_to_node_labels(dummy_edges, node_info, node_order)
    print(f"  Source category: {node_labels[0, 0]} = {node_info[(0,0)]['categories'][node_labels[0, 0]]}")
    print(f"  Node (0,1) category: {node_labels[0, 1]} = {node_info[(0,1)]['categories'][node_labels[0, 1]]}")