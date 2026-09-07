import numpy as np
import heapq
import itertools
import argparse
from pathlib import Path

def neighbours_4(r, c, N):
    """4-connected neighbours of node (r, c) on an NxN grid."""
    for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        rn, cn = r + dr, c + dc
        if 0 <= rn < N and 0 <= cn < N:
            yield rn, cn


def vertex_index(r, c, N):
    """Flatten (row, col) to a single index."""
    return r * N + c


def build_edge_list(N):
    """
    Build the canonical edge list for a 4-connected NxN grid.
    Each edge is (r1, c1, r2, c2) with vertex_index(r1,c1) < vertex_index(r2,c2).
    
    Returns:
        edges: list of (r1, c1, r2, c2) tuples
        edge_index: dict mapping (r1, c1, r2, c2) -> index in the list
        node_to_edges: dict mapping (r, c) -> list of edge indices incident to that node
    """
    edges = []
    edge_set = set()
    
    for r in range(N):
        for c in range(N):
            for rn, cn in neighbours_4(r, c, N):
                # Canonical ordering: smaller vertex index first
                v1 = vertex_index(r, c, N)
                v2 = vertex_index(rn, cn, N)
                if v1 < v2:
                    edge = (r, c, rn, cn)
                else:
                    edge = (rn, cn, r, c)
                
                if edge not in edge_set:
                    edge_set.add(edge)
                    edges.append(edge)
    
    # Sort for deterministic ordering:
    # horizontal edges first (same row), then vertical edges (same column)
    # Total = 2 * N * (N-1): N*(N-1) horizontal + N*(N-1) vertical
    horizontal = sorted([e for e in edges if e[0] == e[2]])  # same row
    vertical = sorted([e for e in edges if e[1] == e[3]])    # same col
    edges = horizontal + vertical
    
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # Build node-to-edge mapping
    node_to_edges = {}
    for r in range(N):
        for c in range(N):
            node_to_edges[(r, c)] = []
    
    for i, (r1, c1, r2, c2) in enumerate(edges):
        node_to_edges[(r1, c1)].append(i)
        node_to_edges[(r2, c2)].append(i)
    
    return edges, edge_index, node_to_edges


# =============================================================================
# Dijkstra under 4-connectivity
# =============================================================================

def dijkstra_4conn(vertex_weights, source, target):
    """
    Run Dijkstra on a 4-connected grid with given vertex weights.
    
    The cost of traversing an edge (u, v) is the cost of the destination vertex v.
    (This matches the Pogancic convention where vertex_weights encode terrain costs.)
    
    Args:
        vertex_weights: (N, N) array of per-vertex costs
        source: (row, col) tuple
        target: (row, col) tuple
    
    Returns:
        path: list of (row, col) tuples forming the shortest path (inclusive of source and target)
        cost: total path cost
    """
    N = vertex_weights.shape[0]
    dist = np.full((N, N), np.inf)
    prev = np.full((N, N, 2), -1, dtype=np.int32)
    visited = np.zeros((N, N), dtype=bool)
    
    sr, sc = source
    dist[sr, sc] = vertex_weights[sr, sc]
    
    # Priority queue: (cost, row, col)
    pq = [(dist[sr, sc], sr, sc)]
    
    while pq:
        d, r, c = heapq.heappop(pq)
        
        if visited[r, c]:
            continue
        visited[r, c] = True
        
        if (r, c) == target:
            break
        
        for rn, cn in neighbours_4(r, c, N):
            if not visited[rn, cn]:
                new_dist = d + vertex_weights[rn, cn]
                if new_dist < dist[rn, cn]:
                    dist[rn, cn] = new_dist
                    prev[rn, cn] = [r, c]
                    heapq.heappush(pq, (new_dist, rn, cn))
    
    # Reconstruct path
    path = []
    r, c = target
    while (r, c) != (-1, -1):
        path.append((r, c))
        pr, pc = prev[r, c]
        r, c = int(pr), int(pc)
    
    path.reverse()
    
    assert path[0] == source, f"Path doesn't start at source: {path[0]} != {source}"
    assert path[-1] == target, f"Path doesn't end at target: {path[-1]} != {target}"
    
    return path, float(dist[target[0], target[1]])


def node_path_to_node_matrix(path, N):
    """Convert a list of (r, c) nodes to an NxN binary indicator matrix."""
    matrix = np.zeros((N, N), dtype=np.float32)
    for r, c in path:
        matrix[r, c] = 1.0
    return matrix


def node_path_to_edge_vector(path, edge_index, num_edges, N):
    """
    Convert a list of (r, c) nodes forming a path to a binary edge vector.

    For each consecutive pair of nodes in the path, activate the corresponding edge.
    """
    edge_vec = np.zeros(num_edges, dtype=np.float32)

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        # Look up the canonical edge (smaller vertex index first)
        v1 = vertex_index(r1, c1, N)
        v2 = vertex_index(r2, c2, N)
        
        if v1 < v2:
            edge = (r1, c1, r2, c2)
        else:
            edge = (r2, c2, r1, c1)
        
        assert edge in edge_index, f"Edge {edge} not found in edge list!"
        edge_vec[edge_index[edge]] = 1.0
    
    return edge_vec


# =============================================================================
# Main preprocessing
# =============================================================================

def preprocess_split(maps, vertex_weights, N, edges, edge_index, split_name):
    """
    Preprocess one data split.
    
    Returns:
        edge_labels: (num_samples, num_edges) binary edge vectors
        node_labels: (num_samples, N, N) binary node matrices  
        costs: (num_samples,) path costs under 4-connectivity
    """
    num_samples = maps.shape[0]
    num_edges = len(edges)
    
    edge_labels = np.zeros((num_samples, num_edges), dtype=np.float32)
    node_labels = np.zeros((num_samples, N, N), dtype=np.float32)
    costs = np.zeros(num_samples, dtype=np.float32)
    
    source = (0, 0)
    target = (N - 1, N - 1)
    
    failed = 0
    for i in range(num_samples):
        if i % 500 == 0:
            print(f"  [{split_name}] Processing sample {i}/{num_samples}...")
        
        try:
            path, cost = dijkstra_4conn(vertex_weights[i], source, target)
            edge_labels[i] = node_path_to_edge_vector(path, edge_index, num_edges, N)
            node_labels[i] = node_path_to_node_matrix(path, N)
            costs[i] = cost
        except Exception as e:
            print(f"  WARNING: Sample {i} failed: {e}")
            failed += 1
    
    if failed > 0:
        print(f"  [{split_name}] {failed}/{num_samples} samples failed!")
    
    return edge_labels, node_labels, costs


def load_split_data(data_dir, split, prefix):
    """Load data for a split, concatenating partN files if needed."""
    single = data_dir / f"{split}_{prefix}.npy"
    if single.exists():
        return np.load(single)

    # Try loading partN files
    parts = sorted(data_dir.glob(f"{split}_{prefix}_part*.npy"))
    if parts:
        arrays = [np.load(p) for p in parts]
        print(f"    Concatenated {len(parts)} parts for {split}_{prefix}")
        return np.concatenate(arrays, axis=0)

    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess Warcraft dataset for 4-connectivity edge-based prediction")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the NxN directory from the raw dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to data_dir/preprocessed_4conn)")
    parser.add_argument("--grid_size", type=int, required=True,
                        help="Grid size N (e.g. 12 or 30)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "preprocessed_4conn"
    output_dir.mkdir(parents=True, exist_ok=True)

    N = args.grid_size
    expected_edges = 2 * N * (N - 1)

    # Build edge list
    print(f"Building 4-connected edge list for {N}x{N} grid...")
    edges, edge_index, node_to_edges = build_edge_list(N)
    num_edges = len(edges)
    print(f"  {num_edges} edges (expected {expected_edges})")
    assert num_edges == expected_edges, f"Expected {expected_edges} edges, got {num_edges}"
    
    # Save edge metadata
    edges_array = np.array(edges)  # (264, 4) - each row is (r1, c1, r2, c2)
    np.save(output_dir / "edges.npy", edges_array)
    
    # Save node_to_edges mapping
    node_to_edges_dict = {f"{r},{c}": indices for (r, c), indices in node_to_edges.items()}
    np.savez(output_dir / "node_to_edges.npz", **{k: np.array(v) for k, v in node_to_edges_dict.items()})
    
    # Process each split
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split...")

        maps = load_split_data(data_dir, split, "maps")
        vertex_weights = load_split_data(data_dir, split, "vertex_weights")

        if maps is None:
            print(f"  Skipping {split}: no data found")
            continue
        
        print(f"  Maps shape: {maps.shape}")
        print(f"  Vertex weights shape: {vertex_weights.shape}")
        
        edge_labels, node_labels, costs = preprocess_split(
            maps, vertex_weights, N, edges, edge_index, split
        )
        
        # Save
        np.save(output_dir / f"{split}_maps.npy", maps)
        np.save(output_dir / f"{split}_vertex_weights.npy", vertex_weights)
        np.save(output_dir / f"{split}_edge_labels.npy", edge_labels)
        np.save(output_dir / f"{split}_node_labels.npy", node_labels)
        np.save(output_dir / f"{split}_costs.npy", costs)
        
        print(f"  Saved: edge_labels {edge_labels.shape}, node_labels {node_labels.shape}")
        print(f"  Path cost stats: min={costs.min():.1f}, max={costs.max():.1f}, mean={costs.mean():.1f}")
        
        # Sanity checks
        edges_per_path = edge_labels.sum(axis=1)                    # (num_samples,)
        nodes_per_path = node_labels.sum(axis=(1, 2))               # (num_samples,) - sum over both spatial dims
        print(f"\n  --- Sanity checks ({split}) ---")
        print(f"  Avg edges per path: {edges_per_path.mean():.1f}")
        print(f"  Avg nodes per path: {nodes_per_path.mean():.1f}")
        print(f"  Edges per path = Nodes per path - 1? "
              f"{np.allclose(edges_per_path, nodes_per_path - 1)}")
    
    # Print edge indexing summary
    n_horiz = N * (N - 1)
    print("\n--- Edge Indexing Summary ---")
    print(f"Edges 0-{n_horiz-1}: horizontal (same row)")
    print(f"  e.g., edge 0 = {edges[0]} (nodes ({edges[0][0]},{edges[0][1]})-({edges[0][2]},{edges[0][3]}))")
    print(f"Edges {n_horiz}-{num_edges-1}: vertical (same column)")
    print(f"  e.g., edge {n_horiz} = {edges[n_horiz]} (nodes ({edges[n_horiz][0]},{edges[n_horiz][1]})-({edges[n_horiz][2]},{edges[n_horiz][3]}))")
    
    print(f"\nPreprocessing complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()