from src.utils.geo_helpers import *
import numpy as np
from collections import defaultdict

'''
def build_graph(points, neighbour_indices, neighbour_distances):
    # Building the graph using adjacency list
    # adjacency list
    # graph[i] = list of (neighbour_index, weight)
    N = points.shape[0]

    # makes N empty lists for each node inside graph list
    graph = [[] for _ in range(N)]
    added_edges = set()

    for i in range(N):
        for j, dist in zip(neighbour_indices[i], neighbour_distances[i]):

            a, b = min(i, j), max(i, j)
            if (a, b) in added_edges:
                continue

            weight = compute_edge_weight(i, j, dist, points)
            if weight is None:
                continue

            graph[i].append((j, weight))
            graph[j].append((i, weight))
            added_edges.add((a, b))

    return graph
'''

def build_graph_vectorized(points, neighbour_indices, neighbour_distances):
    N = points.shape[0]

    # Step 1: flatten neighbors
    i_indices = np.repeat(np.arange(N), [len(n) for n in neighbour_indices])
    j_indices = np.concatenate(neighbour_indices)
    dists = np.concatenate(neighbour_distances)

    # Step 2: vectorized edge weights
    edge_weights = compute_edge_weight_vectorized(i_indices, j_indices, dists, points)

    # Step 3: remove invalid edges (too steep)
    mask = ~np.isnan(edge_weights)  # True for valid edges
    i_indices = i_indices[mask]
    j_indices = j_indices[mask]
    edge_weights = edge_weights[mask]

    # Step 4: remove duplicate edges
    a = np.minimum(i_indices, j_indices)
    b = np.maximum(i_indices, j_indices)
    edges = np.stack([a, b], axis=1)
    edges_unique, unique_idx = np.unique(edges, axis=0, return_index=True)
    a = a[unique_idx]
    b = b[unique_idx]
    edge_weights = edge_weights[unique_idx]

    # Step 5: build adjacency list (compatible with old graph_stats)
    graph = defaultdict(list)
    for u, v, w in zip(a, b, edge_weights):
        graph[u].append((v, w))
        graph[v].append((u, w))

    return graph