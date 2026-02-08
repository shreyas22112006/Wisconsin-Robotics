from src.utils.geo_helpers import compute_edge_weight

def build_graph(points, neighbour_indices, neighbour_distances):
    # Building the graph using adjacency list
    # adjacency list
    # graph[i] = list of (neighbour_index, weight)
    N = points.shape[0]
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
