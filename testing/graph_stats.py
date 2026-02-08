import numpy as np


def compute_graph_stats(graph):
    num_nodes = len(graph)

    # degree of each node
    degrees = np.array([len(neighbors) for neighbors in graph.values()])

    # each edge appears twice
    total_edges = degrees.sum() // 2

    stats = {
        "num_nodes": num_nodes,
        "num_edges": total_edges,
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "avg_degree": float(degrees.mean()),
        "num_isolated_nodes": int((degrees == 0).sum()),
    }

    return stats


def print_graph_stats(stats):
    """
    Nicely formatted printout for presentation/debugging
    """

    print("\n====== GRAPH STATISTICS ======")
    print(f"Number of nodes           : {stats['num_nodes']}")
    print(f"Number of edges           : {stats['num_edges']}")
    print(f"Minimum node degree       : {stats['min_degree']}")
    print(f"Maximum node degree       : {stats['max_degree']}")
    print(f"Average node degree       : {stats['avg_degree']:.2f}")
    print(f"Isolated nodes (degree 0) : {stats['num_isolated_nodes']}")
    print("================================\n")
