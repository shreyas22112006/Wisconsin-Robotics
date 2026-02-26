import numpy as np
from configs.settings import K_NEIGHBORS  # K_NEIGHBORS = 10, how many candidates to check

def find_nearest_node(point, points_array, tree, graph):
    # Convert input [x, y, z] list to NumPy array so cKDTree can process it
    # np.asarray avoids making a copy if it's already a NumPy array
    point = np.asarray(point)

    # Query the KDTree for the K nearest nodes to our GPS point
    # Returns distances and indices sorted closest to furthest
    # e.g. indices = [452, 1203, 87, ...] where 452 is the closest node
    distances, indices = tree.query(point, k=K_NEIGHBORS)
    
    # Iterate through candidates from closest to furthest
    for idx in indices:
        # Check 1: node exists in the graph
        # Check 2: node has at least one edge (not an isolated dead-end node)
        # A node can exist in points_array but have no edges if all its
        # connections were rejected due to being too steep in graph_builder
        if idx in graph and len(graph[idx]) > 0:
            # Return the index and the 3D coordinates of the chosen node
            return idx, points_array[idx]
    
    # Fallback: if all K candidates are isolated nodes (extremely unlikely),
    # just return the geometrically closest one rather than crashing
    return indices[0], points_array[indices[0]]