import numpy as np
from scipy.spatial import cKDTree

def find_nearest_node(point, points_array, tree):
    # Query nearest neighbor using a pre-built KD-tree (same points_array).
    distance, nearest_index = tree.query(point)
    nearest_point = points_array[nearest_index]

    return nearest_index, nearest_point
