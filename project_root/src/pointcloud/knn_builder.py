import numpy as np
from configs.settings import K_NEIGHBORS
from scipy.spatial import cKDTree


def build_knn(points):
    tree = cKDTree(points)

    distances, indices = tree.query(points, k=K_NEIGHBORS + 1)

    neighbour_indices = indices[:, 1:]
    neighbour_distances = distances[:, 1:]

    return neighbour_indices, neighbour_distances
