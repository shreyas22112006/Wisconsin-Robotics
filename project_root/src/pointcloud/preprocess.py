import numpy as np
import pickle
import os
from src.pointcloud.load_clean import load_and_clean_lidar
from src.pointcloud.knn_builder import build_knn
from src.pathplanning.graph_builder import build_graph_vectorized
from src.utils.point_conversion import get_epsg

def preprocess_and_save(laz_path, save_path):
    """
    Runs the full preprocessing pipeline once and saves the result to disk.
    This only needs to be run once per LiDAR file — not every time main runs.
    
    Pipeline:
        LAZ file -> clean points -> KNN -> graph -> save to disk
    
    Args:
        laz_path:  path to the raw LAZ/LAS file
        save_path: path to save the preprocessed data (e.g. "data/map.pkl")
    """

    print("Loading and cleaning point cloud...")
    points = load_and_clean_lidar(laz_path)

    print("Building KNN...")
    neighbour_indices, neighbour_distances, tree = build_knn(points)

    print("Building graph...")
    graph = build_graph_vectorized(points, neighbour_indices, neighbour_distances)

    print("Reading EPSG from LAZ header...")
    epsg = get_epsg(laz_path)

    # Bundle everything needed by main into one dictionary
    data = {
        "points": points,
        "tree": tree,
        "graph": graph,
        "epsg": epsg
    }

    # Save to disk using pickle
    # pickle can serialize complex Python objects like defaultdict and cKDTree
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved preprocessed data to {save_path}")


def load_preprocessed(save_path):
    """
    Loads the preprocessed data from disk.
    Call this in main instead of rerunning the whole pipeline every time.
    
    Args:
        save_path: path to the saved .pkl file
    Returns:
        dictionary containing points, tree, graph, epsg
    """
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded preprocessed data from {save_path}")
    return data