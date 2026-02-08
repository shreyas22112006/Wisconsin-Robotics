import sys
import os

# Add parent folder of project_root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.pathplanning.graph_builder import build_graph_vectorized
from src.pointcloud.knn_builder import build_knn
from src.pointcloud.load_clean import load_and_clean_lidar

from configs.settings import path
from testing import graph_stats

import time


def main():
    lidar_path = path

    start_time = time.time()
    points = load_and_clean_lidar(lidar_path)
    end_time = time.time()
    print(f"Loaded and cleaned point cloud in {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    neighbour_indices, neighbour_distances = build_knn(points)
    end_time = time.time()
    print(f"KNN built in {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    graph = build_graph_vectorized(points, neighbour_indices, neighbour_distances)
    end_time = time.time()
    print(f"Graph built in {end_time - start_time:.2f} seconds.")

    stats = graph_stats.compute_graph_stats(graph)
    graph_stats.print_graph_stats(stats)


if __name__ == "__main__":
    main()