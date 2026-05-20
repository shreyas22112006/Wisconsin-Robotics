import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pathplanning.graph_builder import build_graph_vectorized
from src.pointcloud.knn_builder import build_knn
from src.pointcloud.load_clean import load_and_clean_lidar
import graph_stats


def test_pipeline():
    lidar_path = input("Enter file path : ").strip()

    start_time = time.time()
    points = load_and_clean_lidar(lidar_path)
    print(f"Loaded and cleaned point cloud in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    neighbour_indices, neighbour_distances = build_knn(points)
    print(f"KNN built in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    graph = build_graph_vectorized(points, neighbour_indices, neighbour_distances)
    print(f"Graph built in {time.time() - start_time:.2f} seconds.")

    stats = graph_stats.compute_graph_stats(graph)
    graph_stats.print_graph_stats(stats)


test_pipeline()
