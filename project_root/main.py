import sys
import os

# Add the parent folder of project_root to Python's path
# so that imports like "from src.utils..." resolve correctly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.point_conversion import gps_to_xyz
from src.pointcloud.load_clean import load_and_clean_lidar
from src.pointcloud.knn_builder import build_knn
from src.pathplanning.graph_builder import build_graph_vectorized
from src.utils.nearest_point import find_nearest_node
from src.utils.point_conversion import *

def main():
    # Get the LiDAR file path from the user and load it
    # load_and_clean_lidar removes outliers and voxel downsamples the point cloud
    file_path = input("Enter file path : ").strip()
    points = load_and_clean_lidar(file_path)

    # Build the KDTree and KNN structure from the cleaned point cloud
    # neighbour_indices[i] -> indices of K nearest neighbors of point i
    # neighbour_distances[i] -> their corresponding euclidean distances
    # tree -> the KDTree used for fast spatial queries
    neighbour_indices, neighbour_distances, tree = build_knn(points)

    # Build the weighted graph before finding nearest nodes
    # Must be done first so we can check if nodes are connected when snapping GPS points
    # graph[i] -> list of (neighbour_index, edge_weight) for node i
    # Edges with slope > MAX_SLOPE_DEG are rejected
    graph = build_graph_vectorized(points, neighbour_indices, neighbour_distances)

    # Read EPSG once from the LAZ file and reuse for all GPS conversions
    epsg = get_epsg(file_path)

    # Take GPS input for the start point and convert to XYZ cartesian coordinates
    lat, lon, alt = map(float, input("Enter gps info of starting point (lat lon alt) : ").split())
    x, y, z = gps_to_xyz(lat, lon, alt)

    # Snap the start GPS point to the nearest connected node in the graph
    # start_idx -> index of the nearest valid node in points array
    # start_nearest_point -> 3D coordinates of that node
    start_idx, start_nearest_point = find_nearest_node([x, y, z], points, tree, graph)

    # Take the number of target points from the user
    NT = int(input("Enter number of targets : "))
    target_nodes = []

    for i in range(NT):
        # Take GPS input for each target and convert to XYZ
        lat, lon, alt = map(float, input(f"Enter gps info of {i+1}th target (lat lon alt) : ").split())
        x, y, z = gps_to_xyz(lat, lon, alt)

        # Snap each target GPS point to the nearest connected node in the graph
        nearest_idx, nearest_point = find_nearest_node([x, y, z], points, tree, graph)

        # Store each target as a dictionary with its graph index and 3D coordinates
        target_nodes.append({"index": int(nearest_idx), "point": nearest_point.tolist()})

    # Return everything A* will need:
    # - start node index and coordinates
    # - list of target node indices and coordinates
    # - the full weighted graph for pathfinding
    # - the full points array for heuristic distance calculations in A*
    return {
        "start": {"index": int(start_idx), "point": start_nearest_point.tolist()},
        "targets": target_nodes,
        "graph": graph,
        "points": points
    }

if __name__ == "__main__":
    main()