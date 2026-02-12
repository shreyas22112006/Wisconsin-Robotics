import sys
import os

# Add parent folder of project_root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.point_conversion import gps_to_xyz
from src.pointcloud.load_clean import load_and_clean_lidar
from src.pointcloud.knn_builder import build_knn
from src.pathplanning.graph_builder import build_graph_vectorized
from src.utils.nearest_point import find_nearest_node

def main():
    file_path = input("Enter file path : ").strip()
    points = load_and_clean_lidar(file_path)
    neighbour_indices, neighbour_distances, tree = build_knn(points)

    lat, lon, alt = map(float, input(f"Enter gps info of starting point (lat lon alt) : ").split())

    x, y, z = gps_to_xyz(lat, lon, alt)
    start = [x, y, z]
    start_idx, start_nearest_point = find_nearest_node(start, points, tree)

    NT = int(input("Enter number of targets : "))
    targets = []
    nearest_nodes = []

    nearest_nodes.append([int(start_idx), start_nearest_point.tolist()])

    for i in range(NT):
        lat, lon, alt = map(float, input(f"Enter gps info of {i+1}th target (lat lon alt) : ").split())
        x, y, z = gps_to_xyz(lat, lon, alt)
        coords = [x, y, z]
        targets.append(coords)
        nearest_idx, nearest_point = find_nearest_node(coords, points, tree)
        nearest_nodes.append([int(nearest_idx), nearest_point.tolist()])

    graph = build_graph_vectorized(points, neighbour_indices, neighbour_distances)
    nearest_nodes.append(graph)
    '''
    [start : [nearest_index, nearest_point]
    target1 : [nearest_index, nearest_point]


    targetn : [nearest_index, nearest_point]
    graph
    ]
    '''

    return nearest_nodes
    


if __name__ == "__main__":
    main()