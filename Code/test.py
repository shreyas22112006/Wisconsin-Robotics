import laspy
import numpy as np
import lazrs
import open3d as o3d
import math
from scipy.spatial import cKDTree

map = laspy.read("/Users/shreyas/Desktop/UWM/Wisconsin Robotics/data/USGS_LPC_UT_Southern_QL1_2018_12SUH2529_LAS_2019.laz")

# map.x, map.y, map.z are arrays of coordinates, each contians contains the x/y/z-coordinate of every single LiDAR point in your point cloud.
# vstack stacks them vertically -> gives 3-row array : [ [x1, x2,...], [y1, y2,...], [z1, z2,...] ]
# .T transposes it so each row becomes one point : [x, y, z]
# points.shape[0] gives no. of points in LiDAR cloud
points = np.vstack((map.x, map.y, map.z)).T

# this creates a pointcloud object, like creating a container for all your 3d points
# right now it's empty
pcd = o3d.geometry.PointCloud()

# this converts NumPy array into Open3D format, after this line pcd now actually contains all the points from LiDAR data
pcd.points = o3d.utility.Vector3dVector(points)

# We can compute the average distance from each point to its nearest neighbors.
# If a point is too far from its neighbors compared to most points, it’s likely an outlier (noise).
# pcd had the cleaned point cloud data, ind -> indices of the kept point
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

#extract points from the cleaned cloud as a numpy array
points = np.asarray(pcd.points)

# build kdtree, kd-tree is a spatial data struct allows to efficiently find nearest neighbour
# instead of checking every pair of points the kd-tree organises them in a binary space partitioning struct
# this makes nearest neighbour queries fast from O(N^2) to O(logN)
tree = cKDTree(points)

# we do k+1 because when you ask for neighbours, the first one returned is the point itself
# for each point i
# indices[i] -> indies of the 11 closest points
# distances[i] -> contains their corresponding euclidean distances
k = 10
distances, indices = tree.query(points, k=k+1)

# remove self neighbour
neighbour_indices = indices[:, 1:] # shape -> (N,k)
neighbour_distances = distances[:, 1:] # shape -> (N,k)

# Building the graph using adjacency list

# adjacency list
# graph[i] = list of (neighbour_index, weight)
N = points.shape[0]
graph = [[] for _ in range(len(points))]

# Parameters
max_slope_deg = 36
slope_multilpier = 5

def compute_edge_weight(i, j, dist):
    '''
    Weight = distance * (1 + overall slope cost)
    '''

    dx = points[j][0] - points[i][0]
    dy = points[j][1] - points[i][1]
    dz = points[j][2] - points[i][2]
    '''
    We still need to optimise this for uphill and downhill
    We have used abs somewhere and not in some places still need to check on that
    '''
    horizontal = np.sqrt(dx*dx + dy*dy)
    horizontal = max(horizontal, 1e-6)

    slope_radian = np.arctan(abs(dz) / horizontal)
    slope_degree = np.degrees(slope_radian)

    if slope_degree > max_slope_deg:
        return None

    slope_cost = (slope_degree / max_slope_deg) * slope_multilpier

    edge_weight = dist * (1 + slope_cost)

    return edge_weight


# Add Edges

added_edges = set()   # keeps track of which edges already added

for i in range(N):
    for j, dist in zip(neighbour_indices[i], neighbour_distances[i]):

        # Always store the edge in sorted order to avoid duplicates
        a, b = min(i, j), max(i, j)

        if (a, b) in added_edges:
            continue  # edge already handled

        # Compute weight
        weight = compute_edge_weight(i, j, dist)
        if weight is None:
            continue  # slope too steep → skip

        # Add edge both ways (undirected graph)
        graph[i].append((j, weight))
        graph[j].append((i, weight))

        # Mark this edge as added
        added_edges.add((a, b))