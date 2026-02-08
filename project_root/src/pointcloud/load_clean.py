import laspy
import numpy as np
import open3d as o3d


def load_and_clean_lidar(path):
    map = laspy.read(path)

    x = np.asarray(map.x)
    y = np.asarray(map.y)
    z = np.asarray(map.z)
    
    # map.x, map.y, map.z are arrays of coordinates, each contians contains the x/y/z-coordinate of every single LiDAR point in your point cloud.
    # vstack stacks them vertically -> gives 3-row array : [ [x1, x2,...], [y1, y2,...], [z1, z2,...] ]
    # .T transposes it so each row becomes one point : [x, y, z]
    # points.shape[0] gives no. of points in LiDAR cloud
    points = np.vstack((x, y, z)).T

    # this creates a pointcloud object, like creating a container for all your 3d points
    pcd = o3d.geometry.PointCloud()

    # this converts NumPy array into Open3D format, after this line pcd now actually contains all the points from LiDAR data
    pcd.points = o3d.utility.Vector3dVector(points)

    # We can compute the average distance from each point to its nearest neighbors.
    # If a point is too far from its neighbors compared to most points, it’s likely an outlier (noise).
    # pcd had the cleaned point cloud data, ind -> indices of the kept point
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)

    pcd = pcd.voxel_down_sample(voxel_size = 0.35)
    
    # extract points from the cleaned cloud as a numpy array
    points = np.asarray(pcd.points)
    return points
