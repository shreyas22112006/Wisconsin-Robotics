import laspy
import numpy as np
import open3d as o3d


def load_and_clean_lidar(path):
    map = laspy.read(path)

    x = np.asarray(map.x)
    y = np.asarray(map.y)
    z = np.asarray(map.z)

    points = np.vstack((x, y, z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)


    # voxel size = robot width / 4 to 6
    # ASK DEVANSH
    pcd = pcd.voxel_down_sample(voxel_size = 0.2)

    points = np.asarray(pcd.points)
    return points
