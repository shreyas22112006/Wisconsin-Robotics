import laspy
import numpy as np
from pyproj import Transformer

def get_epsg(laz_path):
    """
    Reads the EPSG code from the LAZ file header.
    
    LiDAR data is always stored in a specific coordinate system (projection).
    The EPSG code is a standard number that identifies which coordinate system
    is being used. For example:
        EPSG:4326  -> standard GPS lat/lon (WGS84)
        EPSG:32612 -> UTM Zone 12N (likely what your Utah LiDAR uses)
    
    We need this code so we can convert GPS coordinates into the exact same
    coordinate system as the LiDAR points, making them directly comparable.
    
    Args:
        laz_path: path to the LAZ/LAS file
    Returns:
        epsg: integer EPSG code of the LAZ file's coordinate system
    """

    # Open the LAZ file and read its header metadata
    # We only need the header here, not the actual point data
    laz = laspy.read(laz_path)

    # Extract the CRS (Coordinate Reference System) object from the header
    # This contains all the projection information about how points are stored
    crs = laz.header.parse_crs()

    # Convert the CRS object to a simple integer EPSG code
    # e.g. 32612 for UTM Zone 12N
    epsg = crs.to_epsg()

    print(f"LAZ file EPSG: {epsg}")
    return epsg


def gps_to_xyz(lat, lon, alt, epsg):
    """
    Converts a GPS coordinate (lat, lon, alt) into the same coordinate
    system as the LiDAR points in the LAZ file.
    
    Why this matters:
        GPS coordinates are in WGS84 (EPSG:4326) — a global lat/lon system.
        LiDAR points are stored in a local projected system (e.g. UTM).
        These two systems are incompatible — you cannot directly compare them.
        This function converts the GPS point into the LiDAR's coordinate system
        so that find_nearest_node can correctly find the closest LiDAR point.
    
    Args:
        lat:  latitude in degrees (WGS84)
        lon:  longitude in degrees (WGS84)
        alt:  altitude in metres (kept as z directly)
        epsg: EPSG code of the LAZ file's coordinate system (from get_epsg)
    Returns:
        x, y, z: coordinates in the same projection as the LiDAR points
    """

    # Create a transformer that converts FROM WGS84 GPS (EPSG:4326)
    # TO the LAZ file's coordinate system (e.g. EPSG:32612 for UTM Zone 12N)
    # always_xy=True ensures we always pass (longitude, latitude) in that order
    # regardless of what the projection expects — avoids a common axis-swap bug
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    # Transform longitude and latitude into projected x and y coordinates
    # Note: transformer.transform takes (longitude, latitude) not (latitude, longitude)
    # because always_xy=True enforces x (lon) before y (lat)
    x, y = transformer.transform(lon, lat)

    # Altitude is used directly as z — no transformation needed
    # since both coordinate systems use metres for height
    z = alt

    return x, y, z