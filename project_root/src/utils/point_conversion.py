from math import *

def gps_to_xyz(lat, lon, alt):
    # Convert degrees to radians
    lat_rad = radians(lat)
    lon_rad = radians(lon)
    
    # WGS84 constants
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2 - f)
    N = a / sqrt(1 - e2 * sin(lat_rad)**2)
    X = (N + alt) * cos(lat_rad) * cos(lon_rad)
    Y = (N + alt) * cos(lat_rad) * sin(lon_rad)
    Z = (N * (1 - e2) + alt) * sin(lat_rad)
    return X, Y, Z