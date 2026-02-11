from math import *

def gps_to_xyz(lat, lon, alt):
    # WGS84 constants
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2 - f)

    N = a / sqrt(1 - e2 * sin(lat)**2)

    X = (N + alt) * cos(lat) * cos(lon)
    Y = (N + alt) * cos(lat) * sin(lon)
    Z = (N * (1 - e2) + alt) * sin(lat)

    return X, Y, Z