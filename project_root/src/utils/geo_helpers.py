import numpy as np
from configs.settings import MAX_SLOPE_DEG, SLOPE_MULTIPLIER

# Weight = distance * (1 + overall slope cost)
def compute_edge_weight(i, j, dist, points):
    dx = points[j][0] - points[i][0]
    dy = points[j][1] - points[i][1]
    dz = points[j][2] - points[i][2]

    horizontal = np.sqrt(dx*dx + dy*dy)
    horizontal = max(horizontal, 1e-6)

    slope_radian = np.arctan(abs(dz) / horizontal)
    slope_degree = np.degrees(slope_radian)

    if slope_degree > MAX_SLOPE_DEG:
        return None

    slope_cost = (slope_degree / MAX_SLOPE_DEG) * SLOPE_MULTIPLIER
    edge_weight = dist * (1 + slope_cost)

    return edge_weight
