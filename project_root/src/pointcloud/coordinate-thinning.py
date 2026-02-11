'''

import numpy as np
import os
from collections import defaultdict

TARGETS = np.array([
    [0.50, 0.50],
    [0.25, 0.25],
    [0.75, 0.25],
    [0.25, 0.75],
    [0.75, 0.75]
])



def thin_points_stratified(path, cell_size):
    """
    For each cell, keep up to 5 points closest to fixed target positions
    inside the cell.
    """
    map = np.load(path)
    xy = map[:, :2]
    xy_min = xy.min(axis=0)
    shifted = xy - xy_min

    cell_idx = np.floor(shifted / cell_size).astype(int)

    # group point indices by cell
    cells = defaultdict(list)
    for i, (cx, cy) in enumerate(cell_idx):
        cells[(cx, cy)].append(i)

    kept_indices = []

    for (cx, cy), inds in cells.items():

        # we need >= 5 points per cell
        if len(inds) <= 5:
            kept_indices.extend(inds)
            continue

        pts = map[inds] # list of all points within this cell
        # normalize to [0,1] within the cell
        local_xy = (pts[:, :2] - (xy_min + np.array([cx, cy]) * cell_size)) / cell_size

        chosen = []
        for t in TARGETS:
            d2 = np.sum((local_xy - t) ** 2, axis=1)
            idx = np.argmin(d2)
            chosen.append(inds[idx])
            local_xy = np.delete(local_xy, idx, axis=0)
            inds = np.delete(inds, idx, axis=0)

        kept_indices.extend(chosen)

    kept_indices = np.array(kept_indices)
    kept_indices = kept_indices[np.argsort(kept_indices)]
    return map[kept_indices]
'''