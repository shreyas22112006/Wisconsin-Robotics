# Wisconsin Robotics — URC Autonomous Path Planner

Autonomous terrain-aware path planner built for the [University Rover Challenge](https://urc.marssociety.org/). Takes a LiDAR scan of the terrain and a set of GPS waypoints, and computes the safest traversable route for the rover across all targets.

## What it does

1. **Loads and cleans** a `.laz` LiDAR point cloud — voxel downsampling + outlier removal
2. **Builds a KNN graph** over the cleaned point cloud using a KDTree
3. **Weights edges by slope** — edges steeper than the rover's limit are rejected entirely
4. **Snaps GPS waypoints** to the nearest connected node in the graph
5. **Solves TSP** over all waypoints using brute-force A* (optimal for ≤10 targets)
6. **Outputs a path** with slope statistics and an interactive map saved as `path_output.html`

## Pipeline

```
.laz file
    └── load_and_clean_lidar()     voxel hash filter + max-Z representative point
            └── build_knn()        KDTree, K=10 nearest neighbors
                    └── build_graph_vectorized()   slope-weighted adjacency list
                            └── GPS waypoints
                                    └── find_nearest_node()    2D snap to graph
                                            └── compute_cost_matrix()   A* all pairs
                                                    └── find_best_order()   brute-force TSP
                                                            └── visualize_path()   folium HTML map
```

## File Structure

```
├── main.py                        entry point
├── configs/
│   └── settings.py                K_NEIGHBORS, MAX_SLOPE_DEG, SLOPE_MULTIPLIER
├── src/
│   ├── pointcloud/
│   │   ├── load_clean.py          voxel hash filter, outlier removal
│   │   └── knn_builder.py         KDTree + KNN query
│   ├── pathplanning/
│   │   ├── graph_builder.py       vectorized slope-weighted graph construction
│   │   ├── astar.py               A* with 3D Euclidean heuristic
│   │   └── multi_target.py        cost matrix, TSP brute-force, path stitching
│   └── utils/
│       ├── geo_helpers.py         vectorized edge weight computation
│       ├── nearest_point.py       2D GPS-to-graph node snapping
│       ├── point_conversion.py    EPSG detection, GPS to projected XY
│       └── visualization.py       folium interactive map output
├── testing/
│   ├── test_pipeline.py           runs load + KNN + graph, prints stats
│   └── graph_stats.py             node/edge degree statistics
└── docs/
    └── documentation.pdf          full technical writeup
```

## Requirements

```
laspy[lazrs]
numpy
scipy
pyproj
folium
```

Install with:

```bash
pip install "laspy[lazrs]" numpy scipy pyproj folium
```

## Running

```bash
python main.py
```

You will be prompted for:
- Path to a `.laz` or `.las` LiDAR file
- GPS coordinates of the start point (`lat lon`)
- Number of target waypoints and their GPS coordinates

The planner prints slope statistics for the final path and saves an interactive satellite map to `path_output.html`.

## Configuration

Edit `configs/settings.py` to tune the planner:

| Parameter | Default | Effect |
|---|---|---|
| `K_NEIGHBORS` | `10` | Number of KNN edges per point |
| `MAX_SLOPE_DEG` | `36` | Maximum traversable slope — steeper edges are rejected |
| `SLOPE_MULTIPLIER` | `5` | How heavily slope penalizes edge cost vs. flat distance |

## Testing

```bash
cd testing
python test_pipeline.py
```

Runs the load, KNN, and graph stages and prints node/edge statistics without requiring GPS input.

## Documentation

Full technical documentation (algorithm derivations, voxel size selection, complexity analysis, coordinate system notes) is in `docs/documentation.pdf`.
