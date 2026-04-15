import os
import pickle
from collections import defaultdict


def _cache_path(laz_path):
    # Store the cache file next to the LAZ file with a .pkl extension
    # e.g. data/mdrs.laz -> data/mdrs.pkl
    base = os.path.splitext(laz_path)[0]
    return base + ".pkl"


def save_cache(laz_path, points, graph):
    path = _cache_path(laz_path)
    # Convert defaultdict to plain dict for safe serialization
    with open(path, 'wb') as f:
        pickle.dump({'points': points, 'graph': dict(graph)}, f)
    print(f"Cache saved to {path}")


def load_cache(laz_path):
    path = _cache_path(laz_path)
    if not os.path.exists(path):
        return None, None
    print(f"Cache found, loading from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Restore as defaultdict(list) so graph access is identical to the original
    graph = defaultdict(list, data['graph'])
    return data['points'], graph
