import heapq
import numpy as np

def astar(graph, points, start_idx, goal_idx):
    N = len(points)
    goal_pos = points[goal_idx]

    # --- HEURISTIC ---
    # Straight line 3D euclidean distance from node n to the goal
    # This guides A* toward the goal efficiently
    # It never overestimates real cost, so A* stays optimal
    def heuristic(n):
        return np.linalg.norm(points[n] - goal_pos)

    # --- OPEN SET ---
    # This is the priority queue (min-heap) of nodes to explore
    # Each entry is (f_score, node_index)
    # f = g + h, where g = cost so far, h = estimated cost to goal
    # heapq always pops the entry with the lowest f_score first
    open_set = []
    heapq.heappush(open_set, (0, start_idx))

    # --- G SCORE ---
    # Pre-allocated numpy array — O(1) reads/writes by index, much faster than a dict
    # Everything starts as infinity (undiscovered), start node costs 0
    g_score = np.full(N, np.inf)
    g_score[start_idx] = 0.0

    # --- CAME FROM ---
    # came_from[n] = which node we came from to reach n on the best path
    # Used at the end to reconstruct the full path by backtracking
    came_from = {}

    # --- VISITED ARRAY ---
    # Pre-allocated boolean array — faster than a Python set for large graphs
    # visited[n] = True once node n has been fully processed
    visited = np.zeros(N, dtype=bool)

    while open_set:

        # Pop the node with the lowest f_score
        f, current = heapq.heappop(open_set)

        # If we've reached the goal, reconstruct and return the path
        if current == goal_idx:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            path.reverse()
            return path, g_score[goal_idx]

        # Skip if already processed
        if visited[current]:
            continue
        visited[current] = True

        # --- EXPLORE NEIGHBORS ---
        for neighbor, weight in graph[current]:

            if visited[neighbor]:
                continue

            # Tentative g score if we go through current to reach neighbor
            tentative_g = g_score[current] + weight

            # Only update if this path to neighbor is cheaper than any previously found
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))

    # If we exhaust the open set without reaching the goal, no path exists
    return None, float('inf')