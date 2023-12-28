from typing import List

import numpy as np
from tqdm import tqdm

from .entities import Cluster, ClusteringPoint, Heap, RockInput
from .metrics import is_jaccard_similar, jaccard


def get_point_neighbors(
    point: ClusteringPoint, points: List[ClusteringPoint], theta: float
):
    """Get all points that are jaccard similar to the given point."""
    neighbors = []
    for other_point in points:
        if (
            is_jaccard_similar(point.x, other_point.x, theta)
            and point.idx != other_point.idx
        ):
            neighbors.append(other_point)
    return neighbors


def compute_points_links(
    points: List[ClusteringPoint], theta: float
) -> np.ndarray:
    """For each point, compute the number of links to other points.
    Returns the `links` matrix of shape NxN, where links[i][j] is the number of
    links between points i and j.
    """
    neighbors = [
        get_point_neighbors(point=point, points=points, theta=theta)
        for point in tqdm(points, desc="Computing neighbors")
    ]
    links = np.zeros((len(points), len(points)))
    for point in tqdm(points, desc="Computing links"):
        point_neighbors = neighbors[point.idx]

        # if two different points are neighbors of the same point, they are linked
        for neighbor_prev_idx in range(len(point_neighbors) - 1):
            for neighbor_next_idx in range(
                neighbor_prev_idx + 1, len(point_neighbors)
            ):
                links[point_neighbors[neighbor_next_idx].idx][
                    point_neighbors[neighbor_prev_idx].idx
                ] += 1
                links[point_neighbors[neighbor_prev_idx].idx][
                    point_neighbors[neighbor_next_idx].idx
                ] += 1
    return links


def get_clusters_links(
    cluster_A: Cluster, cluster_B: Cluster, links: np.ndarray
) -> int:
    """Compute the number of links between cluster A and B (sum of links between points)."""
    return sum(
        [
            links[point_A.idx][point_B.idx]
            for point_A in cluster_A.points
            for point_B in cluster_B.points
        ]
    )
