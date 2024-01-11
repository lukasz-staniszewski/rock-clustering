from collections import Counter, defaultdict
from typing import List

import numpy as np
from process_data import ClusteringPoint

# approximate functions
RATIONAL_ADD = lambda x: (1 + x) / (1 - x)
RATIONAL_SUB = lambda x: (1 - x) / (1 + x)
RATIONAL_EXP = lambda x: np.exp(x) / (1 - x)
RATIONAL_SIN = lambda x: np.sin(x) / (1 - x)

METRIC_SILHOUETTE = "silhouette"
METRIC_PURITY = "purity"


def jaccard(a: np.ndarray[bool], b: np.ndarray[bool]) -> float:
    assert a.shape == b.shape
    return np.sum(a == b) / a.shape[0]


def jaccard_distance(a: np.ndarray[bool], b: np.ndarray[bool]) -> float:
    return 1 - jaccard(a, b)


def is_jaccard_similar(a: np.ndarray[bool], b: np.ndarray[bool], theta_threshold: float = 0.5) -> bool:
    return jaccard(a, b) >= theta_threshold


def purity(points: List[ClusteringPoint], skip_outliers: bool = False) -> float:
    """Calculates the purity of all the clusters."""
    if not points:
        return 0
    purity_points = [point for point in points if point.y is not None] if skip_outliers else points

    by_cluster = defaultdict(list)
    for point in purity_points:
        by_cluster[point.output_cluster_idx].append(point.y)

    return sum(max(Counter(cluster).values()) for cluster in by_cluster.values()) / len(purity_points)


def silhouette(points: List[ClusteringPoint], skip_outliers: bool = False) -> float:
    """Calculates the silhouette metric for all the points (avg)."""

    if not points:
        return 0
    silhouette_points = [point for point in points if point.y is not None] if skip_outliers else points

    by_cluster = defaultdict(list)
    for point in silhouette_points:
        by_cluster[point.output_cluster_idx].append(point)

    silhouettes = []
    for point_i in silhouette_points:
        a_i = np.mean(
            [
                jaccard_distance(point_i.x, point_j.x)
                for point_j in by_cluster[point_i.output_cluster_idx]
                if point_i.original_idx != point_j.original_idx
            ]
        )

        b_i = min(
            [
                np.mean([jaccard_distance(point_i.x, point_j.x) for point_j in by_cluster[cluster_idx]])
                for cluster_idx in by_cluster.keys()
                if cluster_idx != point_i.output_cluster_idx
            ]
        )
        silhouettes.append((b_i - a_i) / max(a_i, b_i))
    return np.mean(np.array(silhouettes))
