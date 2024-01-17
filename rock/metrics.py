from collections import Counter, defaultdict
from typing import List, Set

import numpy as np
from process_data import ClusteringPoint

# approximate functions
RATIONAL_ADD = lambda x: (1 + x) / (1 - x)
RATIONAL_SUB = lambda x: (1 - x) / (1 + x)
RATIONAL_EXP = lambda x: np.exp(x) / (1 - x)
RATIONAL_SIN = lambda x: np.sin(x) / (1 - x)

METRIC_SILHOUETTE = "silhouette"
METRIC_PURITY = "purity"


def jaccard(transaction_a: Set[str], transaction_b: Set[str], eps: float = 1e-6) -> float:
    return len(transaction_a.intersection(transaction_b)) / (len(transaction_a.union(transaction_b)) + eps)


def jaccard_distance(transaction_a: Set[str], transaction_b: Set[str]) -> float:
    return 1 - jaccard(transaction_a=transaction_a, transaction_b=transaction_b)


def is_jaccard_similar(transaction_a: Set[str], transaction_b: Set[str], theta_threshold: float = 0.5) -> bool:
    return jaccard(transaction_a=transaction_a, transaction_b=transaction_b) >= theta_threshold


def purity(points: List[ClusteringPoint], skip_outliers: bool = False, eps: float = 1e-6) -> float:
    """Calculates the purity of all the clusters."""
    if not points:
        return 0
    purity_points = [point for point in points if point.output_cluster_idx is not None] if skip_outliers else points

    by_cluster = defaultdict(list)
    for point in purity_points:
        by_cluster[point.output_cluster_idx].append(point.y)

    return sum(max(Counter(cluster).values()) for cluster in by_cluster.values()) / (len(purity_points) + eps)


def silhouette(points: List[ClusteringPoint], skip_outliers: bool = False) -> float:
    """Calculates the Silhouette metric for all the points (avg)."""

    if not points:
        return 0
    silhouette_points = [point for point in points if point.output_cluster_idx is not None] if skip_outliers else points

    by_cluster = defaultdict(list)
    for point in silhouette_points:
        by_cluster[point.output_cluster_idx].append(point)

    silhouettes = []
    for point_i in silhouette_points:
        if len(by_cluster[point_i.output_cluster_idx]) == 1:
            continue
        a_i = np.mean(
            [
                jaccard_distance(transaction_a=point_i.transaction, transaction_b=point_j.transaction)
                for point_j in by_cluster[point_i.output_cluster_idx]
                if point_i.original_idx != point_j.original_idx
            ]
        )

        b_i = min(
            [
                np.mean(
                    [
                        jaccard_distance(transaction_a=point_i.transaction, transaction_b=point_j.transaction)
                        for point_j in by_cluster[cluster_idx]
                    ]
                )
                for cluster_idx in by_cluster.keys()
                if cluster_idx != point_i.output_cluster_idx
            ]
        )
        silhouettes.append((b_i - a_i) / max(a_i, b_i))
    return np.mean(np.array(silhouettes))
