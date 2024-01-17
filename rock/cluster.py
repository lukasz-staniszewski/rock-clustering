from dataclasses import dataclass
from heapq import heapify, heappush
from typing import Callable, List, Tuple

import numpy as np
from process_data import ClusteringPoint
from tqdm import tqdm

from rock.metrics import is_jaccard_similar

Heap = List


@dataclass
class Cluster:
    """Represents a cluster of points."""

    idx: int
    points: List[ClusteringPoint]
    heap: Heap[Tuple[float, "Cluster"]] | None = None

    def __lt__(self, other):
        return self.idx < other.idx

    def __eq__(self, other):
        return self.idx == other.idx

    def size(self) -> int:
        return len(self.points)

    def init_heap(
        self,
        clusters: List["Cluster"],
        points_links: np.ndarray,
        approx_fn: Callable[[float], float],
        theta: float,
    ) -> None:
        """Initializes the local heap for the cluster (first element is cluster with highest goodness measure)."""
        self.heap = []
        heapify(self.heap)
        for cluster in clusters:
            goodness = get_goodness_measure(
                cluster_A=self,
                cluster_B=cluster,
                points_links=points_links,
                approx_fn=approx_fn,
                theta=theta,
            )
            if cluster.idx != self.idx and goodness > 0:
                # negative goodness to have max heap
                heappush(
                    self.heap,
                    (-goodness, cluster),
                )

    def merge_clusters(self, cluster: "Cluster", new_idx: int) -> "Cluster":
        """Creates a new cluster by merging the current cluster with the given one."""
        return Cluster(
            idx=new_idx,
            points=self.points + cluster.points,
        )


def get_point_neighbors(point: ClusteringPoint, points: List[ClusteringPoint], theta: float):
    """Get all points that are jaccard similar to the given point."""
    neighbors = []
    for other_point in points:
        if point.idx != other_point.idx and is_jaccard_similar(
            transaction_a=point.transaction, transaction_b=other_point.transaction, theta_threshold=theta
        ):
            neighbors.append(other_point)
    return neighbors


def compute_points_links(points: List[ClusteringPoint], theta: float) -> np.ndarray:
    """Computes links matrix by calculating for each pair of points how many common neighbours they have.
    Returns the `links` matrix of shape NxN, where links[i][j] equals to links[j][i] and is the number of
    links between points i and j.
    """
    points_neighbors = [
        get_point_neighbors(point=point, points=points, theta=theta)
        for point in tqdm(points, desc="Links | computing neighbors")
    ]
    links = np.zeros(shape=(len(points), len(points)))
    for point in tqdm(points, desc="Links | computing links"):
        point_neighbors = points_neighbors[point.idx]

        # if two different points are neighbors of the same point, they are linked
        for neighbor_prev_idx in range(len(point_neighbors) - 1):
            for neighbor_next_idx in range(neighbor_prev_idx + 1, len(point_neighbors)):
                links[point_neighbors[neighbor_next_idx].idx][point_neighbors[neighbor_prev_idx].idx] += 1
                links[point_neighbors[neighbor_prev_idx].idx][point_neighbors[neighbor_next_idx].idx] += 1
    return links


def get_clusters_links(cluster_A: Cluster, cluster_B: Cluster, points_links: np.ndarray) -> int:
    """Compute the number of links between two clusters: A and B (sum of links between points)."""
    return sum(
        [points_links[point_A.idx][point_B.idx] for point_A in cluster_A.points for point_B in cluster_B.points]
    )


def get_expected_cluster_size_penalty(
    cluster: Cluster,
    approx_fn: Callable[[float], float],
    theta: float,
) -> float:
    """Compute the expected cluster size penalty for inference process."""
    return cluster.size() ** (1 + 2 * approx_fn(theta))


def get_goodness_measure(
    cluster_A: Cluster,
    cluster_B: Cluster,
    points_links: np.ndarray,
    approx_fn: Callable[[float], float],
    theta: float,
    eps: float = 1e-6,
) -> float:
    """Compute the goodness measure between two clusters: A and B (sum of links between points)."""
    links_sum = get_clusters_links(cluster_A=cluster_A, cluster_B=cluster_B, points_links=points_links)
    size_penalty = (
        (cluster_A.size() + cluster_B.size()) ** (1 + 2 * approx_fn(theta))
        - (cluster_A.size() ** (1 + 2 * approx_fn(theta)))
        - (cluster_B.size() ** (1 + 2 * approx_fn(theta)))
    )
    return links_sum / (size_penalty + eps)
