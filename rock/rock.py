from heapq import heapify, heappush
from typing import List

from .cluster import compute_points_links
from .entities import Cluster, Heap, RockInput


class RockAlgorithm:
    input_dataset: RockInput
    theta: float
    k: int
    clusters: List[Cluster] = []
    Q: Heap[Cluster] = []
    q: List[Heap[Cluster]] = []

    def __init__(self, input_dataset: RockInput, k: int, theta: float):
        self.input_dataset = input_dataset
        self.k = k
        self.theta = theta
        self.points_links = compute_points_links(
            points=input_dataset.data, theta=theta
        )

    def init_clusters(self) -> None:
        for point in self.input_dataset.data:
            self.clusters.append(Cluster(points=[point]))
            self.Q.append(Cluster(points=[point]))
            self.q.append([])
