from heapq import heapify, heappop, heappush
from typing import Callable, List, Tuple

from tqdm import tqdm

from rock.cluster import Cluster, Heap, compute_points_links
from rock.process import RockInput


class RockAlgorithm:
    """Rock algorithm implementation."""

    dataset: RockInput
    theta: float
    k: int
    approx_fn: Callable
    clusters: List[Cluster] = []
    _Q: Heap[Tuple[float, Cluster]] = []

    def __init__(self, dataset: RockInput, k: int, theta: float, approx_fn: Callable, outliers_factor: float = 0.33):
        print("Initializing Rock algorithm...")
        self.dataset = dataset
        self.k = k
        self.theta = theta
        self.approx_fn = approx_fn
        self.outliers_factor = outliers_factor
        self._points_links = compute_points_links(points=dataset.data, theta=theta)
        self._n_iterations = 0
        self._outliers_removed = False
        self._max_idx = -1
        self._init_clusters()
        print("Rock algorithm initialized.")

    def _init_clusters(self) -> None:
        """Initialize all the clusters representing single points, creates local heaps for each cluster and global heap Q."""
        for point in self.dataset.data:
            point_cluster = Cluster(idx=point.idx, points=[point])
            self._max_idx = max(self._max_idx, point.idx)
            self.clusters.append(point_cluster)
        self.n_clusters_start = len(self.clusters)
        self._reset_heaps()

    def _reset_heaps(self) -> None:
        """Resets (or initializes) both: local heaps and the global heap Q."""
        self._reset_qs()
        self._reset_Q()

    def _reset_qs(self) -> None:
        """Resets (or initializes) local heaps for each cluster."""
        for cluster in tqdm(self.clusters, desc="Resetting clusters heaps"):
            cluster.init_heap(
                clusters=self.clusters,
                points_links=self._points_links,
                approx_fn=self.approx_fn,
                theta=self.theta,
            )

    def _reset_Q(self) -> None:
        """Resets (or initializes) the global heap Q."""
        self._Q = []
        heapify(self._Q)
        for cluster in self.clusters:
            if cluster.heap:
                heappush(self._Q, (cluster.heap[0][0], cluster))

    def _update_qs_from_merged(self, cluster_removed_A: Cluster, cluster_removed_B: Cluster) -> None:
        """Updates the local heaps of the neighbours of just removed clusters."""
        neighbours = [cluster[1] for cluster in cluster_removed_A.heap if cluster[1].idx != cluster_removed_B.idx]
        neighbours += [cluster[1] for cluster in cluster_removed_B.heap if cluster[1].idx != cluster_removed_A.idx]

        for cluster in neighbours:
            cluster.init_heap(
                clusters=self.clusters,
                points_links=self._points_links,
                approx_fn=self.approx_fn,
                theta=self.theta,
            )

    def _run_iter(self) -> None:
        """Performs one iteration (merging) of the Rock algorithm along with:
        1. Updating clusters list.
        2. Initializing local heap for the merged cluster.
        3. Updaing local heaps of the neighbours of the merged cluster.
        4. Recreating global heap Q
        """
        # create a new cluster
        best_A: Cluster = heappop(self._Q)[1]
        best_B: Cluster = heappop(best_A.heap)[1]
        self._max_idx += 1
        merged_cluster: Cluster = best_A.merge_clusters(best_B, new_idx=self._max_idx)
        # update clusters list
        self.clusters.remove(best_A)
        self.clusters.remove(best_B)
        self.clusters.append(merged_cluster)
        # update new cluster heap
        merged_cluster.init_heap(
            clusters=self.clusters,
            points_links=self._points_links,
            approx_fn=self.approx_fn,
            theta=self.theta,
        )
        # update neighbours heaps
        self._update_qs_from_merged(cluster_removed_A=best_A, cluster_removed_B=best_B)
        # update global heap Q
        self._reset_Q()

    def _remove_outliers(self) -> None:
        """Removes all the clusters with size 1, next reinitializes the heaps and the global heap Q."""
        print("Removing outliers...")
        len_before = len(self.clusters)
        self.clusters = [cluster for cluster in self.clusters if cluster.size() > 1]
        self._reset_heaps()
        print(f"Removed {len_before - len(self.clusters)} outliers.")

    def run(self):
        print("Running Rock algorithm...")
        max_joins = self.n_clusters_start - self.k
        pbar = tqdm(
            range(max_joins),
            desc="Running Rock",
        )
        for _ in pbar:
            pbar.set_postfix({"Clusters": len(self.clusters)})
            if len(self.clusters) <= self.k:
                print(f"Reached k={self.k} clusters.")
                break
            if len(self._Q) == 0:
                print("No more clusters to merge.")
                break

            self._run_iter()
            self._n_iterations += 1

            if not self._outliers_removed and (len(self.clusters) / self.n_clusters_start) <= self.outliers_factor:
                self._remove_outliers()
                self._outliers_removed = True

        print(f"Rock algorithm finished after {self._n_iterations} iterations.")
