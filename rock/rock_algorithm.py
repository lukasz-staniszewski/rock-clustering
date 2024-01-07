from copy import deepcopy
from heapq import heapify, heappop, heappush
from typing import Callable, List, Tuple

from process_data import RockInput
from tqdm import tqdm

from rock.cluster import Cluster, ClusteringPoint, Heap, compute_points_links, get_expected_cluster_size_penalty
from rock.metrics import is_jaccard_similar


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
        self.dataset_train = dataset.data_train
        self.dataset_test = dataset.data_test
        self.k = k
        self.theta = theta
        self.approx_fn = approx_fn
        self.outliers_factor = outliers_factor
        self._points_links = compute_points_links(points=self.dataset_train, theta=theta)
        self._n_iterations = 0
        self._were_outliers_removed = False
        self._outliers_removed = []
        self._max_idx = -1
        self._init_clusters()
        print("Rock algorithm initialized.")

    def _init_clusters(self) -> None:
        """Initialize all the clusters representing single points, creates local heaps for each cluster and global heap Q."""
        for point in self.dataset_train:
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

    def _predict_best_cluster(self, clustering_point: ClusteringPoint) -> Cluster:
        """Predicts the best cluster for the given point."""
        best_cluster = None
        best_cluster_score = -1
        for cluster in self.clusters:
            size_penalty = get_expected_cluster_size_penalty(
                cluster=cluster,
                approx_fn=self.approx_fn,
                theta=self.theta,
            )
            n_neighbors = sum(
                [
                    int(
                        is_jaccard_similar(
                            a=clustering_point.x,
                            b=neighbour.x,
                            theta_threshold=self.theta,
                        )
                    )
                    for neighbour in cluster.points
                ]
            )
            score = n_neighbors / size_penalty
            if score > best_cluster_score:
                best_cluster_score = score
                best_cluster = cluster

        return best_cluster

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
        indices_before = set([cluster.idx for cluster in self.clusters])
        self.clusters = [cluster for cluster in self.clusters if cluster.size() > 1]
        self._reset_heaps()
        indices_removed = indices_before - set([cluster.idx for cluster in self.clusters])
        self._outliers_removed = [
            clustering_point for clustering_point in self.dataset_train if clustering_point.idx in indices_removed
        ]
        print(f"Removed {len_before - len(self.clusters)} outliers.")

    def create_clusters(self):
        """Performs first phase of the Rock algorithm, which is creating clusters (based on the fragment of database)."""
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

            if (
                not self._were_outliers_removed
                and (len(self.clusters) / self.n_clusters_start) <= self.outliers_factor
            ):
                self._remove_outliers()
                self._were_outliers_removed = True

        print(f"Rock algorithm finished after {self._n_iterations} iterations.")

    def collect_results(self, skip_outliers: bool = False) -> List[ClusteringPoint]:
        """Collects the results of the algorithm - assigns created clusters to each point."""
        print("Collecting results...")
        results = []
        for cluster in self.clusters:
            for point in cluster.points:
                point.output_cluster_idx = cluster.idx
                results.append(point)
        for point in self._outliers_removed:
            predicted_cluster = self._predict_best_cluster(clustering_point=point)
            point.output_cluster_idx = predicted_cluster.idx if not skip_outliers else None
            results.append(point)
        if self.dataset_test is not None:
            for point in self.dataset_test:
                predicted_cluster = self._predict_best_cluster(clustering_point=point)
                point.output_cluster_idx = predicted_cluster.idx
                results.append(point)
        print("Results collected.")
        return results
