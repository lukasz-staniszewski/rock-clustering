import unittest
from unittest.mock import patch

import numpy as np

from rock.cluster import (
    Cluster,
    compute_points_links,
    get_clusters_links,
    get_expected_cluster_size_penalty,
    get_goodness_measure,
    get_point_neighbors,
)


class MockClusteringPoint:
    def __init__(self, idx, transaction=None):
        self.idx = idx
        self.original_idx = idx
        self.transaction = transaction or set()
        self.y = None
        self.output_cluster_idx = None


def mock_get_goodness_measure(cluster_A, cluster_B, points_links, approx_fn, theta, eps=1e-6):
    return 1.0  # Simple mock returning a constant goodness measure


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.points = [MockClusteringPoint(idx) for idx in range(5)]
        self.cluster = Cluster(idx=1, points=self.points)

    def test_initialization(self):
        self.assertEqual(self.cluster.idx, 1)
        self.assertEqual(len(self.cluster.points), 5)

    def test_size_method(self):
        self.assertEqual(self.cluster.size(), 5)

    def test_equality_operator(self):
        other_cluster = Cluster(idx=1, points=self.points)
        self.assertEqual(self.cluster, other_cluster)

    def test_less_than_operator(self):
        other_cluster = Cluster(idx=2, points=self.points)
        self.assertLess(self.cluster, other_cluster)


class TestInitHeap(unittest.TestCase):
    def setUp(self):
        self.points = [MockClusteringPoint(idx) for idx in range(5)]
        self.cluster = Cluster(idx=1, points=self.points)
        self.other_clusters = [Cluster(idx=i, points=self.points) for i in range(2, 6)]

    @patch("rock.cluster.get_goodness_measure", side_effect=mock_get_goodness_measure)
    def test_heap_initialization(self, mock_goodness):
        points_links = [[0] * 5 for _ in range(5)]
        approx_fn = lambda x: x
        theta = 0.5
        self.cluster.init_heap(self.other_clusters, points_links, approx_fn, theta)

        self.assertIsNotNone(self.cluster.heap)
        self.assertEqual(len(self.cluster.heap), len(self.other_clusters))

    def test_heap_content(self):
        points_links = [[3] * 5 for _ in range(5)]
        approx_fn = lambda x: x
        theta = 0.5
        self.cluster.init_heap(self.other_clusters, points_links, approx_fn, theta)

        for item in self.cluster.heap:
            self.assertIsInstance(item[1], Cluster)

        for cluster in self.other_clusters:
            # each should have other clusters in the heap
            self.assertIn(cluster, [item[1] for item in self.cluster.heap])

    def test_heap_order(self):
        points_links = [[0] * 5 for _ in range(5)]
        approx_fn = lambda x: x
        theta = 0.5
        self.cluster.init_heap(self.other_clusters, points_links, approx_fn, theta)

        # Check if the heap is ordered in descending order of goodness measure
        for i in range(len(self.cluster.heap) - 1):
            self.assertGreaterEqual(
                get_goodness_measure(
                    self.cluster.heap[i][1], self.cluster.heap[i + 1][1], points_links, approx_fn, theta
                ),
                0.0,
            )


class TestMergeClusters(unittest.TestCase):
    def setUp(self):
        self.points_cluster_a = [MockClusteringPoint(idx) for idx in range(5)]
        self.points_cluster_b = [MockClusteringPoint(idx + 5) for idx in range(5)]
        self.cluster_a = Cluster(idx=1, points=self.points_cluster_a)
        self.cluster_b = Cluster(idx=2, points=self.points_cluster_b)

    def test_cluster_merging(self):
        new_idx = 3
        merged_cluster = self.cluster_a.merge_clusters(self.cluster_b, new_idx)

        self.assertEqual(merged_cluster.idx, new_idx)
        self.assertEqual(len(merged_cluster.points), len(self.points_cluster_a) + len(self.points_cluster_b))
        self.assertIn(self.points_cluster_a[0], merged_cluster.points)
        self.assertIn(self.points_cluster_b[0], merged_cluster.points)


class TestGetPointNeighbors(unittest.TestCase):
    def setUp(self):
        self.points = [MockClusteringPoint(idx, transaction={idx}) for idx in range(10)]

    def mock_is_jaccard_similar(self, transaction_a, transaction_b, theta_threshold):
        return transaction_a != transaction_b

    def test_neighbor_identification(self):
        with patch("rock.cluster.is_jaccard_similar", side_effect=self.mock_is_jaccard_similar):
            neighbors = get_point_neighbors(self.points[0], self.points, 0.5)
            self.assertEqual(len(neighbors), len(self.points) - 1)

    def test_no_self_neighbor(self):
        with patch("rock.cluster.is_jaccard_similar", side_effect=self.mock_is_jaccard_similar):
            neighbors = get_point_neighbors(self.points[0], self.points, 0.5)
            self.assertNotIn(self.points[0], neighbors)


class TestComputePointsLinks(unittest.TestCase):
    def setUp(self):
        self.points = [MockClusteringPoint(idx, transaction={idx}) for idx in range(10)]

    def test_links_matrix_initialization(self):
        theta = 0.5
        links = compute_points_links(self.points, theta)

        self.assertIsInstance(links, np.ndarray)
        self.assertEqual(links.shape, (len(self.points), len(self.points)))

    def test_correct_link_count(self):
        def mock_is_jaccard_similar(transaction_a, transaction_b, theta_threshold):
            # there is a link between two points if their transactions differ by 1
            return abs(int(list(transaction_a)[0]) - int(list(transaction_b)[0])) == 1

        with unittest.mock.patch("rock.cluster.is_jaccard_similar", side_effect=mock_is_jaccard_similar):
            theta = 0.5
            links = compute_points_links(self.points, theta)
            print(links)

            # there will be link between point 0 and 2, 1 and 3, etc.
            self.assertEqual(links[0, 2], 1)
            self.assertEqual(links[2, 0], 1)
            self.assertEqual(links[1, 2], 0)
            self.assertEqual(links[2, 1], 0)
            self.assertEqual(links[1, 0], 0)


class TestGetClustersLinks(unittest.TestCase):
    def setUp(self):
        points_cluster_a = [MockClusteringPoint(idx, transaction={}) for idx in range(5)]
        points_cluster_b = [MockClusteringPoint(idx, transaction={}) for idx in range(5, 10)]
        self.cluster_a = Cluster(idx=1, points=points_cluster_a)
        self.cluster_b = Cluster(idx=2, points=points_cluster_b)

    def test_link_summation(self):
        points_links = [[0] * 10 for _ in range(10)]
        for i in range(5):
            for j in range(5, 10):
                points_links[i][j] = 1
                points_links[j][i] = 1
        links_sum = get_clusters_links(self.cluster_a, self.cluster_b, points_links)
        self.assertEqual(links_sum, 25)

    def test_link_computation(self):
        points_links = [[0 if i != j else 1 for j in range(10)] for i in range(10)]
        total_links = get_clusters_links(self.cluster_a, self.cluster_b, points_links)
        self.assertEqual(total_links, 0)


class TestGetExpectedClusterSizePenalty(unittest.TestCase):
    def setUp(self):
        self.cluster = Cluster(idx=1, points=[MockClusteringPoint(idx, transaction={}) for idx in range(10)])

    def test_penalty_calculation(self):
        approx_fn = lambda x: x**2
        theta = 0.5
        penalty = get_expected_cluster_size_penalty(self.cluster, approx_fn, theta)
        expected_penalty = 10 ** (1 + 2 * approx_fn(theta))
        self.assertEqual(penalty, expected_penalty)


class TestGetGoodnessMeasure(unittest.TestCase):
    def setUp(self):
        points_cluster_a = [MockClusteringPoint(idx, transaction={}) for idx in range(5)]
        points_cluster_b = [MockClusteringPoint(idx, transaction={}) for idx in range(5, 10)]
        self.cluster_a = Cluster(idx=1, points=points_cluster_a)
        self.cluster_b = Cluster(idx=2, points=points_cluster_b)

        # Mock points links matrix
        self.points_links = [[0] * 10 for _ in range(10)]
        for i in range(5):
            for j in range(5, 10):
                self.points_links[i][j] = 1
                self.points_links[j][i] = 1

    def test_measure_calculation(self):
        approx_fn = lambda x: x**2
        theta = 0.5
        goodness_measure = get_goodness_measure(self.cluster_a, self.cluster_b, self.points_links, approx_fn, theta)
        self.assertIsInstance(goodness_measure, float)

