import unittest

from rock.metrics import ClusteringPoint, is_jaccard_similar, jaccard, jaccard_distance, purity, silhouette
import numpy as np

DELTA = 1e-5

class TestJaccardFunctions(unittest.TestCase):
    def test_jaccard_empty_sets(self):
        self.assertAlmostEqual(jaccard(set(), set()), 0.0, delta=DELTA)

    def test_jaccard_one_empty_one_non_empty(self):
        self.assertAlmostEqual(jaccard(set(), {"a"}), 0.0, delta=DELTA)

    def test_jaccard_identical_sets(self):
        self.assertAlmostEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0, delta=DELTA)

    def test_jaccard_completely_different_sets(self):
        self.assertAlmostEqual(jaccard({"a"}, {"b"}), 0.0)

    def test_jaccard_sets_with_common_elements(self):
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3, delta=DELTA)

    def test_jaccard_distance_identical_sets(self):
        self.assertAlmostEqual(jaccard_distance({"a", "b"}, {"a", "b"}), 0.0, delta=1e-5)

    def test_jaccard_distance_completely_different_sets(self):
        self.assertAlmostEqual(jaccard_distance({"a"}, {"b"}), 1.0)

    def test_is_jaccard_similar_below_threshold(self):
        self.assertFalse(is_jaccard_similar({"a"}, {"b", "c"}, theta_threshold=0.5))

    def test_is_jaccard_similar_above_threshold(self):
        self.assertTrue(is_jaccard_similar({"a", "b"}, {"b", "c"}, theta_threshold=0.33))


class TestPurityFunction(unittest.TestCase):
    def test_purity_empty_points(self):
        self.assertEqual(purity([]), 0.0)

    def test_purity_all_outliers(self):
        points = [ClusteringPoint(original_idx=1, idx=1, transaction={}, output_cluster_idx=None, y="a")]
        self.assertEqual(purity(points, skip_outliers=True), 0.0)

    def test_purity_no_outliers_single_cluster(self):
        points = [ClusteringPoint(original_idx=i, idx=i, transaction={}, output_cluster_idx=1, y="a") for i in range(5)]
        self.assertAlmostEqual(purity(points), 1.0, delta=DELTA)

    def test_purity_multiple_clusters_even_distribution(self):
        points = [ClusteringPoint(original_idx=i, idx=i, transaction={}, output_cluster_idx=i % 2, y="a") for i in range(10)]
        self.assertAlmostEqual(purity(points), 1.0, delta=DELTA)

    def test_purity_uneven_cluster_distribution(self):
        points = [ClusteringPoint(idx=1, original_idx=1, transaction={}, output_cluster_idx=1, y="a")] * 5
        points += [ClusteringPoint(idx=2, original_idx=2, transaction={}, output_cluster_idx=2, y="b")] * 3
        points += [ClusteringPoint(idx=3, original_idx=3, transaction={}, output_cluster_idx=2, y="c")] * 2
        self.assertAlmostEqual(purity(points), (3+5) / 10, delta=DELTA)


class TestSilhouetteFunction(unittest.TestCase):
    def test_silhouette_single_cluster_single_point(self):
        points = [ClusteringPoint(idx=1, original_idx=1, transaction={"a"}, output_cluster_idx=1, y="a")]
        self.assertTrue(np.isnan(silhouette(points)))

    def test_silhouette_multiple_clusters_multiple_points(self):
        points = [
            ClusteringPoint(idx=1, original_idx=1, transaction={"a", "b"}, output_cluster_idx=1, y="a"),
            ClusteringPoint(idx=2, original_idx=2, transaction={"a"}, output_cluster_idx=1, y="b"),
            ClusteringPoint(idx=3, original_idx=3, transaction={"c"}, output_cluster_idx=2, y="c"),
            ClusteringPoint(idx=4, original_idx=4, transaction={"d", "e"}, output_cluster_idx=2, y="d"),
        ]
        self.assertNotEqual(silhouette(points), 0.0)

    def test_silhouette_uneven_cluster_sizes(self):
        points = [ClusteringPoint(idx=1, original_idx=1, transaction={"a", "b"}, output_cluster_idx=1, y="a")] * 3
        points += [ClusteringPoint(idx=2, original_idx=2, transaction={"c"}, output_cluster_idx=2, y="c")]
        self.assertNotEqual(silhouette(points), 0.0)
