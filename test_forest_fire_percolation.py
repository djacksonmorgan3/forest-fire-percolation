import unittest
import numpy as np
from forest_fire_percolation import (
    Forest, make_densities, wilson_interval, estimate_empirical_threshold,
    evaluate_unseen_densities
)


class ForestSearchTests(unittest.TestCase):
    def test_path_reaches_bottom(self):
        forest = Forest.from_grid([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
        self.assertTrue(forest.depth_first_search())
        self.assertTrue(forest.breadth_first_search())

    def test_no_path_reaches_bottom(self):
        forest = Forest.from_grid([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        self.assertFalse(forest.depth_first_search())
        self.assertFalse(forest.breadth_first_search())

    def test_empty_top_row(self):
        forest = Forest.from_grid([[0, 0], [1, 1]])
        self.assertFalse(forest.depth_first_search())
        self.assertFalse(forest.breadth_first_search())

    def test_density_range(self):
        densities = make_densities(0.25)
        self.assertEqual(len(densities), 5)
        self.assertAlmostEqual(densities[0], 0)
        self.assertAlmostEqual(densities[-1], 1)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(70, 100)
        self.assertLess(low, 0.70)
        self.assertGreater(high, 0.70)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_empirical_threshold_interpolates_crossing(self):
        records = [
            {"density": 0.4, "dfs_probability": 0.2},
            {"density": 0.6, "dfs_probability": 0.8},
        ]
        self.assertAlmostEqual(estimate_empirical_threshold(records), 0.5)


    def test_unseen_density_evaluation_holds_out_levels(self):
        rows = []
        for density in np.linspace(0, 1, 11):
            for outcome in ([0] * 8 + [1] * 2 if density < 0.5 else [0] * 2 + [1] * 8):
                rows.append((float(density), outcome))
        metrics = evaluate_unseen_densities(rows, seed=42, holdout_every=3)
        self.assertGreater(metrics["train_rows"], 0)
        self.assertGreater(metrics["test_rows"], 0)
        self.assertGreater(metrics["held_out_density_levels"], 0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["baseline_accuracy"], 0.0)
        self.assertLessEqual(metrics["baseline_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["baseline_brier_score"], 0.0)
        self.assertLessEqual(metrics["baseline_brier_score"], 1.0)
        self.assertAlmostEqual(metrics["baseline_roc_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
