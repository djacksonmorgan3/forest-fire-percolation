import unittest
from forest_fire_percolation import Forest, make_densities


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


if __name__ == "__main__":
    unittest.main()
