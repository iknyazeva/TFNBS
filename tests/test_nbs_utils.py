import unittest
import numpy as np
from tfnbs.nbs_utils import nbs_bct
from tfnbs.utils import fisher_r_to_z
from tfnbs.datasets import generate_fc_matrices


class TestNBSUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.effect_size = 0.5
        cls.n_samples_g1 = 20
        cls.n_samples_g2 = 20
        cls.n_nodes = 10
        cls.seed = 2

        g1, g2, (cov1, cov2) = generate_fc_matrices(cls.n_nodes,
                                                    cls.effect_size,
                                                    n_samples_group1=cls.n_samples_g1,
                                                    n_samples_group2=cls.n_samples_g2,
                                                    seed=cls.seed)
        cls.group1 = fisher_r_to_z(g1)
        cls.group2 = fisher_r_to_z(g2)

    def test_output_shapes_and_keys(self):
        """Check output structure and shape of nbs_bct"""
        p_vals, adj, null = nbs_bct(self.group1, self.group2,
                                    threshold=2.1,
                                    n_permutations=100,
                                    paired=False,
                                    random_state=2,
                                    use_mp=False)

        # Check keys
        self.assertIn("g1>g2", p_vals)
        self.assertIn("g2>g1", p_vals)
        self.assertIn("g1>g2", adj)
        self.assertIn("g2>g1", adj)
        self.assertIn("g1>g2", null)
        self.assertIn("g2>g1", null)

        # Check array shapes
        N = self.group1.shape[1]
        self.assertEqual(p_vals["g1>g2"].shape, (N, N))
        self.assertEqual(adj["g1>g2"].shape, (N, N))
        self.assertTrue(null["g1>g2"].ndim in [1, 2])

    def test_symmetry_of_adjacency(self):
        """Adjacency matrix should be symmetric."""
        _, adj, _ = nbs_bct(self.group1, self.group2,
                            threshold=2.0,
                            n_permutations=50,
                            paired=False,
                            random_state=123,
                            use_mp=False)
        for key in adj:
            np.testing.assert_array_equal(adj[key], adj[key].T)

    #def test_invalid_input_shape(self):
    #    """Expect error if group matrices are mismatched in shape."""
    #    g1 = self.group1.copy()
    #    g2 = self.group2[:-1]  # Invalid: one fewer sample
    #    with self.assertRaises(ValueError):
    #        nbs_bct(g1, g2, threshold=2.0, paired=False)

    def test_paired_behavior(self):
        """Check that paired=True yields valid output for matched subjects."""
        g1 = self.group1[:15]
        g2 = self.group2[:15]
        p_vals, _, _ = nbs_bct(g1, g2,
                               threshold=1.5,
                               n_permutations=10,
                               paired=True,
                               use_mp=False)
        self.assertTrue(isinstance(p_vals, dict))
        self.assertIn("g1>g2", p_vals)


if __name__ == '__main__':
    unittest.main()
