import unittest
from unittest import TestCase
import numpy as np
from tfnbs.pairwise_tfns import compute_t_stat, compute_t_stat_diff
from tfnbs.tfnos import get_tfce_score, get_tfce_score_scipy
from tfnbs.datasets import generate_fc_matrices
from tfnbs.utils import fisher_r_to_z
from tfnbs.pairwise_tfns import compute_t_stat
import time


class TestTFNOS(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.small_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        cls.invalid_matrix = np.array([[1, 1, 2], [1, 0, 1], [2, 1, 0]])
        effect_size = 0.2
        group1, group2, (cov1, cov2) = generate_fc_matrices(30,
                                                            effect_size,
                                                            n_samples_group1=30,
                                                            n_samples_group2=20,
                                                            seed=42)
        t_stat_30 = compute_t_stat(fisher_r_to_z(group1),
                                   fisher_r_to_z(group2), paired=False)

        cls.fc_sim_30 = {"t_stat": t_stat_30,
                         "cov1": cov1.copy(), "cov2": cov2.copy()}

        group1, group2, (cov1, cov2) = generate_fc_matrices(100,
                                                            effect_size,
                                                            n_samples_group1=50,
                                                            n_samples_group2=40,
                                                            seed=42)
        t_stat_100 = compute_t_stat(fisher_r_to_z(group1),
                                    fisher_r_to_z(group2), paired=False)

        cls.fc_sim_100 = {"t_stat": t_stat_100,
                          "cov1": cov1, "cov2": cov2}

    def setUp(self):
        self.E = 0.4
        self.H = 3
        self.n = 10

    def run_and_measure(self, func, matrix):
        """Helper function to measure execution time of a function."""
        start_time = time.time()
        func(matrix, self.E, self.H, self.n)
        return time.time() - start_time

    def test_small_matrix(self):
        """
        Test with a small 3x3 matrix and known output.
        """
        statsmat = self.small_matrix
        result_pos = get_tfce_score(statsmat, 1, 1, 2, start_thres=0)
        result_neg = get_tfce_score(-statsmat, 1, 1, 2, start_thres=0)

        # Expected output (manually calculated)
        expected_pos = np.array([[0.0, 3.0, 5.0], [3.0, 0.0, 3.0], [5.0, 3.0, 0.0]])
        expected_neg = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-5)
        np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-5)

    def test_small_matrix_scipy(self):
        """
        Test with a small 3x3 matrix and known output.
        """
        statsmat = self.small_matrix
        result_nx = get_tfce_score(statsmat, 1, 1, 2, start_thres=0)
        result = get_tfce_score_scipy(statsmat, 1, 1, 2, start_thres=0)

        # Expected output (manually calculated)
        expected = np.array([[0.0, 3.0, 5.0], [3.0, 0.0, 3.0], [5.0, 3.0, 0.0]])
        np.testing.assert_allclose(result_nx, result, rtol=1e-5)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_small_matrix_diff_pars(self):
        """
        Test with a small 3x3 matrix and known output.
        """
        statsmat = self.small_matrix
        result_nx = get_tfce_score(statsmat, 0.4, 4, 100, start_thres=0)
        result_scipy = get_tfce_score_scipy(statsmat, 0.4, 4, 100, start_thres=0)

        # Expected output (manually calculated)
        expected = np.array([[0.0, 0.326, 6.677], [0.326, 0.0, 0.326], [6.677, 0.326, 0.0]])
        np.testing.assert_allclose(result_nx, expected, rtol=1e-2)
        np.testing.assert_allclose(result_scipy, expected, rtol=1e-2)

    def test_stat_symmetry(self):
        statsmat = self.small_matrix
        result = get_tfce_score(statsmat, 1, 1, 2)
        self.assertTrue(np.allclose(result, result.T))

    def test_input_no_self_connections(self):
        statsmat = self.invalid_matrix
        with self.assertRaises(ValueError):
            get_tfce_score(statsmat, 1, 1, 2)

    def test_tfnos_real_matrix_30N(self):
        t_stat = self.fc_sim_30["t_stat"]
        score_pos = get_tfce_score_scipy(t_stat['g2>g1'], self.E, self.H, self.n, start_thres=1.7)
        score_neg = get_tfce_score_scipy(t_stat['g1>g2'], self.E, self.H, self.n, start_thres=1.7)

        self.assertTrue((score_pos >= 0).all())
        self.assertTrue((score_neg >= 0).all())

    def test_time_consumption(self):
        time_original = self.run_and_measure(get_tfce_score, self.fc_sim_100["t_stat"]['g2>g1'])
        time_scipy = self.run_and_measure(get_tfce_score_scipy, self.fc_sim_100["t_stat"]['g2>g1'])

        self.assertLess(time_scipy, time_original)

    def test_scipy_list_params(self):
        statsmat = self.fc_sim_30["t_stat"]['g2>g1']
        result = get_tfce_score_scipy(statsmat, [0.4, 0.4], [1, 2], 10)
        result_nx = get_tfce_score(statsmat, [0.4, 0.4], [1, 2], 10)

        self.assertTrue(result.shape[2] == 2)
        self.assertTrue(result_nx.shape[2] == 2)


