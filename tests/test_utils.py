from unittest import TestCase
from network_statistics.utils import binarize, get_components
from network_statistics.datasets import generate_fc_matrices
from network_statistics.utils import fisher_r_to_z
from network_statistics.pairwise_tfns import compute_t_stat
import numpy as np


class Test(TestCase):

    @classmethod
    def setUpClass(cls):
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

    def test_get_components(self):
        t_stats = self.fc_sim_30["t_stat"]

        A = t_stats >= 1.47
        comps, comp_sizes = get_components(A)
        self.assertTrue(True)
