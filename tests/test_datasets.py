from unittest import TestCase
from tfnbs.datasets import (generate_fc_matrices)
import matplotlib.pyplot as plt
import numpy as np


class Test(TestCase):

    def test_generate_fc_matrices(self):
        N = 30  # Number of ROIs
        effect_size = 0.2  # Magnitude of group differences
        mask = np.zeros((N, N))
        mask[0:10, 0:10] = 1  # Introduce differences in a subnetwork
        mask[10:20, 10:20] = -1  # Ensure symmetry

        group1, group2, (cov1, cov2) = generate_fc_matrices(N, effect_size,  mask, n_samples_group1=50, n_samples_group2=70)
        plt.subplot(141); plt.imshow(mask);
        plt.subplot(142); plt.imshow(cov1);
        plt.subplot(143); plt.imshow(cov2-cov1);
        diff = group2.mean(axis=0)-group1.mean(axis=0)
        plt.subplot(144); plt.imshow(diff);


        plt.show()

        self.assertEqual(group1.shape, (50, N, N))