from unittest import TestCase
from bnetdiff.datasets import (create_mv_normal,
                                         create_simple_random,
                                         create_correlation_data,
                                         generate_fc_matrices)
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

    def test_generate_fc_matrices_def_mask(self):
        N = 100  # Number of ROIs
        effect_size = 0.2  # Magnitude of group differences

        group1, group2, (cov1, cov2) = generate_fc_matrices(N, effect_size,  n_samples_group1=50,
                                                            n_samples_group2=70)
        plt.subplot(131); plt.imshow(cov1); plt.title("Group1 \n covariance matrix")
        plt.subplot(132);
        plt.imshow(cov2 - cov1); plt.title("Theoretical difference \n covariance matrix")
        diff = group2.mean(axis=0) - group1.mean(axis=0)
        plt.subplot(133);
        plt.imshow(diff); plt.title("Empirical group \n difference matrix")
        plt.show()

        self.assertEqual(group1.shape, (50, N, N))
        self.assertTrue(np.allclose(cov2, cov2.T))


    def test_create_simple_random(self):
        arr = create_simple_random(15, 10, 5, mean=2, sigma=1)
        self.assertEqual(arr.shape, (15, 10, 5))

    def test_create_mv_normal(self):
        X_list, (cov, prec) = create_mv_normal(2, n_samples=160,
                                               n_features=15,
                                               alpha=0.9,
                                               smallest_coef=0.2,
                                               largest_coef=0.8)

        plt.figure(figsize=(10, 6))
        plt.subplot(131);
        plt.imshow(cov);
        plt.subplot(132);
        plt.imshow(np.dot(X_list[0].T, X_list[0]) / 160);
        plt.subplot(133);
        plt.imshow(np.dot(X_list[1].T, X_list[1]) / 160);

        plt.show()
        self.fail()

    def test_create_correlation_data(self):
        y, arr, (dim1idxs, dim2idxs) = create_correlation_data(54, 20, 8)
        corrs = [np.corrcoef(y, arr[:, dim1idxs[i], dim2idxs[i]])[0][1]
                 for i in range(len(dim2idxs))]
        corrs_all_0 = [np.corrcoef(y, arr[:, i, 0])[0][1]
                    for i in range(arr.shape[1])]

        self.assertTrue(np.mean(corrs) > np.mean(corrs_all_0))
