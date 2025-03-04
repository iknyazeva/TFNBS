from unittest import TestCase
from bnetdiff.datasets import (create_mv_normal,
                                         create_simple_random,
                                         create_correlation_data)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  #
import numpy as np


class Test(TestCase):
    def test_create_simple_random(self):
        arr = create_simple_random(15, (10, 5),  mean=2, sigma=1)
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
