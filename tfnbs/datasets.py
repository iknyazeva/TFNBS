import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix


def generate_fc_matrices(N,  effect_size, mask=None, n_samples_group1=50, n_samples_group2=50,
                         repeated_measures=False, seed=None):
    """
    Generate synthetic functional connectivity correlation matrices for groupwise comparisons
    or repeated measures.

    Parameters:
        N (int): Number of ROIs (regions of interest) i.e. an N*N matrix.
        effect_size (float): Magnitude of correlation difference between groups.
        mask (np.ndarray, optional): Binary mask (N*N) matrix to apply correlation differences.
        n_samples_group1 (int): Number of matrices in group 1 (default: 50).
        n_samples_group2 (int): Number of matrices in group 2 (default: 50).
        repeated_measures (bool): If True, generate within-subject repeated measures data
                                i.e. Paired else independently (default = False).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        group1 (np.ndarray): Array of with functional connectivity matrics of group1 as (n_samples_group1, N, N).
        group2 (np.ndarray): Array of with functional connectivity matrics of group2 as (n_samples_group2, N, N).
        (base_cov, mod_cov): Original Covariance matrix and Modified covariance matrix with effect_size introduced
                             for group1 and group2 matrices. 

    >>> N = 6; e = 0.2; mask = np.zeros((N, N))
    >>> mask[0:2, 0:2] = 1; mask[2:4, 2:4] = -1
    >>> g1, g2, (c1,c2) = generate_fc_matrices(N, e, mask, 5, 10, seed = 0)
    >>> g1.shape == (5, 6, 6)
    True
    >>> g2.shape == (10, 6, 6)
    True
    >>> np.allclose(c1,c1.T)
    True
    """

    if seed is not None:
        np.random.seed(seed)
    if mask is None:
        mask = np.zeros((N, N))
        N_pos_block = N//3
        mask[:N_pos_block, :N_pos_block] = 1
        N_neg_block = N//3
        mask[N_pos_block:N_pos_block+N_neg_block, N_pos_block:N_pos_block+N_neg_block] = -1

    # Generate base covariance matrix
    base_cov = make_sparse_spd_matrix(N, alpha=0.8, norm_diag=True, random_state=seed)

    # Create modified covariance for the second condition or group
    mod_cov = base_cov.copy()
    mod_cov[mask == 1] += effect_size + np.abs(np.random.normal(0,0.05, mod_cov[mask == 1].shape)) # Increase correlations in masked regions
    mod_cov[mask == -1] -= effect_size - np.abs(np.random.normal(0,0.1, mod_cov[mask == 1].shape)) # Decrease correlations in masked regions
    np.fill_diagonal(mod_cov, 1.0)
    mod_cov = (mod_cov+mod_cov.T)/2


    def enforce_spd(matrix, eps=1e-6):
        """ Ensures a matrix is symmetric positive definite (SPD) by adjusting eigenvalues. """
        eigvals, eigvecs = np.linalg.eigh(matrix)       # Get eigenvalues & eigenvectors
        eigvals[eigvals < eps] = eps                    # Replace negative eigenvalues with small positive value
        return eigvecs @ np.diag(eigvals) @ eigvecs.T   # Reconstruct SPD matrix


    # Enforce SPD property
    mod_cov = enforce_spd(mod_cov)


    # Generate sample correlation matrices with variability
    def generate_samples(cov_matrix, n_samples):
        return np.array([np.corrcoef(np.random.multivariate_normal(np.zeros(N), cov_matrix, size=N).T)
                         for _ in range(n_samples)])

    if repeated_measures:
        # Generate paired data (same subjects measured twice)
        group1 = generate_samples(base_cov, n_samples_group1)
        group2 = generate_samples(mod_cov, n_samples_group1)  # Same number as group1
    else:
        # Generate independent groups
        group1 = generate_samples(base_cov, n_samples_group1)
        group2 = generate_samples(mod_cov, n_samples_group2)

    return group1, group2, (base_cov, mod_cov)
