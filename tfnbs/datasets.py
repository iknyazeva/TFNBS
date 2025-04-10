from scipy import linalg
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix


def generate_fc_matrices(N,  effect_size, mask=None, n_samples_group1=50, n_samples_group2=50,
                         repeated_measures=False, seed=None):
    """
    Generate example functional connectivity correlation matrices for groupwise comparisons
    or repeated measures.

    Parameters:
    - N (int): Number of ROIs (regions of interest).
    - mask (np.ndarray): Binary mask (N x N) indicating where differences should be applied.
    - effect_size (float): Magnitude of the difference applied in the masked regions.
    - n_samples_group1 (int): Number of matrices in group 1 (default: 50).
    - n_samples_group2 (int): Number of matrices in group 2 (default: 50).
    - repeated_measures (bool): If True, generate within-subject repeated measures data.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - group1 (np.ndarray): (n_samples_group1, N, N) correlation matrices for group 1.
    - group2 (np.ndarray): (n_samples_group2, N, N) correlation matrices for group 2.

    
    >>> import numpy as np   
    >>> N = 6; e = 0.2; mask = np.zeros((N, N))
    >>> mask[0:2, 0:2] = 1; mask[2:4, 2:4] = -1
    >>> g1, g2, (c1,c2) = generate_fc_matrices(N, e, mask, 5, 10, seed = 0)
    >>> g1.shape
    (5, 6, 6)
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
    mod_cov[mask == -1] -= effect_size - np.abs(np.random.normal(0,0.1, mod_cov[mask == 1].shape))  # Decrease correlations in masked regions
    np.fill_diagonal(mod_cov, 1.0)
    mod_cov = (mod_cov+mod_cov.T)/2


    def enforce_spd(matrix, eps=1e-6):
        """ Ensures a matrix is symmetric positive definite (SPD) by adjusting eigenvalues. """
        eigvals, eigvecs = np.linalg.eigh(matrix)  # Get eigenvalues & eigenvectors
        eigvals[eigvals < eps] = eps  # Replace negative eigenvalues with small positive value
        return eigvecs @ np.diag(eigvals) @ eigvecs.T  # Reconstruct SPD matrix


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


def create_simple_random(n_rep: int,
                         dims: tuple[int],
                         mean: float = 3.,
                         sigma: float = 1.):
    """

    Args:
        n_rep: Number of repeated measures
        dim1: ROI dimension M
        dim2: ROI dimension N
        mean: peak mean values 
        sigma: standard deviation 

    Returns: 
        arr: array of shape n_rep*M*N

    >>> arr = create_simple_random(15, [10, 5], mean=3, sigma=1.5)
    >>> arr.shape
    (15, 10, 5)
    

    """
    arr = np.array([np.random.normal(mean, sigma,
                                     size=dims) for _ in range(n_rep)])
    return arr

# Repeated function? 
def create_nd_random_arr(n_rep: int,
                         dims: tuple,
                         mean: float = 3.,
                         sigma: float = 1.):
    arr = np.array([np.random.normal(mean, sigma,
                                     size=dims) for _ in range(n_rep)])
    return arr


def create_mv_normal(n_rep: int,
                     n_samples: int = 100,
                     n_features: int = 20,
                     alpha: float = 0.9,
                     smallest_coef: float = 0.4,
                     largest_coef: float = 0.7):
    """ Create predefined number of multivariate time series with the same covariance

    Args:
        n_samples:
        n_features:
        alpha:
        smallest_coef:
        largest_coef:

    Returns:
        Time series records: with n_rep*n_samples*n_features
        covariance matrix
        prec?

    
            
    """

    prec = make_sparse_spd_matrix(
        n_features, alpha=alpha, smallest_coef=smallest_coef, largest_coef=largest_coef
    )
    cov = linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]

    X_list = []
    for _ in range(n_rep):
        X = np.random.multivariate_normal(np.zeros(n_features),
                                          cov, size=n_samples)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        X_list.append(X)

    return X_list, (cov, prec)


def create_correlation_data(n_rep: int,
                            dim1: int,
                            dim2: int,
                            mean: float = 3.,
                            sigma: float = 1.):
    """ Create functional connectivity matrices and add correlation between themn
    
    Args:
        n_rep: Number of repeated measures
        dim1: ROI dimension M
        dim2: ROI dimension N
        mean: peak mean values 
        sigma: standard deviation 

    Returns: 
        y: Target vector 
        arr: subject matrices of shape n_rep*M*N 
        dim1idxs: indexs with correlatioon 
        dim2idxs: indexs with correlatioon 
    >>> y, arr, (dim1idxs, dim2idxs) = create_correlation_data(10, 20, 5)
    >>> corrs = [np.corrcoef(y, arr[:, dim1idxs[i], dim2idxs[i]])[0][1] for i in range(len(dim2idxs))]
    >>> corrs_all_0 = [np.corrcoef(y, arr[:, i, 0])[0][1] for i in range(arr.shape[1])]
    >>> np.mean(corrs) > np.mean(corrs_all_0)
    True
        
    """
    arr = np.array([np.random.normal(mean, sigma,
                                     size=(dim1, dim2)) for _ in range(n_rep)])
    dim1idxs = np.random.randint(dim1, size=max(1, dim1 // 5))
    dim2idxs = np.random.randint(dim2, size=len(dim1idxs))
    y = np.random.normal(0, 1, size=n_rep)
    for idx1, idx2 in zip(dim1idxs, dim2idxs):
        y += np.random.uniform(0.6, 0.9) * arr[:, idx1, idx2]


    return y, arr, (dim1idxs, dim2idxs)
