import numpy as np
import numpy.typing as npt
from .pairwise_tfns import *
from .utils import *

def nbs_bct(group1: npt.NDArray[np.float64],
            group2: npt.NDArray[np.float64],
            threshold: int = 2.0,
            n_permutations: int = 100,
            paired: bool = True,
            use_mp: bool = True,
            random_state: Optional[int] = None,
            n_processes: Optional[int] = None,
            **kwargs):
    
    """
    Function to compute Network-Based Statistics (NBS) between two individual or independent groups of connectivity matrices using a fixed t-statistic threshold.    
    
    Parameters:
        group1 (np.ndarray): Group 1 matrices as an array of shape (K, N, N) with k subjects with N ROIs.
        group2 (np.ndarray): Group 2 matrices as an array of shape (K, N, N) with k subjects with N ROIs.
        threshold (int): Threshold for t-statistics to define edges included in clusters.
        n_permutations (int): Number of permutations for null distribution.
        paired (bool): Perform paried T-Test (default = True) else if False, individual test 
        use_mp (bool): Use multiple cores for computation (default = True)
        random_state (int, optional): Random state seed.
        n_processes (int, optional): Number of processor cores for parallel computation.

    Returns: 
        p_values (dict[np.ndarray]): Computed p-values for given groups in two conditions 'g1>g2' and 'g2>g1' indexed as dictionary keys.
        adj_matrices (np.ndarray): Adjoint matrix of N*N dimensions representing boolean states greater than threshold value.
        max_null_dict (dict): Dictionary with null distributions used to compute p-values.

    Note: The output adjoint matrix consists of both comparisons comprise of tails explicitly given as g1>g2 & g2>g1
    
    >>> g1 = np.random.rand(10, 5, 5)
    >>> g2 = np.random.rand(10, 5, 5)
    >>> p_vals, adjs, nulls = nbs_bct(g1, g2, threshold=1.5, n_permutations=100, paired=True, use_mp=False, random_state=0) 
    >>> adjs['g1>g2'].shape == (5, 5)
    True

    """

    if paired:
        t_func = compute_t_stat_diff
        emp_t_dict = t_func(compute_diffs(group1, group2))
    else:
        t_func = compute_t_stat
        emp_t_dict = t_func(group1, group2, paired=False, **kwargs)

    adj_matrices = {}
    for key in emp_t_dict:
        emp_t = emp_t_dict[key]
        adj = (emp_t > threshold).astype(np.uint8)
        if adj.shape[-1] == adj.shape[-2]:
            adj = np.triu(adj, 1)
            adj = adj + adj.T
        adj_matrices[key] = adj

    max_null_dict = compute_null_dist(group1,
                                      group2,
                                      t_func,
                                      n_permutations=n_permutations,
                                      paired=paired,
                                      use_mp=use_mp,
                                      random_state=random_state,
                                      n_processes=n_processes)

    keys = list(emp_t_dict.keys())
    p_values = dict()
    if len(emp_t_dict[keys[0]].shape) == 2:
        for key in keys:
            emp_t = emp_t_dict[key][..., np.newaxis]
            p_values[key] = np.mean(emp_t < max_null_dict[key], axis=-1)
    else:
        for key in keys:
            emp_t = emp_t_dict[key][..., np.newaxis]
            p_values[key] = np.mean(emp_t < max_null_dict[key].swapaxes(0, 1)[None, None, ...], axis=-1)

    return p_values, adj_matrices, max_null_dict