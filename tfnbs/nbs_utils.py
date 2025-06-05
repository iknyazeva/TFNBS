import numpy as np
import numpy.typing as npt
from pairwise_tfns import *
from utils import *

def nbs_bct(group1: npt.NDArray[np.float64],
            group2: npt.NDArray[np.float64],
            threshold: int = 2.0,
            n_permutations: int = 1000,
            paired: bool = True,
            use_mp: bool = True,
            random_state: Optional[int] = None,
            n_processes: Optional[int] = None,
            **kwargs):
    
    '''
    Parameters:
        group1 (np.ndarray): Input matrices of group 1 of dimension K*N*N
        group2 (np.ndarray): Input matrices of group 2 of dimension K*N*N
        threshold (int): Explcity cut-off threshold for T-statistics
        n_permutations (int): Number of permutations for null distribution
        paired (bool): Test type (False, individual) (True, Paired test)
        use_mp (bool): Use multiple cores for computation
        random_state (int): static Random state
        n_processes (int): number of processor cores for parallel computation

    Returns: 
        p_values (np.ndarray): Computed p-values for given groups
        adj_matrices (np.ndarray): Adjoint matrix of N*N dimensions representing boolean state greater than threshold
        max_null_dict : REVISE

    Note: The output adjoint matrix consists of both comparisons comprise of tails explicitly given as g1>g2 & g2>g1
    
    ---- NEED TO VERIFY IF EXPLICIT TAIL FUNCTIONALITY IS ACCURATE 

    '''
    # Need to add conditional check for shapes of Group 1/group 2 

    # need to add condition for Paired condition iX == iy should be true

    # need to add condition to ensure number of cores are mentioned whne using use_mp = true 


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