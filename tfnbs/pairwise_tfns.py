import numpy as np
import numpy.typing as npt
from numpy import ndarray, dtype, triu_indices
from typing import Optional, Callable, Any, Union, Dict, Tuple
from functools import partial
from .tfnos import get_tfce_score_scipy
from multiprocessing import Pool


def compute_p_val(group1: npt.NDArray[np.float64],
                  group2: npt.NDArray[np.float64],
                  n_permutations: int = 1000,
                  paired: bool = True,
                  tf: bool = True,
                  use_mp: bool = True,
                  random_state: Optional[int] = None,
                  n_processes: Optional[int] = None,
                  **kwargs):
    """
    Function to compute P-values for statistical data using TFNOS and Standard T-Test approaches for paired and individual groups.
    
    Parameters:
        group1 (np.float64): Input array of matrices of group 1 with shape (subjects_g1, N, N).
        group2 (np.float64): Input array of matrices of group 2 with shape (subjects_g2, N, N).
        n_permutations (int): Number of permutations for null distribution (default = 1000).
        paired (bool): Test type (False, individual), (True,paired).
        tf (bool): T statistics to be generated via TFCE or standard t-test (True, TFCE T-statistics), (False, Standard T-test).
        use_mp (bool): Use parallel pools for computing (default = True).
        random_state (int): Set Random seed (optional).
        n_processes (int): Set CPU cores for parallel computing (optional).

    Returns: 
        p_values (dict[str, np.ndarray]): Dictionary containing computed p-values for given data.
            - 'g1>g2': P-values for group 1 > group 2.
            - 'g2>g1': P-values for group 2 > group 1.

    >>> group1 = np.random.rand(5, 3, 3); for arr in group1: np.fill_diagonal(arr,1)
    >>> group2 = np.random.rand(8, 3, 3); for arr in group2: np.fill_diagonal(arr,1)
    >>> p_vals = compute_p_val(group1, group2, n_permutations=10, paired=False, tf=False, random_state = 0)
    >>> p_vals['g2>g1'].mean() < p_vals['g1>g2'].mean() 
    True
    
    """
    if paired is True:
        t_func = compute_t_stat_tfnos_diffs if tf else compute_t_stat_diff
        emp_t_dict = t_func(compute_diffs(group1, group2), **kwargs)

    else:
        t_func = compute_t_stat_tfnos if tf else compute_t_stat
        emp_t_dict = t_func(group1, group2, paired=False, **kwargs)
    max_null_dict = compute_null_dist(group1,
                                      group2,
                                      t_func,
                                      n_permutations=n_permutations,
                                      paired=paired,
                                      use_mp=use_mp,
                                      random_state=random_state,
                                      n_processes=n_processes,
                                      **kwargs)

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

    return p_values


def _permutation_task_ind(full_group: npt.NDArray[np.float64],
                          func: Callable[..., Any],
                          n1: int,
                          seed: int,
                          **func_kwargs,
                          ) -> Dict[str, Union[float, npt.NDArray[np.float64]]]:
    """
    Compute maximum t-statistic for a single permutation for individual sample groups.

    Parameters:
        full_group (np.ndarray): Concatenated data array of shape (n_samples_1 + n_samples_2, *dims).
        func (Callable): Function to compute the t-statistic, either compute_permute_t_stat_diff or compute_permute_t_stat_tfnos_diff.
        n1 (int): Number of samples in group 1.
        seed (int): Random seed for this permutation and reproducability of results.

    Returns:
        dict: Dictionary with keys:
            - "g1>g2": Maximum t-statistics for group 1 > group 2.
            - "g2>g1": Maximum t-statistics for group 2 > group 1.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(full_group.shape[0])
    new_group1 = full_group[idx[:n1]]
    new_group2 = full_group[idx[n1:]]
    perm_stat_dict = func(new_group1, new_group2, paired=False, **func_kwargs)
    if perm_stat_dict["g1>g2"].shape == full_group[0].shape:
        max_dict = {"g1>g2": np.max(perm_stat_dict["g1>g2"]).astype(np.float64),
                    "g2>g1": np.max(perm_stat_dict["g2>g1"]).astype(np.float64)}
    else:
        max_dict = {
            "g1>g2": np.max(perm_stat_dict["g1>g2"], axis=tuple(range(perm_stat_dict["g1>g2"].ndim - 1))).astype(
                np.float64),
            "g2>g1": np.max(perm_stat_dict["g2>g1"], axis=tuple(range(perm_stat_dict["g2>g1"].ndim - 1))).astype(
                np.float64)}
    return max_dict


def _permutation_task_paired(diffs: npt.NDArray[np.float64],
                             func: Callable[..., Any],
                             seed: Optional[int] = None,
                             **func_kwargs) -> Dict[str, Union[float, npt.NDArray[np.float64]]]:
    """
    Compute maximum t-statistic for a single permutation for paired sample groups.

    Parameters:
        diffs (np.ndarray): Arrays of shape (n_samples, *dims) containing paired differences between two conditions.
        func (Callable): Function to compute the t-statistic, either compute_permute_t_stat_diff or compute_permute_t_stat_tfnos_diff.
        seed (int): Random seed for this permutation (optional).

    Returns:
        dict: Dictionary with keys:
            - "g1>g2": Maximum t-statistics for group 1 > group 2.
            - "g2>g1": Maximum t-statistics for group 2 > group 1.
    """
    n_dims = len(diffs.shape) - 1
    faked_dims = [1] * n_dims
    rng = np.random.RandomState(seed)
    new_diffs = rng.choice([1, -1], diffs.shape[0]).reshape(-1, *faked_dims) * diffs
    perm_stat_dict = func(new_diffs, **func_kwargs)
    if perm_stat_dict["g1>g2"].shape == diffs[0].shape:
        max_dict = {"g1>g2": np.max(perm_stat_dict["g1>g2"]).astype(np.float64),
                    "g2>g1": np.max(perm_stat_dict["g2>g1"]).astype(np.float64)}
    else:
        max_dict = {
            "g1>g2": np.max(perm_stat_dict["g1>g2"], axis=tuple(range(perm_stat_dict["g1>g2"].ndim - 1))).astype(
                np.float64),
            "g2>g1": np.max(perm_stat_dict["g2>g1"], axis=tuple(range(perm_stat_dict["g2>g1"].ndim - 1))).astype(
                np.float64)}
    return max_dict


def compute_null_dist(group1: npt.NDArray[np.float64],
                      group2: npt.NDArray[np.float64],
                      func: Callable[..., Any],
                      n_permutations: int = 1000,
                      paired: bool = False,
                      random_state: Optional[int] = None,
                      n_processes: Optional[int] = None,
                      use_mp: bool = False,
                      **func_kwargs) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute null distribution of maximum t-statistics for multiple permutations of independent or paired groups.

    Parameters:
        group1 (np.ndarray): Array of shape (n_samples_1, *dims) containing data for group 1.
            For EEG, dims could be (n_channels, n_frequencies, n_corr_types).
        group2 (np.ndarray): Array of shape (n_samples_2, *dims) containing data for group 2.
            Trailing dimensions must match group 1.
        func (Callable): Function to compute the t-statistic as input. 
        paired (bool): Computation to be done as repeated measures or individual group comparisons.
        random_state (int): Seed for random number generator. Ensures reproducibility (optional).
        n_processes (int): Number of parallel processes to use if 'use_mp=True', if None, uses cpu_count().
        use_mp (bool): Whether to use parallel computations (default = False).

    Returns:
        t_maxes_dict (dict[str, np.ndarray]): Dictionary with keys:
            - 'g1>g2': Maximum t-statistics for group 1 > group 2.
            - 'g2>g1': Maximum t-statistics for group 2 > group 1.

    Raises:
        ValueError: If shapes are incompatible, sample sizes are too small, or n_permutations < 1.
        
    >>> group1 = np.random.rand(3, 3)
    >>> group2 = np.random.rand(3, 3)
    >>> null_d = compute_null_dist(group1, group2, compute_t_stat)
    >>> isinstance(null_d, dict)
    True

    """
    # Validate inputs
    if group1.shape[1:] != group2.shape[1:]:
        raise ValueError("Trailing dimensions of group1 and group2 must match.")
    n1, n2 = group1.shape[0], group2.shape[0]
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 samples.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    # Concatenate groups once
    if paired:
        array_to_permute = compute_diffs(group1, group2)
    else:
        array_to_permute = np.concatenate((group1, group2), axis=0)

    # Set random state and generate unique seeds
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2 ** 32 - 1, size=n_permutations, dtype=np.int64)

    # Prepare arguments for starmap: list of (full_group, n1, seed) tuples
    #task_args = [(full_group, func, n1, seed, func_kwargs) for seed in seeds]

    # Compute t-statistics based on use_cycle
    if use_mp is False:
        # Sequential computation with a for loop
        if paired:
            sample_output_dict = _permutation_task_paired(array_to_permute, func, seeds[0], **func_kwargs)
        else:
            sample_output_dict = _permutation_task_ind(array_to_permute, func, n1, seeds[0], **func_kwargs)
        group_keys = list(sample_output_dict.keys())
        output_shape = sample_output_dict[group_keys[0]].shape

        # Allocate space based on determined shape
        t_maxes_dict = {key: np.empty((n_permutations, *output_shape), dtype=np.float64) for key in group_keys}
        #t_maxes = np.empty(n_permutations, dtype=np.float64)
        for i, seed in enumerate(seeds[1:]):
            print(f"  Permutation {i + 1} of {n_permutations}")
            if paired:
                perm_dict = _permutation_task_paired(array_to_permute, func, seed, **func_kwargs)
                for k, v in t_maxes_dict.items():
                    t_maxes_dict[k][i] = perm_dict[k]
            else:
                perm_dict = _permutation_task_ind(array_to_permute, func, n1, seed, **func_kwargs)
                for k, v in t_maxes_dict.items():
                    t_maxes_dict[k][i] = perm_dict[k]
    else:
        # Parallel computation with multiprocessing

        # Set number of processes
        if n_processes is None:
            import multiprocessing
            n_processes = multiprocessing.cpu_count()
        n_processes = min(n_processes, n_permutations)

        # Use multiprocessing Pool with starmap
        with Pool(processes=n_processes) as pool:
            #t_maxes = pool.starmap(_permutation_task_ind, task_args)
            if paired:
                task_dict = partial(_permutation_task_paired, array_to_permute, func, **func_kwargs)
            else:
                task_dict = partial(_permutation_task_ind, array_to_permute, func, n1, **func_kwargs)
            results = pool.map(task_dict, seeds)
            group_keys = list(results[0].keys())
            output_shape = results[0][group_keys[0]].shape
            t_maxes_dict = {key: np.empty((n_permutations, *output_shape), dtype=np.float64) for key in group_keys}

            for i, perm_dict in enumerate(results):
                for k in group_keys:
                    t_maxes_dict[k][i] = perm_dict[k]

    return t_maxes_dict


def compute_permute_t_stat_ind(group1: npt.NDArray[np.float64],
                               group2: npt.NDArray[np.float64],
                               random_state: Optional[int] = None) -> tuple[float, float]:
    """
    Computes the maximum t-statistic for a single permutation of independent groups.
    
    Parameters:
        group1 (np.ndarray): Array of shape (n_samples_1, *dims) containing data for group 1.
            For EEG, dims could be (n_channels, n_frequencies, n_corr_types).
        group2 (np.ndarray): Array of shape (n_samples_2, *dims) containing data for group 2.
            Trailing dimensions must match group 1.
        random_state (int): Seed for random number generator. If None, uses system randomness.
            Defaults to None (Optional).

    Returns:
        Maximum t-statistic across all dimensions for the permuted groups as tuple[float, float]
            - 'g1>g2': Maximum t-statistics for group 1 > group 2.
            - 'g2>g1': Maximum t-statistics for group 2 > group 1.

    Raises:
        ValueError: If shapes are incompatible or sample sizes are too small.

    Notes:
        Permutes group assignments by shuffling the concatenated data and splitting
        into original group sizes. Assumes compute_t_stat_ind computes Welch's t-test.
        Useful for building a null distribution in permutation testing.

    >>> group1 = np.random.rand(5, 3, 3)
    >>> group2 = np.random.rand(5, 3, 3)
    >>> perm_t_pos, perm_t_neg = compute_permute_t_stat_ind(group1, group2, 10)
    >>> perm_t_pos >1
    True
    >>> perm_t_neg >1
    True

    """
    # Validate input shapes
    if group1.shape[1:] != group2.shape[1:]:
        raise ValueError("Trailing dimensions of group1 and group2 must match.")
    n1, n2 = group1.shape[0], group2.shape[0]
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 samples.")

    # Set random state for reproducibility
    rng = np.random.RandomState(random_state)

    # Concatenate groups along sample axis
    full_group = np.concatenate((group1, group2), axis=0)

    # Generate shuffled indices efficiently
    index_shuf = rng.permutation(full_group.shape[0])

    # Split into permuted groups
    new_group1 = full_group[index_shuf[:n1]]
    new_group2 = full_group[index_shuf[n1:]]

    # Compute and return maximum t-statistic
    t_stat_dict = compute_t_stat_ind(new_group1, new_group2)
    return np.max(t_stat_dict["g2>g1"]).astype(float), np.max(t_stat_dict["g1>g2"]).astype(float)


def compute_permute_t_stat_diff(diffs: npt.NDArray) -> tuple[float, float]:
    """
    Computes the maximum t-statistics for a single permutation of paired groups of data

    Parameters: 
        diffs (np.ndarray): Array of shape (n_subjects, *dims) containing paired differences

    Returns: 
        Maximum t-statistic across all dimensions for the permuted groups as tuple[float, float]
            - 'g1>g2': Maximum t-statistics for group 1 > group 2.
            - 'g2>g1': Maximum t-statistics for group 2 > group 1.
    
    >>> group1 = np.random.rand(5, 3, 3)
    >>> group2 = np.random.rand(5, 3, 3)
    >>> diffs = group2 - group1
    >>> perm_t_pos, perm_t_neg = compute_permute_t_stat_ind(diffs)
    >>> perm_t_pos > 1
    True
    >>> perm_t_neg > 1
    True
    """
    n_dims = len(diffs.shape) - 1
    faked_dims = [1] * n_dims
    perm_diffs = np.random.choice([1, -1], diffs.shape[0]).reshape(-1, *faked_dims) * diffs
    t_stat_dict = compute_t_stat_diff(perm_diffs)
    return np.max(t_stat_dict["g2>g1"]).astype('float'), np.max(t_stat_dict["g1>g2"]).astype(float)


def compute_t_stat_tfnos(group1: npt.NDArray[np.float64],
                         group2: npt.NDArray[np.float64],
                         paired: bool = False,
                         e: Union[float, list[float]] = 0.4,
                         h: Union[float, list[float]] = 3,
                         n: int = 10) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute TFCE-enhanced t-statistics for independent groups, return separate
    scores for positive (g2 > g1) and negative (g1 > g2) effects.

    Parameters:
        group1 (np.ndarray): Array of shape (n_samples_1, N*N) containing data for group 1.
        group2 (np.ndarray): Array of shape (n_samples_2, N*N) containing data for group 2.
        paired (bool): Flag to compute pairwise t-statistics or as per individual groups.         
        e (float or List[float]): Exponent parameter for TFCE transformation (default=0.4).
        h (float or List[float]): Height parameter for TFCE transformation (default=3).
        n (int): Number of integration steps in TFCE transformation (default=10).

    Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with:
            - 'g2>g1': TFCE score for positive t-values (g2 > g1).
            - 'g1>g2': TFCE score for negative t-values (g1 > g2).

    Notes:
        - Uses TFCE transformation on Welch's t-statistics.

    >>> np.random.seed(2)
    >>> group1 = np.random.rand(5, 3, 3); group2 = np.random.rand(5, 3, 3)
    >>> for i in range(group1.shape[0]):
    >>>     np.fill_diagonal(group1[i], 0)
    >>>     np.fill_diagonal(group2[i], 0)
    >>> t_stat_dict = compute_t_stat(group1, group2, False)
    >>> results = get_tfce_score_scipy(t_stat_dict["g1>g2"], 0.4, 3, 10)
    >>> upper_vals = results.reshape(3, 3)[np.triu_indices(3, k=1)]
    >>> round(upper_vals.mean(), 6) < N
    True        
    """
    t_stat_dict = compute_t_stat(group1, group2, paired=paired)
    score_pos = get_tfce_score_scipy(t_stat_dict["g2>g1"], e, h, n)
    score_neg = get_tfce_score_scipy(t_stat_dict["g1>g2"], e, h, n)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def compute_t_stat_tfnos_diffs(diffs: npt.NDArray[np.float64],
                               e: Union[float, list[float]] = 0.4,
                               h: Union[float, list[float]] = 3,
                               n: int = 10,
                               start_thres: float = 1.65) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute TFCE-enhanced t-statistics from difference matrices and return separate
    scores for positive (g2 > g1) and negative (g1 > g2) effects.

    Parameters:
        diffs (np.ndarray): Array of shape (*dims) representing pairwise differences between two groups.
        e (float or List[float]): Exponent parameter for TFCE transformation (default=0.4).
        h (float or List[float]): Height parameter for TFCE transformation (default=3).
        n (int): Number of integration steps in TFCE transformation (default=10).

    Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with:
            - 'g2>g1': TFCE score for positive t-values (g2 > g1).
            - 'g1>g2': TFCE score for negative t-values (g1 > g2).

    Notes:
        - Uses TFCE transformation on Welch's t-statistics.

    >>> np.random.seed(2)
    >>> group1 = np.random.rand(5, 3, 3); group2 = np.random.rand(5, 3, 3)
    >>> for i in range(group1.shape[0]):
    >>>     np.fill_diagonal(group1[i], 0)
    >>>     np.fill_diagonal(group2[i], 0)
    >>> diff = group1-group2
    >>> result = compute_t_stat_tfnos_diffs(diff, e=0.4, h=3, n=10, start_thres=1.65)
    >>> upper_vals = result["g1>g2"].reshape(3, 3)[np.triu_indices(3, k=1)]
    >>> round(upper_vals.mean(), 6) < N
    True

    """
    t_stat_dict = compute_t_stat_diff(diffs)
    score_pos = get_tfce_score_scipy(t_stat_dict["g2>g1"], e, h, n, start_thres=start_thres)
    score_neg = get_tfce_score_scipy(t_stat_dict["g1>g2"], e, h, n, start_thres=start_thres)
    return {"g2>g1": score_pos, "g1>g2": score_neg}


def compute_t_stat(group1: npt.NDArray[np.float64],
                   group2: npt.NDArray[np.float64],
                   paired: bool = True) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute empirical t-statistics for paired or independent groups.

    Args:
        group1: Array of shape (n_samples_1, N*N) containing data for group 1.
        group2: Array of shape (n_samples_2, NxN) containing data for group 2.
            Must match group1's trailing dimensions; n_samples_2 may differ if paired=False.
        paired: If True, compute paired t-test on differences; if False, independent t-test.
            Defaults to True.

        Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with keys:
            - 'g2>g1': Array of t-values where group 2 > group 1 (positive t-values).
            - 'g1>g2': Array of t-values where group 1 > group 2 (negative t-values, converted to positive).

    Raises:
        ValueError: If shapes are incompatible or sample sizes don't match for paired test.


    >>> group_1 = np.array([[0, 2, 1], [3, 0, 1], [2, 2, 0]])
    >>> group_2 = np.array([[0, 1, 3], [1, 0, 1], [3, 1, 0]])
    >>> result = compute_t_stat(group1, group2, paired = True)
    >>> result['g2>g1'].shape[0] ==  group_1.shape[0]
    True

    """
    # Validate input shapes
    if group1.shape[1:] != group2.shape[1:]:
        raise ValueError("Trailing dimensions of group1 and group2 must match.")
    if paired and group1.shape[0] != group2.shape[0]:
        raise ValueError("Sample sizes must match for paired t-test.")

    if paired:
        diffs = compute_diffs(group1, group2)
        t_stat_dict = compute_t_stat_diff(diffs)
    else:
        t_stat_dict = compute_t_stat_ind(group1, group2)
    return t_stat_dict


def compute_diffs(group1: npt.NDArray[np.float64],
                  group2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute differences between paired samples (second group minus first group)

    Parameters:
        group1 (np.ndarray): Array of shape (n_samples, *dims) for group 1.
        group2 (np.ndarray): Array of shape (n_samples, *dims) for group 2, matching group1's shape.

    Returns:
        Array of differences with shape (n_samples, *dims).
    
    >>> group_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> group_2 = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    >>> (compute_diffs(group_1, group_2)==np.eye(3)).all()
    True
    """
    return group2 - group1


def compute_t_stat_diff(diff: npt.NDArray[np.float64]) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute t-statistics for paired differences between two groups of data. 

    Parameters:
        diff (np.ndarray): Array of shape (n_samples, *dims) containing paired differences.
            For EEG, dims could be (n_channels, n_frequencies, n_corr_types).

     Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with keys:
            - 'g2>g1': Array of t-values where group 2 > group 1 (positive t-values).
            - 'g1>g2': Array of t-values where group 1 > group 2 (negative t-values, converted to positive).

    Notes:
        Uses sample standard deviation with ddof=1 for unbiased variance estimation.

    >>> group_1 = np.array([[0, 2, 1], [3, 0, 1], [2, 2, 0]])
    >>> group_2 = np.array([[0, 1, 3], [1, 0, 1], [3, 1, 0]])
    >>> result = compute_t_stat_diff(compute_diffs(group1, group2))
    >>> result['g2>g1'].shape[0] == group1.shape[0]
    True
    """

    # assert np.allclose(diff.mean(axis=0), diff.mean(axis=0).T, atol=1e-8), "Only symmetric differences are supported. Participants should be along 0 axis"
    n = diff.shape[0]
    if n < 2:
        raise ValueError("At least 2 samples required for t-statistic.")

    # Compute mean and standard error in one pass with NumPy
    x_mean = np.mean(diff, axis=0)
    x_std = np.std(diff, axis=0, ddof=1)  # Unbiased estimator
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = x_mean / (x_std / np.sqrt(n))
        t_stat = np.where(x_std == 0, 0, t_stat)  # Set t=0 where std=0
        # Split into positive and negative components
    pos_t = np.where(t_stat > 0, t_stat, 0)
    neg_t = np.where(t_stat < 0, -t_stat, 0)  # Convert negatives to positive values

    return {"g2>g1": pos_t, "g1>g2": neg_t}


def compute_t_stat_ind(group1: npt.NDArray[np.float64],
                       group2: npt.NDArray[np.float64]) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute t-statistics for independent samples and split results into positive (g2 > g1)
    and negative (g1 > g2) values.

    Parameters:
        group1 (np.ndarray): Array of shape (n_samples_1, *dims) for group 1.
        group2 (np.ndarray): Array of shape (n_samples_2, *dims) for group 2.
            Trailing dimensions of either groups must match.

    Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with keys:
            - 'g2>g1': Array of t-values where group 2 > group 1 (positive t-values).
            - 'g1>g2': Array of t-values where group 1 > group 2 (negative t-values, converted to positive).

    Notes:
        Uses Welch's t-test (unequal variances assumed) with ddof=1 for variance.

    >>> np.random.seed(0)
    >>> g1 = np.random.randn(10, 5)
    >>> g2 = np.random.randn(12, 5) + 0.5  # Slightly higher mean
    >>> result = compute_t_stat_ind(g1, g2)
    >>> result["g2>g1"].shape == result["g1>g2"].shape
    True
    >>> (result["g2>g1"] >= 0).all() and (result["g1>g2"] >= 0).all()
    True
    """
    n1, n2 = group1.shape[0], group2.shape[0]
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 samples.")

    # Compute means and variances
    x_mean_1 = np.mean(group1, axis=0)
    x_mean_2 = np.mean(group2, axis=0)
    x_var_1 = np.var(group1, axis=0, ddof=1) / n1  # Sample variance, unbiased
    x_var_2 = np.var(group2, axis=0, ddof=1) / n2

    # Compute t-statistic with Welch's formula
    denominator = np.sqrt(x_var_1 + x_var_2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = (x_mean_2 - x_mean_1) / denominator
        t_stat = np.where(denominator == 0, 0, t_stat)  # Handle zero variance

    # Split into positive and negative components
    pos_t = np.where(t_stat > 0, t_stat, 0)
    neg_t = np.where(t_stat < 0, -t_stat, 0)  # Convert negatives to positive values

    return {"g2>g1": pos_t, "g1>g2": neg_t}
