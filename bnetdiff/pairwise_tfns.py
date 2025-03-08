import numpy as np
from typing import Optional, Callable, Any, Union, Dict, Tuple
import numpy.typing as npt
from functools import partial

from numpy import ndarray, dtype

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
    Compute max t-statistic for a single permutation (worker function).

    Args:
        full_group: Concatenated array of shape (n_samples_1 + n_samples_2, *dims).
        n1: Number of samples in group 1.
        seed: Random seed for this permutation.

    Returns:
        Maximum t-statistic (scalar) for the permutation.
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
    Compute max t-statistic for a single permutation (worker function) for paired groups.

    Args:
        diffs: Difference between arrays of shape (n_samples, *dims).
        func: Function to compute the t-statistic, either compute_permute_t_stat_diff or compute_permute_t_stat_tfnos_diff.
        seed: Random seed for this permutation.

    Returns:
        Maximum t-statistic (scalar) for the permutation.
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
    Compute maximum t-statistics for multiple permutations of independent groups using multiprocessing.

    Args:
        paired: if repeated measures or individual group comparisons
        func: function to compute t-statistics.
        group1: Array of shape (n_samples_1, *dims) containing data for group 1.
            For EEG, dims could be (n_channels, n_frequencies, n_corr_types).
        group2: Array of shape (n_samples_2, *dims) containing data for group 2.
            Trailing dimensions must match group1.
        n_permutations: Number of permutations to perform. Defaults to 1000.
        random_state: Seed for random number generator. If None, uses system randomness.
            Defaults to None.
        n_processes: Number of CPU processes to use. If None, uses cpu_count().
            Defaults to None.
        use_mp: Whether to use multiprocessing.

    Returns:
        Array of shape (n_permutations,) containing maximum t-statistics for each permutation.

    Raises:
        ValueError: If shapes are incompatible, sample sizes are too small, or n_permutations < 1.
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
    seeds = rng.randint(0, 2 ** 32 - 1, size=n_permutations)

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
    Compute the maximum t-statistic for a single permutation of independent groups.

    Args:
        group1: Array of shape (n_samples_1, *dims) containing data for group 1.
            For EEG, dims could be (n_channels, n_frequencies, n_corr_types).
        group2: Array of shape (n_samples_2, *dims) containing data for group 2.
            Trailing dimensions must match group1.
        random_state: Seed for random number generator. If None, uses system randomness.
            Defaults to None.

    Returns:
        Maximum t-statistic (scalar) across all dimensions for the permuted groups.

    Raises:
        ValueError: If shapes are incompatible or sample sizes are too small.

    Notes:
        Permutes group assignments by shuffling the concatenated data and splitting
        into original group sizes. Assumes compute_t_stat_ind computes Welch’s t-test.
        Useful for building a null distribution in permutation testing.
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

    Args:
        group1: Array of shape (n_samples_1, N*N) containing data for group 1.
        group2: Array of shape (n_samples_2, NxN) containing data for group 2.
        paired: Whether to compute pairwise t-statistics.
        e: Exponent parameter for TFCE transformation (default=0.4).
        h: Height parameter for TFCE transformation (default=3).
        n: Number of integration steps in TFCE (default=10).

    Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with:
            - "g2>g1": TFCE score for positive t-values (g2 > g1).
            - "g1>g2": TFCE score for negative t-values (g1 > g2).

    Notes:
        - Uses TFCE transformation on Welch’s t-statistics.
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

    Args:
        diffs: Array of shape (*dims) representing pairwise differences.
        e: Exponent parameter for TFCE transformation (default=0.4).
        h: Height parameter for TFCE transformation (default=3).
        n: Number of integration steps in TFCE (default=10).

    Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with:
            - "g2>g1": TFCE score for positive t-values (g2 > g1).
            - "g1>g2": TFCE score for negative t-values (g1 > g2).

    Notes:
        - Uses TFCE transformation on Welch’s t-statistics.
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
            - "g2>g1": Array of t-values where group 2 > group 1 (positive t-values).
            - "g1>g2": Array of t-values where group 1 > group 2 (negative t-values, converted to positive).

    Raises:
        ValueError: If shapes are incompatible or sample sizes don’t match for paired test.
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
    Compute differences between paired samples, second minus first group

    Args:
        group1: Array of shape (n_samples, *dims) for group 1.
        group2: Array of shape (n_samples, *dims) for group 2, matching group1’s shape.

    Returns:
        Array of differences with shape (n_samples, *dims).
    """
    return group2 - group1


def compute_t_stat_diff(diff: npt.NDArray[np.float64]) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute t-statistics for paired differences.

    Args:
        diff: Array of shape (n_samples, *dims) containing paired differences.
            For EEG, dims could be (n_channels, n_frequencies, n_corr_types).

     Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with keys:
            - "g2>g1": Array of t-values where group 2 > group 1 (positive t-values).
            - "g1>g2": Array of t-values where group 1 > group 2 (negative t-values, converted to positive).

    Notes:
        Uses sample standard deviation with ddof=1 for unbiased variance estimation.
    """

    assert np.allclose(diff.mean(axis=0), diff.mean(axis=0).T, atol=1e-8), "Only symmetric differences are supported. Participants should be along 0 axis"
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

    Args:
        group1: Array of shape (n_samples_1, *dims) for group 1.
        group2: Array of shape (n_samples_2, *dims) for group 2.
            Trailing dimensions must match.

    Returns:
        Dict[str, npt.NDArray[np.float64]]: Dictionary with keys:
            - "g2>g1": Array of t-values where group 2 > group 1 (positive t-values).
            - "g1>g2": Array of t-values where group 1 > group 2 (negative t-values, converted to positive).

    Notes:
        Uses Welch’s t-test (unequal variances assumed) with ddof=1 for variance.

       Examples:
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

    # Compute t-statistic with Welch’s formula
    denominator = np.sqrt(x_var_1 + x_var_2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = (x_mean_2 - x_mean_1) / denominator
        t_stat = np.where(denominator == 0, 0, t_stat)  # Handle zero variance

    # Split into positive and negative components
    pos_t = np.where(t_stat > 0, t_stat, 0)
    neg_t = np.where(t_stat < 0, -t_stat, 0)  # Convert negatives to positive values

    return {"g2>g1": pos_t, "g1>g2": neg_t}
