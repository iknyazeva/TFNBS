import numpy as np
from typing import Optional, Callable, Any, Union
import numpy.typing as npt
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
    if paired is True:
        t_func = compute_t_stat_tfnos_diffs if tf else compute_t_stat_diff
        emp_t = t_func(compute_diffs(group1, group2), **kwargs)

    else:
        t_func = compute_t_stat_tfnos if tf else compute_t_stat
        emp_t = t_func(group1, group2, paired=False, **kwargs)
    max_null = compute_null_dist(group1,
                                 group2,
                                 t_func,
                                 n_permutations=n_permutations,
                                 paired=paired,
                                 use_mp=use_mp,
                                 random_state=random_state,
                                 n_processes=n_processes,
                                 **kwargs)

    if len(emp_t.shape) == 2:
        emp_t = emp_t[..., np.newaxis]
        p_values = np.mean(emp_t < max_null, axis=-1)
    else:
        emp_t = emp_t[..., np.newaxis]


        p_values = np.mean(emp_t < max_null.swapaxes(0,1)[None, None, ...], axis=-1)

    return p_values.squeeze()


def _permutation_task_ind(full_group: npt.NDArray[np.float64],
                          func: Callable[..., Any],
                          n1: int,
                          seed: int,
                          **func_kwargs,
                          ) -> Union[float, npt.NDArray[np.float64]]:
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
    perm_stat = func(new_group1, new_group2, paired=False, **func_kwargs)
    if perm_stat.shape == full_group[0].shape:
        return np.max(perm_stat).astype(np.float64)
    else:
        return np.max(perm_stat, axis=tuple(range(perm_stat.ndim - 1))).astype(np.float64)


def _permutation_task_paired(diffs: npt.NDArray[np.float64],
                             func: Callable[..., Any],
                             seed: Optional[int] = None,
                             **func_kwargs) -> Union[float, npt.NDArray[np.float64]]:
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
    perm_stat = func(new_diffs, **func_kwargs)
    if perm_stat.shape == diffs[0].shape:
        return np.max(perm_stat).astype(np.float64)
    else:
        return np.max(perm_stat, axis=tuple(range(perm_stat.ndim - 1))).astype(np.float64)


def compute_null_dist(group1: npt.NDArray[np.float64],
                      group2: npt.NDArray[np.float64],
                      func: Callable[..., Any],
                      n_permutations: int = 1000,
                      paired: bool = False,
                      random_state: Optional[int] = None,
                      n_processes: Optional[int] = None,
                      use_mp: bool = False,
                      **func_kwargs) -> npt.NDArray[np.float64]:
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
            sample_output = _permutation_task_paired(array_to_permute, func, seeds[0], **func_kwargs)
        else:
            sample_output = _permutation_task_ind(array_to_permute, func, n1, seeds[0], **func_kwargs)
        output_shape = sample_output.shape
        # Allocate space based on determined shape
        t_maxes = np.empty((n_permutations, *output_shape), dtype=np.float64)
        #t_maxes = np.empty(n_permutations, dtype=np.float64)
        for i, seed in enumerate(seeds[1:]):
            if paired:
                t_maxes[i] = _permutation_task_paired(array_to_permute, func, seed, **func_kwargs)
            else:
                t_maxes[i] = _permutation_task_ind(array_to_permute, func, n1, seed, **func_kwargs)
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
                task = partial(_permutation_task_paired, array_to_permute, func, **func_kwargs)
            else:
                task = partial(_permutation_task_ind, array_to_permute, func, n1, **func_kwargs)
            t_maxes = pool.map(task, seeds)

    return np.array(t_maxes, dtype=np.float64)


def compute_permute_t_stat_ind(group1: npt.NDArray[np.float64],
                               group2: npt.NDArray[np.float64],
                               random_state: Optional[int] = None) -> float:
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
    t_stats = compute_t_stat_ind(new_group1, new_group2)
    return np.max(np.abs(t_stats)).astype(float)


def compute_permute_t_stat_diff(diffs: npt.NDArray) -> npt.NDArray:
    n_dims = len(diffs.shape) - 1
    faked_dims = [1] * n_dims
    perm_diffs = np.random.choice([1, -1], diffs.shape[0]).reshape(-1, *faked_dims) * diffs
    return np.max(np.abs(compute_t_stat_diff(perm_diffs)))


def compute_t_stat_tfnos(group1: npt.NDArray[np.float64],
                         group2: npt.NDArray[np.float64],
                         paired: bool = False,
                         e=0.4,
                         h=3,
                         n=10) -> npt.NDArray[np.float64]:
    t_stat = compute_t_stat(group1, group2, paired=paired)
    score_pos = get_tfce_score_scipy(t_stat, e, h, n)
    score_neg = get_tfce_score_scipy(-t_stat, e, h, n)
    return score_pos + score_neg


def compute_t_stat_tfnos_diffs(diffs: npt.NDArray[np.float64],
                               e=0.4,
                               h=3,
                               n=10) -> npt.NDArray[np.float64]:
    t_stat = compute_t_stat_diff(diffs)
    score_pos = get_tfce_score_scipy(t_stat, e, h, n)
    score_neg = get_tfce_score_scipy(-t_stat, e, h, n)
    return score_pos + score_neg


def compute_t_stat(group1: npt.NDArray[np.float64],
                   group2: npt.NDArray[np.float64],
                   paired: bool = True) -> npt.NDArray[np.float64]:
    """
    Compute empirical t-statistics for paired or independent groups.

    Args:
        group1: Array of shape (n_samples_1, *dims) containing data for group 1.
            For EEG, dims could be (n_channels, n_channels, n_frequencies ).
        group2: Array of shape (n_samples_2, *dims) containing data for group 2.
            Must match group1's trailing dimensions; n_samples_2 may differ if paired=False.
        paired: If True, compute paired t-test on differences; if False, independent t-test.
            Defaults to True.

    Returns:
        Array of t-statistics with shape equal to dims (e.g., n_channels, n_frequencies, n_corr_types).

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
        t_stat = compute_t_stat_diff(diffs)
    else:
        t_stat = compute_t_stat_ind(group1, group2)
    return t_stat


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


def compute_t_stat_diff(diff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute t-statistics for paired differences.

    Args:
        diff: Array of shape (n_samples, *dims) containing paired differences.
            For EEG, dims could be (n_channels, n_frequencies, n_corr_types).

    Returns:
        Array of t-statistics with shape dims.

    Notes:
        Uses sample standard deviation with ddof=1 for unbiased variance estimation.
    """
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
    return t_stat


def compute_t_stat_ind(group1: npt.NDArray[np.float64],
                       group2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute t-statistics for independent samples.

    Args:
        group1: Array of shape (n_samples_1, *dims) for group 1.
        group2: Array of shape (n_samples_2, *dims) for group 2.
            Trailing dimensions must match.

    Returns:
        Array of t-statistics with shape dims.

    Notes:
        Uses Welch’s t-test (unequal variances assumed) with ddof=1 for variance.
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

    return t_stat
