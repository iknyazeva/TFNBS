import numpy as np
import numpy.typing as npt
import warnings


def fisher_r_to_z(r: npt.NDArray[np.float64],
                  handle_bounds: bool = True,
                  max_z: float = 5) -> npt.NDArray[np.float64]:
    """
    Apply Fisher r-to-z transformation to correlation coefficients, handling boundary cases.

    Args:
        r: Array of correlation coefficients with values in [-1, 1].
           Can be any shape (scalar, 1D, 2D, etc.).
        handle_bounds: If True, replace infinite z-values (from r = ±1) with finite values.
           If False, allow infinities and raise a warning. Defaults to True.
        max_z: Maximum absolute z-value to use when handle_bounds=True.
           Defaults to 1e10 (a large but finite value).

    Returns:
        Array of z-scores with the same shape as r.

    Raises:
        ValueError: If any value in r is outside [-1, 1].

    Warnings:
        UserWarning: If r contains ±1, indicating boundary values were encountered.

    Notes:
        The transformation is z = arctanh(r). For r = ±1, z approaches ±infinity.
        When handle_bounds=True, these are capped at ±max_z.
    """
    # Convert input to numpy array and ensure float type
    r = np.asarray(r, dtype=np.float64)

    # Check that all values are in [-1, 1]
    if np.any((r < -1) | (r > 1)):
        raise ValueError("Correlation coefficients must be in the range [-1, 1].")

    # Apply Fisher transformation
    with np.errstate(invalid='ignore'):  # Suppress warnings for arctanh at ±1
        z = np.arctanh(r)

    # Check for boundary values (r = ±1)
    bounds_mask = np.isclose(r, 1.0) | np.isclose(r, -1.0)
    if np.any(bounds_mask):
        warnings.warn(
            "Input contains r = ±1, resulting in infinite z-values. "
            f"{'Clapped at ±' + str(max_z) if handle_bounds else 'Left as infinity.'}",
            UserWarning
        )
        if handle_bounds:
            # Replace inf with finite values
            z = np.where(bounds_mask, np.sign(r) * max_z, z)

    return z


# Inverse function (z to r) for completeness
def fisher_z_to_r(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert Fisher z-scores back to correlation coefficients.

    Args:
        z: Array of z-scores (any shape).

    Returns:
        Array of correlation coefficients in (-1, 1) with the same shape as z.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.tanh(z)


def get_components(A, no_depend=False):
    '''
    Returns the components of an undirected graph specified by the binary and
    undirected adjacency matrix adj. Components and their constitutent nodes
    are assigned the same index and stored in the vector, comps. The vector,
    comp_sizes, contains the number of nodes beloning to each component.

    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected adjacency matrix
    no_depend : Any
        Does nothing, included for backwards compatibility

    Returns
    -------
    comps : Nx1 np.ndarray
        vector of component assignments for each node
    comp_sizes : Mx1 np.ndarray
        vector of component sizes

    Notes
    -----
    Note: disconnected nodes will appear as components with a component
    size of 1

    Note: The identity of each component (i.e. its numerical value in the
    result) is not guaranteed to be identical the value returned in BCT,
    matlab code, although the component topology is.

    Many thanks to Nick Cullen for providing this implementation
    '''

    if not np.all(A == A.T):  # ensure matrix is undirected
        raise AssertionError('get_components can only be computed for undirected'
                             ' matrices.  If your matrix is noisy, correct it with np.around')

    A = binarize(A, copy=True)
    n = len(A)
    np.fill_diagonal(A, 1)

    edge_map = [{u, v} for u in range(n) for v in range(n) if A[u, v] == 1]
    union_sets = []
    for item in edge_map:
        temp = []
        for s in union_sets:

            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        union_sets = temp

    comps = np.array([i + 1 for v in range(n) for i in
                      range(len(union_sets)) if v in union_sets[i]])
    comp_sizes = np.array([len(s) for s in union_sets])

    return comps, comp_sizes


def binarize(W, copy=True):
    '''
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W

#--
class BCTParamError(RuntimeError):
    pass


def teachers_round(x):
    '''
    Do rounding such that .5 always rounds to 1, and not bankers rounding.
    This is for compatibility with matlab functions, and ease of testing.
    '''
    if ((x > 0) and (x % 1 >= 0.5)) or ((x < 0) and (x % 1 > 0.5)):
        return int(np.ceil(x))
    else:
        return int(np.floor(x))


def pick_four_unique_nodes_quickly(n, seed=None):
    '''
    This is equivalent to np.random.choice(n, 4, replace=False)

    Another fellow suggested np.random.random_sample(n).argpartition(4) which is
    clever but still substantially slower.
    '''
    rng = get_rng(seed)
    k = rng.randint(n**4)
    a = k % n
    b = k // n % n
    c = k // n ** 2 % n
    d = k // n ** 3 % n
    if (a != b and a != c and a != d and b != c and b != d and c != d):
        return (a, b, c, d)
    else:
        # the probability of finding a wrong configuration is extremely low
        # unless for extremely small n. if n is extremely small the
        # computational demand is not a problem.

        # In my profiling it only took 0.4 seconds to include the uniqueness
        # check in 1 million runs of this function so I think it is OK.
        return pick_four_unique_nodes_quickly(n, rng)


def cuberoot(x):
    '''
    Correctly handle the cube root for negative weights, instead of uselessly
    crashing as in python or returning the wrong root as in matlab
    '''
    return np.sign(x) * np.abs(x)**(1 / 3)


def dummyvar(cis, return_sparse=False):
    '''
    This is an efficient implementation of matlab's "dummyvar" command
    using sparse matrices.

    input: partitions, NxM array-like containing M partitions of N nodes
        into <=N distinct communities

    output: dummyvar, an NxR matrix containing R column variables (indicator
        variables) with N entries, where R is the total number of communities
        summed across each of the M partitions.

        i.e.
        r = sum((max(len(unique(partitions[i]))) for i in range(m)))
    '''
    # num_rows is not affected by partition indexes
    n = np.size(cis, axis=0)
    m = np.size(cis, axis=1)
    r = np.sum((np.max(len(np.unique(cis[:, i])))) for i in range(m))
    nnz = np.prod(cis.shape)

    ix = np.argsort(cis, axis=0)
    # s_cis=np.sort(cis,axis=0)
    # FIXME use the sorted indices to sort by row efficiently
    s_cis = cis[ix][:, range(m), range(m)]

    mask = np.hstack((((True,),) * m, (s_cis[:-1, :] != s_cis[1:, :]).T))
    indptr, = np.where(mask.flat)
    indptr = np.append(indptr, nnz)

    import scipy.sparse as sp
    dv = sp.csc_matrix((np.repeat((1,), nnz), ix.T.flat, indptr), shape=(n, r))
    return dv.toarray()


def get_rng(seed=None):
    """
    By default, or if `seed` is np.random, return the global RandomState
    instance used by np.random.
    If `seed` is a RandomState instance, return it unchanged.
    Otherwise, use the passed (hashable) argument to seed a new instance
    of RandomState and return it.

    Parameters
    ----------
    seed : hashable or np.random.RandomState or np.random, optional

    Returns
    -------
    np.random.RandomState
    """
    if seed is None or seed == np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, np.random.RandomState):
        return seed
    try:
        rstate =  np.random.RandomState(seed)
    except ValueError:
        rstate = np.random.RandomState(random.Random(seed).randint(0, 2**32-1))
    return rstate
