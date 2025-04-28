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

    >>> test is needed here?


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


    # generate matrices  
    # convert to z scores 
    # compute -t stat difference
    # using t stats, calculate adjoint matrix
    # get_components 
    # Check instances where compnent_size == 1
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
