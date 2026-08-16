import numpy as np


def BlockMean(X, V, W=None):
    """
    2D block mean over 1st and 2nd dimensions.
    The mean of V*W elements along the 1st and 2nd dimensions is calculated.

    Parameters
    ----------
    X : np.ndarray
        UINT8 or DOUBLE (float64) array of any size.
    V : int
        Block size along the 1st dimension.
    W : int, optional
        Block size along the 2nd dimension. If omitted, a square V*V block is used.

    Returns
    -------
    Y : np.ndarray
        Array of same dtype as X. The 1st and 2nd dimensions are V and W times
        shorter: [floor(rows/V) x floor(cols/W) x (further dims...)].
        If the size of the 1st or 2nd dimension is not a multiple of V or W,
        the remaining elements at the end are skipped.
        The empty array is returned for empty inputs or if the 1st or 2nd
        dimension is shorter than V or W.

    Notes
    -----
    Translated from MATLAB (Jan Simon, 2009-2010). Supports float64 and uint8.
    MATLAB reshape is column-major (Fortran order); numpy reshape uses order='F'
    to match exactly.
    """
    if W is None:
        W = V

    V = int(V)
    W = int(W)

    S = list(X.shape)
    if len(S) < 2:
        raise ValueError("X must be at least 2-dimensional")

    M = S[0] - (S[0] % V)
    N = S[1] - (S[1] % W)

    if M * N == 0:
        return X.flat[[]]  # empty, preserving type

    MV = M // V
    NW = N // W

    # Slice input to multiples of V and W along first two dims
    extra_dims = S[2:] if len(S) > 2 else []
    X_cut = X[:M, :N]

    # MATLAB: reshape(X(1:M, 1:N, :), V, MV, W, NW, [])
    # MATLAB uses column-major (Fortran) order -- must use order='F' in numpy
    new_shape = [V, MV, W, NW] + (extra_dims if extra_dims else [])
    XM = X_cut.reshape(new_shape, order='F')

    if X.dtype == np.float64:
        # MATLAB: sum(sum(XM, 1), 3) .* (1.0 / (V * W))
        # sum along dim 1 (axis 0 in 0-based) then dim 3 (axis 2 after first reduction)
        Y = XM.sum(axis=0).sum(axis=1) * (1.0 / (V * W))
    elif X.dtype == np.uint8:
        # MATLAB uint8 path: round then cast
        total = XM.sum(axis=0).sum(axis=1)
        Y = np.uint8(np.round(total / (V * W)))
    else:
        # Fallback: cast to float64
        X = X.astype(np.float64)
        return BlockMean(X, V, W)

    # Restore shape metadata: S[0]=MV, S[1]=NW, rest unchanged
    S[0] = MV
    S[1] = NW
    Y = Y.reshape(S)

    return Y
