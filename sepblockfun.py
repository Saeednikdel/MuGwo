import numpy as np


def sepblockfun(X, blockdims, fun):
    """
    Perform a separable operation on sub-blocks of an input array.
    Here, a function op() is said to be separable if for any array B with
    elements B(i,j,k,...), the operation op(B(:)), resulting in a scalar,
    can be equivalently done by applying op() first along i, then along j,
    then along k, etc...

    Parameters
    ----------
    X : np.ndarray
        A full array.
    blockdims : list or array-like
        A vector of integers specifying the dimensions of the sub-blocks.
        The array X must partition evenly into blocks of this size.
        If blockdims[i] is set to np.inf then it will be replaced with
        blockdims[i] = X.shape[i].
    fun : callable or str
        Function handle to an operation assumed to be separable
        (Examples: max, min, sum, prod, mean, etc...). The function must
        accept the input syntax fun(B, axis) where B is an input array
        and axis is an axis along which to operate. Alternatively, fun
        can be one of the following strings: 'max', 'min', 'sum', 'mean', 'prod'.

    Returns
    -------
    X : np.ndarray
        The output array. Y[i] = fun(Xi.ravel()) where Xi is the i-th
        sub-block of the input array X.

    Examples
    --------
    # Divide a 400x400x400 array into 10x10x10 blocks.
    A = np.random.rand(400, 400, 400)
    Ameans = sepblockfun(A, [10, 10, 10], 'mean')
    Amins  = sepblockfun(A, [10, 10, 10], 'min')
    Amaxs  = sepblockfun(A, [10, 10, 10], 'max')

    # blockwise standard deviations
    Astds = np.sqrt(sepblockfun(A**2, [10, 10, 10], 'mean') - Ameans**2)
    """
    # Map string names to numpy axis-aware functions
    if isinstance(fun, str):
        fun_map = {
            'max':  lambda b, axis: np.max(b, axis=axis),
            'min':  lambda b, axis: np.min(b, axis=axis),
            'sum':  lambda b, axis: np.sum(b, axis=axis),
            'mean': lambda b, axis: np.mean(b, axis=axis),
            'prod': lambda b, axis: np.prod(b, axis=axis),
        }
        if fun not in fun_map:
            raise ValueError(f"Unrecognized fun() selection: '{fun}'")
        fun = fun_map[fun]

    blockdims = list(blockdims)
    X = np.array(X)

    nn = max(len(blockdims), X.ndim)

    # Pad blockdims with 1s if shorter than number of dimensions
    while len(blockdims) < nn:
        blockdims.append(1)

    sz = list(X.shape)
    # Pad sz with 1s if X has fewer dims than blockdims
    while len(sz) < nn:
        sz.append(1)
        X = np.expand_dims(X, axis=-1)

    # Replace Inf entries with full dimension size (equivalent to MATLAB's ~isfinite)
    for i in range(nn):
        if not np.isfinite(blockdims[i]):
            blockdims[i] = sz[i]

    newdims = [sz[i] // blockdims[i] for i in range(nn)]

    # Build reshape args: interleave blockdims and newdims
    # MATLAB: reshape(X, [blockdims(1), newdims(1), blockdims(2), newdims(2), ...])
    # MATLAB reshape is column-major (Fortran order) -- use order='F' in numpy
    reshape_args = []
    for i in range(nn):
        reshape_args.append(int(blockdims[i]))
        reshape_args.append(int(newdims[i]))

    X = X.reshape(reshape_args, order='F')

    # Apply fun along each "block" axis.
    # After reshape to [b1,n1,b2,n2,...], MATLAB reduces axis 2*ii-1 (1-based)
    # = axis 2*ii (0-based) with keepdims. Without keepdims, after ii reductions
    # the target block axis is always at position ii (axes shift down by 1 each time).
    for ii in range(nn):
        X = fun(X, ii)

    X = X.reshape(newdims)
    return X
