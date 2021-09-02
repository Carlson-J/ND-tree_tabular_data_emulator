import numpy as np
from itertools import permutations
from numba import jit


def fit_nd_linear_model(F, X, transforms: object = None):
    """
    :param transforms: (str or None) transform to do on value before fit.
    :param F: (nd-array) function values
    :param X: (2 arrays) the coordinates of the lower and upper corner of the hypercube over which
        F is defined, ordered as [[x0_0, x1_0, ..., xN_0], [x0_N, ...xN_N]]
    :return: (array) weights [f_0...0, f_10...0, f_010..0, ..., f_1...1, x0_0..0, x1_0...0, ..., xN_0...0, x0_1...1,
        ..., xN_1...1]
    """
    dims = F.ndim
    transform_var = 0
    if transforms is not None:
        # add an additional var for the transform variable
        transform_num_vars = 1
    else:
        transform_num_vars = 0
    weights = np.zeros([2 ** dims + 2 * dims + transform_num_vars])
    # save corners of the hypercube
    for i in range(2 ** dims):
        index = tuple([int(d) * -1 for d in f'{i:0{dims}b}'[::-1]])
        weights[i+transform_num_vars] = F[index]
    # do any needed transforms on function values
    f = weights[transform_num_vars:2**dims+transform_num_vars]   # alis part of the weights the contain f
    # handle case where f is only zeros
    if transforms == 'log':
        weights[:transform_num_vars] = compute_log_transform_weight(f)
        weights[transform_num_vars:2 ** dims + transform_num_vars] = np.log10(weights[transform_num_vars:2 ** dims + transform_num_vars])

    elif transforms is not None:
        raise ValueError(f"No transform of type '{transforms}' found")
    # save two corner coordinates
    weights[-2 * dims:-dims] = X[0][:]
    weights[-dims:] = X[1][:]
    return weights


def compute_log_transform_weight(f):
    """
    Finds the variable needed to transform the node values so that they are all positive.
    :param f: (1d array) array of values in a call/node. These values may be modified!
    :return: (double) transform var
    """
    transform_var = 0
    if np.all(f == 0):
        transform_var = 0
    else:
        # determine if it should be multiplied by a sign
        if abs(np.min(f)) > abs(np.max(f)):
            # do reflection
            f[:] = f[:] * -1
            transform_var = -1
        # shift values to be all positive
        if np.min(f) == 0:
            transform_var = np.max(f)
            f[:] += np.max(f)
        elif np.min(f) < 0:
            if transform_var == 0:
                transform_var = -2 * np.min(f)
            else:
                transform_var *= -2 * np.min(f)
            f[:] += -2 * np.min(f)
    return transform_var

def nd_linear_model(weights, X, transform=None):
    """
    Wrapper for _nd_linear_model that takes into account any transforms
    :param weights:[f_0...0, f_10...0, f_010..0, ..., f_1...1, x0_0..0, x1_0...0, ..., xN_0...0, x0_1...1,
        ..., xN_1...1]
    :param X: (2d array) Points to interpolate at. We assume x and weights are in correct format. Each row is
        a different point.
    :param transform: (string or None) transform that is do after interpolation
    :return:
    """
    if transform is None:
        return _nd_linear_model(weights, X)
    elif transform == 'log':
        # do linear fit and then the transform
        sol = _nd_linear_model(weights[1:], X)
        sol = 10**sol
        if weights[0] != 0:
            sol -= abs(weights[0])
            sol *= np.sign(weights[0])
    else:
        raise ValueError(f"No transform '{transform}' found.")
    return sol


@jit(nopython=True)
def _nd_linear_model(weights, X):
    """
    do an ND-linear interpolation at the points in X given model weight values.
    https://math.stackexchange.com/a/1342377
    :param weights:[f_0...0, f_10...0, f_010..0, ..., f_1...1, x0_0..0, x1_0...0, ..., xN_0...0, x0_1...1,
        ..., xN_1...1]
    :param X: (2d array) Points to interpolate at. We assume x and weights are in correct format. Each row is
        a different point.
    :return:
    """
    X = np.atleast_2d(X)
    dims = int(len(X[0, :]))
    sol = np.zeros_like(X[:, 0], dtype=np.float64)
    for k, x in enumerate(X):
        # transform x based on domain of hypercube
        x = (x - weights[2 ** dims:2 ** dims + dims]) / (
                    weights[2 ** dims + dims:2 ** dims + 2 * dims] - weights[2 ** dims:2 ** dims + dims])
        # iterate over each corner of hypercube
        for i in range(2 ** dims):
            w = 1
            # get weight for corner
            for j in range(dims):
                # get binary index of current corner
                bit = (i >> j) & 1
                w *= 1 - x[j] if bit == 0 else x[j]
            sol[k] += weights[i] * w
    return sol
