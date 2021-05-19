import numpy as np
from itertools import permutations
from numba import jit


def fit_nd_linear_model(F, X):
    """
    :param F: (nd-array) function values
    :param X: (2 arrays) the coordinates of the lower and upper corner of the hypercube over which
        F is defined, ordered as [[x0_0, x1_0, ..., xN_0], [x0_N, ...xN_N]]
    :return: (array) weights [f_0...0, f_10...0, f_010..0, ..., f_1...1, x0_0..0, x1_0...0, ..., xN_0...0, x0_1...1,
        ..., xN_1...1]
    """
    dims = F.ndim
    weights = np.zeros([2 ** dims + 2 * dims])
    # save corners of the hypercube
    for i in range(2 ** dims):
        index = tuple([int(d) * -1 for d in f'{i:0{dims}b}'[::-1]])
        weights[i] = F[index]
    # save two corner coordinates
    weights[-2 * dims:-dims] = X[0][:]
    weights[-dims:] = X[1][:]
    return weights


@jit(nopython=True)
def nd_linear_model(weights, X):
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
