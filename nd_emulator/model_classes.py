import numpy as np
from itertools import permutations


def fit_nd_linear_model(F, X):
    """
    :param F: (nd-array) function values
    :param X: (2 arrays) the coordinates of the lower and upper corner of the hypercube over which
        F is defined, ordered as [[x0_0, x1_0, ..., xN_0], [x0_N, ...xN_N]]
    :return: (array) weights
    """
    dims = F.ndim
    weights = np.zeros([2**dims + 2*dims])
    # save corners of the hypercube
    for i in range(2**dims):
        index = tuple([int(d)*-1 for d in f'{i:0{dims}b}'[::-1]])
        weights[i] = F[index]
    # save two corner coordinates
    weights[-2*dims:-dims] = X[0][:]
    weights[-dims:] = X[1][:]
    return weights


def nd_linear_model(weights, x):
    """
    do an ND-linear interpolation at the point x given model weight values.
    :param weights:
    :param x:
    :return:
    """

    return