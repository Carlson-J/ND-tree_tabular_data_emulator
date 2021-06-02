import numpy as np


def load_test_data(shape, domain):
    """
    This function creates data for the function f(x[i,:]) = product([x[i,j] for j in range(len(shape))])
    given the shape and domain.
    :param shape: (array) Each entry is the number of points in each dimension, with the length of the array
        being the number of dimensions. Each entry must equal 2^a+1, for some integer a.
    :param domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
        dimension.
    :return:
    """
    domain = np.array(domain)
    # check inputs
    for num_points in shape:
        assert (((num_points-1) & (num_points - 2) == 0) and (num_points-1) != 0)
    assert (len(shape) == len(domain))
    assert (domain.ndim == 2)
    assert (domain.shape[1] == 2)
    for a in domain:
        assert (a[0] < a[1])

    # Create test points.
    ranges = [np.linspace(*domain[i], shape[i]) for i in range(len(shape))]
    X = np.meshgrid(*ranges, indexing='ij')

    # create output
    f = np.prod(X, axis=0)

    # pack into dict
    sol = {'f': f}

    return sol
