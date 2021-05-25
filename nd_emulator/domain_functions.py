import numpy as np


def compute_ranges(domain, spacing, dims):
    """
    Create an array for each axis that contains the intervals along that axis
    :param domain:  (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
            dimension.
    :param spacing: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :param dims: [(int),...,(int)] the number of elements in each dimension
    :return: (list) a list of arrays that hold the point locations along the axis corresponding to the index in the list
    """
    ranges = []
    for i in range(len(dims)):
        if spacing[i] == 'linear':
            ranges.append(np.linspace(domain[i][0], domain[i][1], dims[i]))
        elif spacing[i] == 'log':
            ranges.append(np.logspace(np.log10(domain[i][0]), np.log10(domain[i][1]), dims[i]))
        else:
            raise ValueError(f"spacing type '{spacing[i]}' unknown")
    return ranges


def transform_domain(domain, spacings, reverse=False):
    """
    Transform the domain so that the spacing is linear, e.g., for log spacing we take the log. For linear we do nothing.
    The original input array is not modified
    :param domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
            dimension.
    :param spacings: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :param reverse: undo previous transforms. For linear, nothing is done, for log, we compute 10^()
    :return: transformed domain
    """
    new_domain = domain.copy()
    for i in range(len(spacings)):
        if spacings[i] == 'linear':
            continue
        elif spacings[i] == 'log':
            if reverse:
                new_domain[i] = 10**(new_domain[i])
            else:
                new_domain[i] = np.log10(new_domain[i])
    return new_domain


def transform_point(point, spacings, reverse=False):
    """
    Transform the point so that the spacing is linear, e.g., for log spacing we take the log. For linear we do nothing.
    The original input array is not modified
    :param point: (array) [x0, x1, ...]
    :param spacings: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :param reverse: (bool) undo spacing transform
    :return: transformed point
    """
    new_point = point.copy()
    for i in range(len(spacings)):
        if spacings[i] == 'linear':
            continue
        elif spacings[i] == 'log':
            if reverse:
                new_point[i] = 10 ** (new_point[i])
            else:
                new_point[i] = np.log10(new_point[i])
    return new_point