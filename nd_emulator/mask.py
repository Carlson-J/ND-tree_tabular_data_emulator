import numpy as np
from numba import jit

DOMAIN_ROUNDING_MAP = [None, 'expand', 'contract']
SPACING_OPTIONS = ['linear', 'log']


def get_mask_dims(mask):
    """
    Compute the dims of a given mask
    :param mask:
    :return: dims
    """
    if mask is None:
        return None
    num_dims = len(mask)
    dims = np.zeros([num_dims], dtype=int)
    for j in range(num_dims):
        dims[j] = mask[j].stop - mask[j].start
    return dims


def create_mask(sub_domain, domain, dims, spacings, domain_rounding_type=None):
    """
            A mask for nd data. It is assumed that the data is evenly spaced after the transform has been done.
            For sub_domains that are entirely outside the domain, None is returned for the mask.
            For sub_domains that are partially outside the domain, the intersection of the two are returned.
            For sub_domains that condain only an edge, i.e., the hyper-volume is zero, None is returned
            :param sub_domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
                dimension for the new domain.
            :param domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
                dimension for the full domain.
            :param dims: (list) the number of elements along each axis of the data matrix
            :param spacings: (list) how the points are spaced in each dimension.
            :param domain_rounding_type: [None, 'expand', 'contract'] How to handle subdomains that do not line up with
                points. expand will round the index to the extreme and contract will do the opposite.
            :return (tuple) of intervals or None if the intersection between domains is the empty set or None
            """
    EPS = 10 ** -10
    # Make sure the dimensions are correct
    assert (len(domain) == len(sub_domain) and len(domain) == len(spacings) and len(domain) == len(dims))
    num_dims = len(spacings)
    # get domain intersection
    d_sub = sub_domain.copy()
    d = domain.copy()
    index_ranges = np.zeros([num_dims, 2], dtype=int)  # this will be changed
    index_ranges = get_mask_indices(index_ranges, num_dims, np.array(d_sub), np.array(sub_domain), np.array(domain),
                                    np.array(d)
                                    , dims, DOMAIN_ROUNDING_MAP.index(domain_rounding_type), EPS,
                                    np.array([SPACING_OPTIONS.index(s) for s in spacings]))
    if index_ranges is None:
        return None
    mask = []
    for i in range(len(index_ranges)):
        mask.append(slice(index_ranges[i][0], index_ranges[i][1]))
    mask = tuple(mask)
    return mask


@jit(nopython=True)
def get_mask_indices(index_ranges, num_dims, d_sub, sub_domain, domain, d, dims, domain_rounding_type, EPS, spacings):
    for i in range(num_dims):
        # only keep intersection range
        d_sub[i][0] = max(sub_domain[i][0], domain[i][0])
        d_sub[i][1] = min(sub_domain[i][1], domain[i][1])
        # return None if there is no intersection
        if d_sub[i][0] >= d_sub[i][1]:
            return None
    # do needed spacings
    for i in range(num_dims):
        if spacings[i] == 0:
            continue
        elif spacings[i] == 1:
            # assert (np.all(np.array(domain[i]) > 0))
            d[i] = np.log10(domain[i])
            d_sub[i] = np.log10(d_sub[i])

    # compute index ranges
    for i in range(num_dims):
        dx = (d[i][1] - d[i][0]) / (dims[i] - 1)
        lo_tmp = (d_sub[i][0] - d[i][0]) / dx
        hi_tmp = (d_sub[i][1] - d[i][0]) / dx
        if domain_rounding_type == 0:
            lo = int(np.around(lo_tmp))
            if abs(lo - lo_tmp) > EPS:
                raise RuntimeError
            hi = int(np.around(hi_tmp))
            if abs(hi - hi_tmp) > EPS:
                raise RuntimeError
        elif domain_rounding_type == 1:
            lo = int(np.floor(lo_tmp))
            hi = int(np.ceil(hi_tmp))
        elif domain_rounding_type == 2:
            lo = int(np.ceil(lo_tmp))
            hi = int(np.floor(hi_tmp))
        # return None if one dim is flat
        if hi == lo:
            return None
        index_ranges[i] = [lo, hi + 1]
    return index_ranges
