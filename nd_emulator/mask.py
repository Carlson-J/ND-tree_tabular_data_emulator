import numpy as np


def get_mask_dims(mask):
    """
    Compute the dims of a given mask
    :param mask:
    :return: dims
    """
    num_dims = len(mask)
    dims = np.zeros([num_dims], dtype=int)
    for j in range(num_dims):
        dims[j] = mask[j].stop - mask[j].start
    return dims


def create_mask(sub_domain, domain, dims, spacings, domain_rounding_type=None):
    """
            A mask for nd data. It is assumed that the data is evenly spaced after the transform has been done.
            :param sub_domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
                dimension for the new domain. Should be contained inside full domain
            :param domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
                dimension for the full domain.
            :param dims: (list) the number of elements along each axis of the data matrix
            :param spacings: (list) how the points are spaced in each dimension.
            :param domain_rounding_type: [None, 'expand', 'contract'] How to handle subdomains that do not line up with
                points. expand will round the index to the extreme and contract will do the opposite.
            """
    EPS = 10 ** -10
    # Make sure the sub-domain is within the original domain
    assert (len(domain) == len(sub_domain) and len(domain) == len(spacings) and len(domain) == len(dims))
    num_dims = len(spacings)
    for i in range(num_dims):
        assert (domain[i][0] <= sub_domain[i][0] and domain[i][1] >= sub_domain[i][1])

    # do needed spacings
    d = domain.copy()
    d_sub = sub_domain.copy()
    for i in range(num_dims):
        if spacings[i] == 'linear':
            continue
        elif spacings[i] == 'log':
            assert (np.all(np.array(domain[i]) > 0))
            d[i] = np.log10(domain[i])
            d_sub[i] = np.log10(sub_domain[i])

    # compute index ranges
    index_ranges = np.zeros([num_dims, 2], dtype=int)
    for i in range(num_dims):
        dx = (d[i][1] - d[i][0]) / (dims[i] - 1)
        lo_tmp = (d_sub[i][0] - d[i][0]) / dx
        hi_tmp = (d_sub[i][1] - d[i][0]) / dx
        if domain_rounding_type is None:
            lo = int(np.around(lo_tmp))
            if abs(lo - lo_tmp) > EPS:
                raise RuntimeError
            hi = int(np.around(hi_tmp))
            if abs(hi - hi_tmp) > EPS:
                raise RuntimeError
        elif domain_rounding_type == 'expand':
            lo = int(np.floor(lo_tmp))
            hi = int(np.ceil(hi_tmp))
        elif domain_rounding_type == 'contract':
            lo = int(np.ceil(lo_tmp))
            hi = int(np.floor(hi_tmp))
        else:
            raise ValueError(f"Unknown domain_rounding_type: {domain_rounding_type}")
        index_ranges[i] = [lo, hi + 1]
    mask = []
    for i in range(len(index_ranges)):
        mask.append(slice(index_ranges[i][0], index_ranges[i][1]))
    mask = tuple(mask)
    return mask
