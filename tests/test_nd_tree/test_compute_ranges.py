import numpy as np
from nd_emulator.nd_tree import compute_ranges


def test_compute_ranges():
    """
    GIVEN: the domain, dims, and spacing
    WHEN: compute list of values along each axis corresponding to the point intervals
    THEN: intervals are correct
    """
    EPS = 10**-15
    # setup
    dims = [15, 7]
    domain = [[-2,10], [1e-4, 1e5]]
    spacing = ['linear', 'log']
    x = np.linspace(domain[0][0], domain[0][1], dims[0])
    y = np.logspace(np.log10(domain[1][0]), np.log10(domain[1][1]), dims[1])

    # get ranges
    ranges = compute_ranges(domain, spacing, dims)

    assert(np.all(abs(ranges[0] - x) < EPS))
    assert(np.all(abs(ranges[1] - y) < EPS))
