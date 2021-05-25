import numpy as np
from nd_emulator.domain_functions import compute_ranges, transform_domain


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


def test_transform_domain():
    """
    GIVEN: a domain and spacings
    WHEN: domain is transforms back and forth
    THEN: correct transformation
    """
    EPS = 10**-15
    domain = np.array([[0, 1], [1e-1, 1e1], [0, 1]])
    spacings = ['linear', 'log', 'linear']
    # transform domain
    new_domain = transform_domain(domain, spacings)
    assert np.all(abs(new_domain - np.array([[0, 1], [-1, 1], [0, 1]])) <= EPS)
    # transform back
    old_domain = transform_domain(new_domain, spacings, reverse=True)
    assert np.all(abs(old_domain - domain) <= EPS)
