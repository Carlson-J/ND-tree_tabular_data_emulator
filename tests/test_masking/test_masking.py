from nd_emulator.mask import create_mask
import numpy as np
import pytest


def test_create_mask():
    """
    GIVEN: the old domain, new domain, and dimensions of the data.
    WHEN: create a mask function
    THEN: function returns the correct subset of data
    """
    domain = [[-1, 1], [-2, 2]]
    dims = [21, 41]
    # create dataset
    x = np.linspace(domain[0][0], domain[0][1], dims[0])
    y = np.linspace(domain[1][0], domain[1][1], dims[1])
    X, Y = np.meshgrid(x, y)
    Z = X+Y
    # --- Test domain_rounding_type = None --- #
    # create subdomain
    sub_domain = [[-0.5, 0.5], [-1, 1]]
    transforms = [None, None]
    mask = create_mask(sub_domain, domain, dims, transforms, domain_rounding_type=None)
    # check if mask worked
    assert(np.all(Z[5:16, 10:31] == Z[mask]))
    # make sure the exact gives the expected error
    with pytest.raises(RuntimeError):
        sub_domain = [[-0.499, 0.499], [-1, 1]]
        transforms = [None, None]
        mask = create_mask(sub_domain, domain, dims, transforms, domain_rounding_type=None)

    # --- Test domain_rounding_type = expand --- #
    # create subdomain
    sub_domain = [[-0.499, 0.499], [-1, 1]]
    transforms = [None, None]
    mask = create_mask(sub_domain, domain, dims, transforms, domain_rounding_type='expand')
    # check if mask worked
    assert (np.all(Z[5:16, 10:31] == Z[mask]))

    # --- Test domain_rounding_type = contract --- #
    # create subdomain
    sub_domain = [[-0.499, 0.499], [-1, 1]]
    transforms = [None, None]
    mask = create_mask(sub_domain, domain, dims, transforms, domain_rounding_type='contract')
    # check if mask worked
    assert (np.all(Z[6:15, 10:31] == Z[mask]))
