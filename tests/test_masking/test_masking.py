from nd_emulator.mask import create_mask
import numpy as np
import pytest


@pytest.fixture
def domain_2d():
    domain = [[-1, 1], [-2, 2]]
    dims = [21, 41]
    # create dataset
    x = np.linspace(domain[0][0], domain[0][1], dims[0])
    y = np.linspace(domain[1][0], domain[1][1], dims[1])
    X, Y = np.meshgrid(x, y)
    Z = X + Y
    return X, Y, Z, domain, dims


@pytest.fixture
def domain_2d_log():
    domain = [[-1, 1], [1e-2, 1e2]]
    dims = [21, 41]
    # create dataset
    x = np.linspace(domain[0][0], domain[0][1], dims[0])
    y = np.logspace(np.log10(domain[1][0]), np.log10(domain[1][1]), dims[1])
    X, Y = np.meshgrid(x, y)
    Z = X + Y
    return X, Y, Z, domain, dims


def test_create_mask_no_rounding(domain_2d):
    """
    GIVEN: linear domains with no rounding
    WHEN: create a mask function
    THEN: function returns the correct subset of data
    """
    X, Y, Z, domain, dims = domain_2d
    # --- Test domain_rounding_type = None --- #
    # create subdomain
    sub_domain = [[-0.5, 0.5], [-1, 1]]
    spacings = ['linear', 'linear']
    mask = create_mask(sub_domain, domain, dims, spacings, domain_rounding_type=None)
    # check if mask worked
    assert (np.all(Z[5:16, 10:31] == Z[mask]))
    # make sure the exact gives the expected error
    with pytest.raises(RuntimeError):
        sub_domain = [[-0.499, 0.499], [-1, 1]]
        spacings = ['linear', 'linear']
        mask = create_mask(sub_domain, domain, dims, spacings, domain_rounding_type=None)


def test_create_mask_expanding(domain_2d):
    """
    GIVEN: linear domains with expanding rounding
    WHEN: create a mask function
    THEN: function returns the correct subset of data
    """
    X, Y, Z, domain, dims = domain_2d
    # create subdomain
    sub_domain = [[-0.499, 0.499], [-1, 1]]
    spacings = ['linear', 'linear']
    mask = create_mask(sub_domain, domain, dims, spacings, domain_rounding_type='expand')
    # check if mask worked
    assert (np.all(Z[5:16, 10:31] == Z[mask]))


def test_create_mask_contracting(domain_2d):
    """
    GIVEN: linear domains with contracting rounding
    WHEN: create a mask function
    THEN: function returns the correct subset of data
    """
    X, Y, Z, domain, dims = domain_2d
    # create subdomain
    sub_domain = [[-0.499, 0.499], [-1, 1]]
    spacings = ['linear', 'linear']
    mask = create_mask(sub_domain, domain, dims, spacings, domain_rounding_type='contract')
    # check if mask worked
    assert (np.all(Z[6:15, 10:31] == Z[mask]))


def test_create_mask_log(domain_2d_log):
    """
    GIVEN: log domains with no rounding
    WHEN: create a mask function
    THEN: function returns the correct subset of data
    """
    X, Y, Z, domain, dims = domain_2d_log
    # create subdomain
    sub_domain = [[-0.5, 0.5], [1e-1, 1e1]]
    spacings = ['linear', 'log']
    mask = create_mask(sub_domain, domain, dims, spacings, domain_rounding_type=None)
    # check if mask worked
    assert (np.all(Z[5:16, 10:31] == Z[mask]))
