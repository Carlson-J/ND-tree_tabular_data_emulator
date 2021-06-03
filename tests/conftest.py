import pytest
import numpy as np
from nd_emulator import create_mask


@pytest.fixture
def dataset_2d():
    # Create a 2D grid spaced in linear space
    domain = [[0, 1], [0, 1]]  # don't change this or it will break tests
    dims = [2 ** 3 + 1, 2 ** 4 + 1]
    spacing = ['linear', 'linear']
    x0 = np.linspace(domain[0][0], domain[0][1], dims[0])
    x1 = np.linspace(domain[1][0], domain[1][1], dims[1])
    X0, X1 = np.meshgrid(x0, x1, indexing='ij')

    return {
               'f': X0 * X1,
               'df_x0': X1,
               'df_x1': X0,
               'df_x0_x1': np.ones_like(X1)
           }, domain, spacing


@pytest.fixture
def dataset_2d_log():
    # Create a 2D grid spaced in linear space
    domain = [[0, 1], [1e-0, 1e1]]  # don't change this or it will break tests
    dims = [2 ** 3 + 1, 2 ** 4 + 1]
    spacing = ['linear', 'log']
    x0 = np.linspace(domain[0][0], domain[0][1], dims[0])
    x1 = np.logspace(np.log10(domain[1][0]), np.log10(domain[1][1]), dims[1])
    X0, X1 = np.meshgrid(x0, x1, indexing='ij')

    return {
               'f': X0 * X1,
               'df_x0': X1,
               'df_x1': X0,
               'df_x0_x1': np.ones_like(X1)
           }, domain, spacing


@pytest.fixture
def dataset_2d_non_linear():
    # Create a 2D grid spaced in linear space
    domain = [[0, 1], [0, 1]]  # don't change this or it will break tests
    dims = [2 ** 6 + 1, 2 ** 6 + 1]
    spacing = ['linear', 'linear']
    x0 = np.linspace(domain[0][0], domain[0][1], dims[0])
    x1 = np.linspace(domain[1][0], domain[1][1], dims[1])
    X0, X1 = np.meshgrid(x0, x1, indexing='ij')

    return {
               'f': np.cos(X0)*2 + np.sin(X1)
           }, domain, spacing


@pytest.fixture
def dataset_2d_log_non_linear():
    # Create a 2D grid spaced in linear space
    domain = [[0, 1], [1e0, 1e2]]  # don't change this or it will break tests
    dims = [2 ** 6 + 1, 2 ** 6 + 1]
    spacing = ['linear', 'log']
    x0 = np.linspace(domain[0][0], domain[0][1], dims[0])
    x1 = np.logspace(np.log10(domain[1][0]), np.log10(domain[1][1]), dims[1])
    X0, X1 = np.meshgrid(x0, x1, indexing='ij')

    return {
               'f': X0 * np.log10(X1)
           }, domain, spacing


@pytest.fixture
def dataset_4d():
    domain = [[0, 1], [0, 2], [0, 3], [0, 4]]
    dims = [2 ** 3 + 1, 2 ** 4 + 1, 2 ** 4 + 1, 2 ** 4 + 1]
    spacing = ['linear', 'linear', 'linear', 'linear']
    X = []
    for i in range(len(dims)):
        X.append(np.linspace(domain[i][0], domain[i][1], dims[i]))
    F = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for z in range(dims[3]):
                    F[i, j, k, z] = X[0][i] + X[1][j] + X[2][k] + X[3][z] + 1.
    return {
               'f': F,
           }, domain, spacing


@pytest.fixture
def dataset_4d_log():
    # Create a 2D grid spaced in linear space
    domain = [[0, 1], [1e-0, 1e1], [0, 1], [1, 2]]  # don't change this or it will break tests
    dims = [2 ** 3 + 1, 2 ** 4 + 1, 2 ** 4 + 1, 2 ** 4 + 1]
    spacing = ['linear', 'log', 'linear', 'linear']
    F = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for z in range(dims[3]):
                    F[i, j, k, z] = i + j + k + z + 1.
    return {
               'f': F,
           }, domain, spacing


@pytest.fixture
def dataset_4d_log_non_linear():
    # Create a 2D grid spaced in linear space
    domain = [[0, 1], [1e-0, 1e1], [0, 1], [1, 2]]
    # don't change this or it will break tests
    dims = [2 ** 5 + 1, 2 ** 5 + 1, 2 ** 5 + 1, 2 ** 5 + 1]
    X = []
    for i in range(len(dims)):
        X.append(np.linspace(domain[i][0], domain[i][1], dims[i]))
    spacing = ['linear', 'linear', 'linear', 'linear']
    F = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for z in range(dims[3]):
                    F[i, j, k, z] = X[0][i]**2 * X[1][j] * X[2][k] * X[3][z] + 1.
    return {
               'f': F,
           }, domain, spacing


@pytest.fixture
def default_root_node_2d(dataset_2d):
    F, domain, spacing = dataset_2d
    return {
        'domain': domain,
        'children': None,
        'id': [0],
        'model': None,
        'mask': create_mask(domain, domain, F['f'].shape, spacing),
        'error': None
    }


@pytest.fixture
def default_root_node_2d_log(dataset_2d_log):
    F, domain, spacing = dataset_2d_log
    return {
               'domain': domain,
               'children': None,
               'id': [0],
               'model': None,
               'mask': create_mask(domain, domain, F['f'].shape, spacing),
               'error': None
           }, spacing
