from nd_emulator import build_emulator, EmulatorCpp, load_emulator, make_cpp_emulator
from data_loading_functions import load_test_data
import nd_emulator
import pytest
import numpy as np
import matplotlib.pyplot as plt
from time import time
import subprocess
import sys
import shutil
import os


@pytest.mark.dependency()
def test_init(dataset_2d):
    """
    GIVEN: nd tree paramaters
    WHEN: tree is built
    THEN: No errors occur
    """
    model_classes = [{'type': 'nd-linear'}]
    max_depth = 3
    error_threshold = 0.1
    data, domain, spacing = dataset_2d
    try:
        build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    except Exception as err:
        assert False, f'initializing build_emulator raised an exception: {err}'


def test_2d_interpolation(dataset_2d_log):
    """
    GIVEN: 2d function evaluations and a domain.
    WHEN: Create an emulator from data
    THEN: Correctly sorts and interpolates data
    """
    EPS = 10 ** -14
    N = 200
    data, domain, spacing = dataset_2d_log
    error_threshold = 0
    max_depth = 2
    model_classes = [{'type': 'nd-linear', 'transforms': None}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    # Compute new values over domain
    X, Y = np.meshgrid(np.linspace(domain[0][0], domain[0][1], N), np.logspace(np.log10(domain[1][0])
                                                                               , np.log10(domain[1][1]), N))
    input = np.array([X.flatten(), Y.flatten()]).T
    f_interp = emulator(input).reshape([N, N])
    f_true = X * Y
    error = abs(f_true - f_interp)
    # resize and plot
    plt.imshow(error, origin='lower')
    plt.title("Should not see any grid structure")
    plt.colorbar()
    plt.show()

    # check if error is low
    assert np.all(error <= EPS)


def test_basic_4d(dataset_4d):
    """
        GIVEN: 4d data and params for emulator
        WHEN: Emulator is saved, destroyed, and loaded
        THEN: Emulator produces same values before and after
        """
    EPS = 10 ** -14
    SAVE_NAME = 'saved_emulator_4d'
    SAVE_LOC = '.'
    data, domain, spacing = dataset_4d
    error_threshold = 0
    max_depth = 2
    model_classes = [{'type': 'nd-linear'}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    inputs = np.array([
        [0, 0, 0, 0],
        [0.5, 1, 2, 0.1],
        [0.2, 1.3, 0.3, 4.0],
        [1, 1, 1, 1]
    ])
    output = emulator(inputs)
    # test outputs
    for i, point in enumerate(inputs):
        sol = sum(point) + 1
        assert (abs(output[i] - sol) <= EPS)
    # save it
    emulator.save(SAVE_LOC, SAVE_NAME)


def test_saving_nd_tree(dataset_4d_log):
    """
    GIVEN: 4d data and params for emulator
    WHEN: Emulator is saved, destroyed, and loaded
    THEN: Emulator produces same values before and after
    """
    EPS = 10 ** -15
    SAVE_NAME = 'saved_emulator'
    SAVE_LOC = '.'
    data, domain, spacing = dataset_4d_log
    error_threshold = 0
    max_depth = 2
    model_classes = [{'type': 'nd-linear', 'transforms': None}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    inputs = np.random.uniform(0.1, 0.5, size=[100, len(spacing)])
    output_true = emulator(inputs)
    # save it
    emulator.save(SAVE_LOC, SAVE_NAME)


def test_1d_interpolation():
    """
    GIVEN: 1d domain and function values
    WHEN: build tree
    THEN: refines correctly
    """
    # create function and domain
    domain = [[0, 2]]
    x = np.linspace(domain[0][0], domain[0][1], 2 ** 5 + 1)
    y = 3 * x
    y[x > 1] = np.sin(x[x > 1])
    data = {'f': y}
    spacing = ['linear']

    # build tree
    EPS = 10 ** -13
    N = 200
    error_threshold = 1e-2
    max_depth = 5
    model_classes = [{'type': 'nd-linear', 'transforms': None}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    # Compute new values over domain
    x_test = np.linspace(domain[0][0], domain[0][1], N).reshape([N, 1])
    f_interp = emulator(x_test)
    f_true = 3 * x_test
    f_true[x_test > 1] = np.sin(x_test[x_test > 1])
    error = abs(f_true.flatten() - f_interp)
    # get node locations
    cell_locations = emulator.get_cell_locations()
    plt.figure()
    plt.plot(cell_locations.flatten(), min(y) * np.ones_like(cell_locations.flatten()), '*', label='Cell Locations')
    # Plot errors
    plt.plot(x_test, error, '--', label='error')
    plt.plot(x_test, f_true, label='true')
    plt.plot(x_test, f_interp, '--', label='interp')
    plt.plot(x, y, '.', label='data')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def test_depth_control(dataset_2d):
    """
    GIVEN: tree params with excess depth
    WHEN: tree is built
    THEN: depth is changed
    """
    model_classes = [{'type': 'nd-linear'}]
    max_depth = 20
    error_threshold = -1
    data, domain, spacing = dataset_2d
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes, expand_index_domain=True)
    assert (emulator.params.max_depth == 4)


def test_non_aligned_data():
    """
    GIVEN: 1d domain and function values that are regularly spaced but not 2^a + 1 is size
    WHEN: build tree
    THEN: refines correctly
    """
    # create function and domain
    domain = [[0, 2]]
    shape = [2 ** 5 + 10]
    x = np.linspace(domain[0][0], domain[0][1], shape[0])
    y = 3 * x
    y[x > 1] = np.sin(x[x > 1])
    data = {'f': y}
    spacing = ['linear']

    # build tree
    EPS = 10 ** -13
    N = 200
    error_threshold = 1e-1
    max_depth = 6
    model_classes = [{'type': 'nd-linear'}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    # Compute new values over domain
    x_test = np.linspace(domain[0][0], domain[0][1], N).reshape([N, 1])
    f_interp = emulator(x_test)
    f_true = 3 * x_test
    f_true[x_test > 1] = np.sin(x_test[x_test > 1])
    error = abs(f_true.flatten() - f_interp)
    # get node locations
    cell_locations = emulator.get_cell_locations()
    plt.figure()
    plt.plot(x, data['f'], '.g', label='data')
    plt.plot(x, np.zeros_like(data['f']), '.g')
    plt.plot(cell_locations.flatten(), np.zeros_like(cell_locations.flatten()), '*', label='Cell Locations')
    # Plot errors
    plt.plot(x_test, error, '--', label='error')
    plt.plot(x_test, f_true, label='true')
    plt.plot(x_test, f_interp, '--', label='interp')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def test_miss_aligned_2d():
    """
    Given: 2d domain and function that are not aligned with tree
    When: tree is built
    Then: correctly expands cells
    """
    # create function and domain
    domain = [[0, 2], [0, 2]]
    shape = [2 ** 3 + 10, 2 ** 3 + 10]
    x = np.linspace(domain[0][0], domain[0][1], shape[0])
    X, Y = np.meshgrid(x, x)
    Z = 1.0 / (X + Y + 1e-3)
    data = {'f': Z}
    spacing = ['linear', 'linear']

    # build tree
    EPS = 10 ** -13
    N = 50
    error_threshold = 1e-0
    max_depth = 6
    model_classes = [{'type': 'nd-linear'}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    # Compute new values over domain
    x_test = np.linspace(domain[0][0], domain[0][1], N).reshape([N, 1])
    X_test, Y_test = np.meshgrid(x_test, x_test)
    points = np.array([X_test.flatten(), Y_test.flatten()]).T
    f_interp = emulator(points)
    f_true = 1.0 / (X_test + Y_test + 1e-3)
    error = abs(f_true.flatten() - f_interp)
    emulator.save('.', 'miss_aligned_2d')
    # get node locations
    cell_locations = emulator.get_cell_locations()
    plt.figure()
    plt.imshow(f_true, extent=[0, 2, 0, 2], origin='lower')
    plt.scatter(X.flatten(), Y.flatten())
    plt.scatter(cell_locations[:, 0], cell_locations[:, 1], label='Cell Locations')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    # Plot errors
    plt.figure()
    plt.imshow(error.reshape([N, N]), extent=[0, 2, 0, 2], origin='lower')
    plt.colorbar()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def test_miss_aligned_3d():
    """
    GIVEN: 3d data that is not the correct size
    WHEN: Tree is build
    THEN: Correct number of leaves are made
    :return:
    """
    # Create dataset
    domain = [[0, 1], [0, 1], [0, 1]]
    dims = [3, 11, 12]
    x = np.linspace(domain[0][0], domain[0][1], dims[0])
    F = np.zeros(dims)
    F[0, 0, 0] = 100
    data = {'f': F}
    max_depth = 4
    spacing = ['linear', 'linear', 'linear']

    # build emulator
    error_threshold = 1e-10
    model_classes = [{'type': 'nd-linear'}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes,
                              expand_index_domain=True)

    # check for correct number of cells
    correct_num_points = 57
    num_points = len(emulator.point_map)
    assert (num_points == correct_num_points)


def test_miss_aligned_2d_extended():
    """
    Given: 2d domain and function that are not aligned with tree and uneven domain
    When: tree is built
    Then: correctly expands index domain
    """
    # create function and domain
    domain = [[0, 1], [0, 2]]
    shape = [2 ** 2 + 3, 2 ** 3 + 10]
    x = np.linspace(domain[0][0], domain[0][1], shape[0])
    y = np.linspace(domain[1][0], domain[1][1], shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = 1.0 / (X + Y + 1e-3)
    data = {'f': Z}
    spacing = ['linear', 'linear']

    # build tree
    EPS = 10 ** -13
    N = 500
    error_threshold = 1e-0
    max_depth = 6
    model_classes = [{'type': 'nd-linear'}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes,
                              expand_index_domain=True)
    # Compute new values over domain
    x_test = np.linspace(domain[0][0], domain[0][1], N).reshape([N, 1])
    y_test = np.linspace(domain[1][0], domain[1][1], N).reshape([N, 1])
    X_test, Y_test = np.meshgrid(x_test, y_test)
    points = np.array([X_test.flatten(), Y_test.flatten()]).T
    f_interp = emulator(points)
    f_true = 1.0 / (X_test + Y_test + 1e-3)
    error = abs(f_true.flatten() - f_interp)
    emulator.save('.', 'miss_aligned_2d_extended')
    # get node locations
    cell_locations = emulator.get_cell_locations()
    plt.figure()
    plt.scatter(X.flatten(), Y.flatten())
    plt.scatter(cell_locations[:, 0], cell_locations[:, 1], label='Cell Locations')
    plt.imshow(f_true, extent=np.array(domain).flatten(), origin='lower')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')
    # Plot errors
    plt.figure()
    plt.imshow(error.reshape([N, N]), extent=np.array(domain).flatten(), origin='lower')
    plt.colorbar()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def test_linear_log_transform():
    """
    GIVEN: data and log transformed model class
    WHEN: emulator is made
    THEN: correctly interpolates in log space and transforms back.
    """

    # create function and domain
    domain = [[0, 3]]
    x = np.linspace(domain[0][0], domain[0][1], 2 ** 5 + 1)
    y = 3 * x
    y[x > 1] = np.sin(x[x > 1])
    y[x > 2] = 3 ** x[x > 2]
    data = {'f': y}
    spacing = ['linear']

    # build tree
    EPS = 10 ** -2
    N = 200
    error_threshold = 1e-1
    max_depth = 5
    model_classes = [{'type': 'nd-linear', 'transforms': 'log'}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes, relative_error=True)
    # Compute new values over domain
    x_test = np.linspace(domain[0][0], domain[0][1], N).reshape([N, 1])
    f_interp = emulator(x_test)
    f_true = 3 * x_test
    f_true[x_test > 1] = np.sin(x_test[x_test > 1])
    f_true[x_test > 2] = 3**x_test[x_test > 2]
    error = abs(f_true.flatten() - f_interp)
    # get node locations
    cell_locations = emulator.get_cell_locations()
    plt.figure()
    plt.plot(cell_locations.flatten(), min(y) * np.ones_like(cell_locations.flatten()), '*', label='Cell Locations')
    # Plot errors
    plt.plot(x_test, np.log10(error), '--', label=' log error')
    plt.plot(x_test, f_true, label='true')
    plt.plot(x_test, f_interp, '--', label='interp')
    plt.plot(x, y, '.', label='data')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # make sure the errors do not change from where they are right now
    # in each section. Values where computed when things looked right.
    s1_err = np.max(error[(x_test.flatten() < 0.9)])
    s2_err = np.max(error[(x_test.flatten() > 1.1) & (x_test.flatten() < 1.9)])
    s3_err = np.max(error[x_test.flatten() > 2.1])
    assert(s3_err < 10**-13)
    assert(s2_err < 0.02)
    assert(s1_err < 0.2)


def test_2d_log_transforms(dataset_2d_non_linear):
    """
    GIVEN: 2d function evaluations and a domain.
    WHEN: Create an emulator from data with log transforms
    THEN: Correctly sorts and interpolates data

    Note: This data saved here is used by the cpp test. It must be copied to the correct dir to be updated.
    """
    EPS = 0.03
    N = 200
    data, domain, spacing = dataset_2d_non_linear
    error_threshold = 0.01
    max_depth = 2
    model_classes = [{'type': 'nd-linear', 'transforms': 'log'}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    # Compute new values over domain
    X, Y = np.meshgrid(np.linspace(domain[0][0], domain[0][1], N), np.linspace(domain[1][0], domain[1][1], N))
    input = np.array([X.flatten(), Y.flatten()]).T
    f_interp = emulator(input).reshape([N, N])
    f_true = np.cos(X)*2 + np.sin(Y)
    error = abs(f_true - f_interp)
    # resize and plot
    plt.imshow(error, origin='lower')
    plt.title("Grid structure should be seen")
    plt.colorbar()
    plt.show()

    emulator.save('.', 'non_linear2d')
    # check if error is low
    assert np.all(error <= EPS)


def test_linear_model_switching():
    """
    GIVEN: data and log transformed model class
    WHEN: emulator is made
    THEN: correctly interpolates in log space and transforms back.
    """

    # create function and domain
    domain = [[0, 3]]
    x = np.linspace(domain[0][0], domain[0][1], 2 ** 5 + 1)
    y = 3 * x
    y[x > 1] = np.sin(x[x > 1])
    y[x > 2] = 3 ** x[x > 2]
    data = {'f': y}
    spacing = ['linear']

    # build tree
    EPS = 10 ** -2
    N = 200
    error_threshold = 1e-2
    max_depth = 5
    model_classes = [{'type': 'nd-linear', 'transforms': None}, {'type': 'nd-linear', 'transforms': 'log'}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes, relative_error=True)
    # Compute new values over domain
    x_test = np.linspace(domain[0][0], domain[0][1], N).reshape([N, 1])
    f_interp = emulator(x_test)
    f_true = 3 * x_test
    f_true[x_test > 1] = np.sin(x_test[x_test > 1])
    f_true[x_test > 2] = 3**x_test[x_test > 2]
    error = abs(f_true.flatten() - f_interp)
    # get node locations
    cell_locations, model_types = emulator.get_cell_locations(include_model_type=True)
    colors_map = np.array(model_types)[:, 1] == 'log'
    colors_map_i = [not v for v in colors_map]

    plt.figure()
    plt.plot(cell_locations.flatten()[colors_map_i], min(y) * np.ones_like(cell_locations.flatten()[colors_map_i]), 'b*', label='Cell Locations lin')
    plt.plot(cell_locations.flatten()[colors_map], min(y) * np.ones_like(cell_locations.flatten())[colors_map], 'r*', label='Cell Locations log')
    # Plot errors
    plt.plot(x_test, np.log10(error), '--', label=' log error')
    plt.plot(x_test, f_true, label='true')
    plt.plot(x_test, f_interp, '--', label='interp')
    plt.plot(x, y, '.', label='data')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # make sure the errors do not change from where they are right now
    # in each section. Values where computed when things looked right.
    s1_err = np.max(error[(x_test.flatten() < 0.9)])
    s2_err = np.max(error[(x_test.flatten() > 1.1) & (x_test.flatten() < 1.9)])
    s3_err = np.max(error[x_test.flatten() > 2.1])
    assert(s3_err < 10**-13)
    assert(s2_err < 0.01)
    assert(s1_err < 10**-13)
