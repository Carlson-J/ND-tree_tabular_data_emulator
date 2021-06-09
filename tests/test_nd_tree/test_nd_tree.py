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
    model_classes = [{'type': 'nd-linear', 'transforms': [None] * 2}]
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
    model_classes = [{'type': 'nd-linear', 'transforms': ['linear'] * 4}]
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
        assert(abs(output[i] - sol) <= EPS)
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
    model_classes = [{'type': 'nd-linear', 'transforms': ['linear'] * 4}]
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
    model_classes = [{'type': 'nd-linear', 'transforms': [None]}]
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
    plt.plot(cell_locations.flatten(), min(y)*np.ones_like(cell_locations.flatten()), '*', label='Cell Locations')
    # Plot errors
    plt.plot(x_test, error, '--', label='error')
    plt.plot(x_test, f_true, label='true')
    plt.plot(x_test, f_interp, '--', label='interp')
    plt.plot(x, y, '.', label='data')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def test_2d_convergence(dataset_2d_non_linear):
    EPS = 10 ** -4
    N = 100
    data, domain, spacing = dataset_2d_non_linear
    error_threshold = 0
    model_classes = [{'type': 'nd-linear', 'transforms': [None] * 2}]

    # Compute true values
    X, Y = np.meshgrid(np.linspace(domain[0][0], domain[0][1], N), np.linspace(domain[1][0], domain[1][1], N))
    input = np.array([X.flatten(), Y.flatten()]).T
    f_true = np.cos(X) * 2 + np.sin(Y)

    # error arrays
    num_depths = 6
    l1_error = np.zeros(num_depths)
    lI_error = np.zeros(num_depths)

    for i in range(0, num_depths):
        # Create emulator
        emulator = build_emulator(data, i + 1, domain, spacing, error_threshold, model_classes)
        # test at different points
        f_interp = emulator(input).reshape([N, N])
        # compute error
        error = abs(f_true - f_interp)
        l1_error[i] = np.mean(error)
        lI_error[i] = np.max(error)
    # resize and plot
    plt.figure()
    plt.plot(np.arange(1, num_depths + 1), l1_error, label='L1 Norm')
    plt.plot(np.arange(1, num_depths + 1), lI_error, label='LI Norm')
    plt.title("Should look linear with a negative slope.")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    plt.show()
    # check if error is low
    assert np.all(error <= EPS)


def test_4d_convergence(dataset_4d_log_non_linear):
    """
    Given: 4d data set
    WHEN: Different max depths are given
    THEN: The error converges to 0
    """
    EPS = 10 ** -1
    data, domain, spacing = dataset_4d_log_non_linear
    # create test data
    low = np.array(domain)[:, 0]
    high = np.array(domain)[:, 1]
    N = 100
    inputs = np.random.uniform(low, high, size=[N, 4])
    f_true = (inputs[:, 0] ** 2 * inputs[:, 1] * inputs[:, 2] * inputs[:, 3] + 1).flatten()
    error_threshold = 0
    model_classes = [{'type': 'nd-linear', 'transforms': [None] * 4}]

    # error arrays
    num_depths = 2
    l1_error = np.zeros(num_depths)
    lI_error = np.zeros(num_depths)

    for i in range(0, num_depths):
        # Create emulator
        emulator = build_emulator(data, i + 1, domain, spacing, error_threshold, model_classes)
        # test at different points
        f_interp = emulator(inputs).flatten()
        # compute error
        error = abs(f_true - f_interp) / abs(f_true)
        l1_error[i] = np.mean(error)
        lI_error[i] = np.max(error)

    plt.figure()
    plt.plot(np.arange(1, num_depths + 1), l1_error, label='L1 Norm')
    plt.plot(np.arange(1, num_depths + 1), lI_error, label='LI Norm')
    plt.title("Should look linear with a negative slope.")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    plt.show()
    # check if error is low
    assert np.all(l1_error[-1] <= EPS)


@pytest.mark.dependency()
def test_2d_log_transforms(dataset_2d_log_non_linear):
    """
    GIVEN: 2d function evaluations and a domain.
    WHEN: Create an emulator from data with log transforms
    THEN: Correctly sorts and interpolates data
    """
    EPS = 10 ** -1
    N = 200
    data, domain, spacing = dataset_2d_log_non_linear
    error_threshold = 0
    max_depth = 2
    model_classes = [{'type': 'nd-linear', 'transforms': [None, 'log']}]
    # Create emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
    # Compute new values over domain
    X, Y = np.meshgrid(np.linspace(domain[0][0], domain[0][1], N), np.logspace(np.log10(domain[1][0])
                                                                               , np.log10(domain[1][1]), N))
    input = np.array([X.flatten(), Y.flatten()]).T
    f_interp = emulator(input).reshape([N, N])
    f_true = X * np.log10(Y)
    error = abs(f_true - f_interp)
    # resize and plot
    plt.imshow(error, origin='lower')
    plt.title("Should not see any grid structure")
    plt.colorbar()
    plt.show()

    emulator.save('.', 'non_linear2d')
    # check if error is low
    assert np.all(error <= EPS)


@pytest.mark.dependency(depends=["test_2d_log_transforms"])
def test_cpp_emulator():
    save_directory = '.'
    emulator_name = 'non_linear2d'
    cpp_source_dir = nd_emulator.__path__[0] + '/../cpp_emulator'
    make_cpp_emulator(save_directory, emulator_name, cpp_source_dir)

    EPS = 10**-10
    # load CPP emulator
    emulator_cpp = EmulatorCpp(save_directory + '/' + emulator_name + "_table.hdf5", emulator_name,
                               save_directory + '/' + emulator_name + "_lib.so")

    # load python emulator
    emulator_py = load_emulator(save_directory + '/' + emulator_name + "_table.hdf5")

    # create test data
    domain = emulator_py.params.domain
    low = np.array(domain)[:, 0]
    high = np.array(domain)[:, 1]
    N = 100
    inputs = np.zeros([len(domain[:, 0]), N])
    for i in range(len(domain[:, 0])):
        inputs[i, :] = np.random.uniform(low[i], high[i], size=[N])

    # time the evaluation of both methods
    # # CPP
    start_cpp = time()
    out_cpp = emulator_cpp(inputs)
    end_cpp = time()
    dt_cpp = end_cpp - start_cpp
    # # Python
    start_py = time()
    out_py = emulator_py(inputs.T)
    end_py = time()
    dt_py = end_py - start_py

    # check if answer is the same
    diff = abs(out_cpp - out_py)

    assert (np.max(diff) < EPS)

    print(f"cpp time: {dt_cpp} \npy  time: {dt_py} \npy/cpp : {dt_py/dt_cpp}\ndiff: L1={np.mean(diff)}, LI={np.max(diff)}")


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
    N = 500
    error_threshold = 1e-1
    max_depth = 5
    model_classes = [{'type': 'nd-linear', 'transforms': [None]}]
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
    plt.plot(cell_locations.flatten(), min(f_true) * np.ones_like(cell_locations.flatten()), '*', label='Cell Locations')
    # Plot errors
    plt.plot(x_test, error, '--', label='error')
    plt.plot(x_test, f_true, label='true')
    plt.plot(x_test, f_interp, '--', label='interp')
    plt.plot(x, data['f'], '.', label='data')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()