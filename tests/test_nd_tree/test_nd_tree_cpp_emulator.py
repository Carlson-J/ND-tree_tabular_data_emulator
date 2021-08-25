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


@pytest.mark.dependency(depends=["test_2d_log_transforms"])
def test_cpp_emulator():
    save_directory = '.'
    emulator_name = 'non_linear2d'
    cpp_source_dir = nd_emulator.__path__[0] + '/../cpp_emulator'
    make_cpp_emulator(save_directory, emulator_name, cpp_source_dir)

    EPS = 10 ** -2
    # load CPP emulator
    emulator_cpp = EmulatorCpp(save_directory + '/' + emulator_name + "_table.hdf5", emulator_name,
                               save_directory + '/' + emulator_name + ".so")

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

    assert (np.mean(diff) < EPS)

    print(
        f"cpp time: {dt_cpp} \npy  time: {dt_py} \npy/cpp : {dt_py / dt_cpp}\ndiff: L1={np.mean(diff)}, LI={np.max(diff)}")



def test_cpp_emulator_miss_aligned_2d():
    save_directory = './'
    emulator_name = 'miss_aligned_2d'
    cpp_source_dir = nd_emulator.__path__[0] + '/../cpp_emulator'
    make_cpp_emulator(save_directory, emulator_name, cpp_source_dir)

    EPS = 10**-2
    # load CPP emulator
    emulator_cpp = EmulatorCpp(save_directory + '/' + emulator_name + "_table.hdf5", emulator_name,
                               save_directory + '/' + emulator_name + ".so")

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

    assert (np.mean(diff) < EPS)

    print(f"cpp time: {dt_cpp} \npy  time: {dt_py} \npy/cpp : {dt_py/dt_cpp}\ndiff: L1={np.mean(diff)}, LI={np.max(diff)}")


def test_cpp_emulator_miss_aligned_2d_extended():
    save_directory = './'
    emulator_name = 'miss_aligned_2d_extended'
    cpp_source_dir = nd_emulator.__path__[0] + '/../cpp_emulator'
    make_cpp_emulator(save_directory, emulator_name, cpp_source_dir)

    EPS = 10**-2
    # load CPP emulator
    emulator_cpp = EmulatorCpp(save_directory + '/' + emulator_name + "_table.hdf5", emulator_name,
                               save_directory + '/' + emulator_name + ".so")

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

    assert (np.mean(diff) < EPS)

    print(f"cpp time: {dt_cpp} \npy  time: {dt_py} \npy/cpp : {dt_py/dt_cpp}\ndiff: L1={np.mean(diff)}, LI={np.max(diff)}")


def test_cpp_emulator_multiple_models():
    save_directory = '.'
    emulator_name = 'non_linear_multi_model'
    cpp_source_dir = nd_emulator.__path__[0] + '/../cpp_emulator'

    # create emulator
    # create function and domain
    domain = [[0, 3]]
    x = np.linspace(domain[0][0], domain[0][1], 2 ** 5 + 1)
    y = 3 * x
    y[x > 1] = np.sin(x[x > 1])
    y[x > 2] = 3 ** x[x > 2]
    data = {'f': y}
    spacing = ['linear']

    # build tree
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
    f_true[x_test > 2] = 3 ** x_test[x_test > 2]
    error = abs(f_true.flatten() - f_interp)
    # make sure the errors do not change from where they are right now
    # in each section. Values where computed when things looked right.
    s1_err = np.max(error[(x_test.flatten() < 0.9)])
    s2_err = np.max(error[(x_test.flatten() > 1.1) & (x_test.flatten() < 1.9)])
    s3_err = np.max(error[x_test.flatten() > 2.1])
    assert(s3_err < 10**-13)
    assert(s2_err < 0.01)
    assert(s1_err < 10**-13)

    # save emulator
    emulator.save(save_directory, emulator_name)

    # create cpp emulator
    make_cpp_emulator(save_directory, emulator_name, cpp_source_dir)

    EPS = 10 ** -10
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

    print(
        f"cpp time: {dt_cpp} \npy  time: {dt_py} \npy/cpp : {dt_py / dt_cpp}\ndiff: L1={np.mean(diff)}, LI={np.max(diff)}")

