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

