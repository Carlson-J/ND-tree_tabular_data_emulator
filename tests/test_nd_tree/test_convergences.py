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


def test_2d_convergence(dataset_2d_non_linear):
    EPS = 10 ** -4
    N = 100
    data, domain, spacing = dataset_2d_non_linear
    error_threshold = 0
    model_classes = [{'type': 'nd-linear'}]

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
    model_classes = [{'type': 'nd-linear'}]

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
