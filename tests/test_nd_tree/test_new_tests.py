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