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
    N = 500
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
