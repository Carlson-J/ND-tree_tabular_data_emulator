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
