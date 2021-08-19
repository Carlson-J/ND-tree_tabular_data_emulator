import os
import sys
import matplotlib.pyplot as plt

from data_loading_functions.load_test_data import load_test_data
from nd_emulator import build_emulator, make_cpp_emulator
from data_loading_functions import load_SRO_EOS
import numpy as np
"""
Build the emulator for the SRO EOS
"""

if __name__ == "__main__":
    # ------- Edit the details here ------- #

    cpp_source_dir = '../cpp_emulator'
    # Load table"../tables/SRO_training_rho1025_temp513_ye129_gitM6eba730_20210624.h5"
    EOS_file = "../../tables/SRO_training_rho1025_temp513_ye129_gitM6eba730_20210624.h5"
    vars, domain = load_SRO_EOS(EOS_file)
    spacing = ['linear', 'linear', 'linear']      # We will do the transform ahead of time.

    relative_error_dict = {
        'Abar': True,
        'X3he': False,
        'X4li': False,
        'Xa': False,
        'Xd': False,
        'Xh': False,
        'Xn': False,
        'Xp': False,
        'Xt': False,
        'Zbar': True,
        'cs2': True,
        'dedt': True,
        'dpderho': True,
        'dpdrhoe': True,
        'entropy': True,
        'gamma': True,
        'logenergy': True,
        'logpress': True,
        'mu_e': True,
        'mu_n': True,
        'mu_p': True,
        'muhat': True,
        'munu': True
    }
    N = 1
    assert(len(sys.argv) == 2)
    var = sys.argv[1]
    # errors = np.logspace(-10, -5, N)[::-1]
    errors = [0.1]
    # sizes = np.zeros([N, num_vars])

    # Directory where the emulator should be saved. Will be created if it does note exist.
    save_directory = f"./emulators_lin/{var}/"
    for i in range(N):
        # for j, key in enumerate(vars.keys()):
        # Name of emulator. This will be used to construct the name used when calling the compiled version and
        # -- and determining the filenames of the various saved files.
        # -- It should not contain spaces, nasty special characters or a file extension
        emulator_name = f"sro_{var}_max_err_{i}"  #linear_err{errors[i]:0.2e}"

        # Specify model types
        # -- add each model type to the list in the format of a dict {'type': name, ...}
        model_classes = [{'type': 'nd-linear', 'transforms': None}]#, {'type': 'nd-linear', 'transforms': 'log'}]

        # set tree parameters
        # -- Make sure the depth of the tree is not so deep that there is not enough data
        # -- for example. If the smallest values in dims is 2**3+1, then the max depth you
        # -- can choose is 3.
        max_depth = 9  # as much refinement as needed.
        error_threshold = errors[i]
        max_test_points = 100       # The max number of points to eval in a cell when estimating the error
        relative_error = relative_error_dict[var]      # Whether or not the error threshold is absolute or relative error

        # create the emulator (should not need to modify this)
        emulator = build_emulator(vars[var], max_depth, domain, spacing, error_threshold, model_classes,
                                max_test_points=max_test_points, relative_error=relative_error,
                                expand_index_domain=True, error_type="max")

        # ------- No not edit below here ------- #
        # create folder to save files in
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # save compact emulator
        emulator.save(save_directory, emulator_name)
        
        make_cpp_emulator(save_directory, emulator_name, cpp_source_dir)

        print(f'done: N={i}, type={var}')
