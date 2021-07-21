import os
from data_loading_functions.load_test_data import load_test_data
from nd_emulator import build_emulator, make_cpp_emulator
from data_loading_functions.load_SRO import load_SRO_EOS
import subprocess
import sys
import shutil
"""
This script is used to create a compact emulator using a tree based decomposition of the domain and
multiple model classes to fit the function in each resulting cell.

Save directory: 
"""

if __name__ == "__main__":
    # ------- Edit the details here ------- #
    # If the emulator has already been made, but you want to regenerate the C++ library set to True
    skip_emulator_creation = True
    # Directory where the emulator should be saved. Will be created if it does note exist.
    save_directory = "./profiling_v2_sparse"
    cpp_source_dir = './cpp_emulator'
    # Name of emulator. This will be used to construct the name used when calling the compiled version and
    # -- and determining the filenames of the various saved files.
    # -- It should not contain spaces, nasty special characters or a file extension
    emulator_name = "test_v2_sparse"

    if not skip_emulator_creation:

        # Load function data
        # -- You should create your own function to load your data and put it in the data_loading_functions folder
        # -- where you can call it from.
        filepath = "../tables/SRO_training_rho1025_temp513_ye129_gitM6eba730_20210624.h5"
        data_raw, domain = load_SRO_EOS(filepath, vars_to_load=['Abar'])
        data = data_raw['Abar']
        spacing = ['linear', 'linear', 'linear']

        # Specify model types
        # -- add each model type to the list in the format of a dict {'type': name, ...}
        model_classes = [{'type': 'nd-linear'}]

        # set tree parameters
        # -- Make sure the depth of the tree is not so deep that there is not enough data
        # -- for example. If the smallest values in dims is 2**3+1, then the max depth you
        # -- can choose is 3.
        max_depth = 9
        error_threshold = 1e-4 
        max_test_points = 100       # The max number of points to eval in a cell when estimating the error
        relative_error = True      # Whether or not the error threshold is absolute or relative error

        # create the emulator (should not need to modify this)
        emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes,
                                  max_test_points=max_test_points, relative_error=relative_error,
                                  expand_index_domain=True)

    # ------- No not edit below here ------- #
    # create folder to save files in
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # save compact emulator
    if not skip_emulator_creation:
        emulator.save(save_directory, emulator_name)

    make_cpp_emulator(save_directory, emulator_name, cpp_source_dir)

    print('done')
