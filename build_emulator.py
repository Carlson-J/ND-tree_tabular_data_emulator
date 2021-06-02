import os
from data_loading_functions.load_test_data import load_test_data
from nd_emulator import build_emulator
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
    skip_emulator_creation = False
    # Directory where the emulator should be saved. Will be created if it does note exist.
    save_directory = "./test"
    # Name of emulator. This will be used to construct the name used when calling the compiled version and
    # -- and determining the filenames of the various saved files.
    # -- It should not contain spaces, nasty special characters or a file extension
    emulator_name = "testing"

    if not skip_emulator_creation:
        # Specify domain and number of points in each dimension
        # -- Note that the number of points must equal 2^a-1 for some integer a
        # -- the spacing should be how the points are spaced. They must be evenly spaced, but
        # -- that spacing can be in linear or log space.
        domain = [[0, 1], [2, 3]]
        dims = [2**3+1, 2**4+1]
        spacing = ['linear', 'linear']

        # Load function data
        # -- You should create your own function to load your data and put it in the data_loading_functions folder
        # -- where you can call it from.
        data = load_test_data(dims, domain)

        # Specify model types
        # -- add each model type to the list in the format of a dict {'type': name, ...}
        model_classes = [{'type': 'nd-linear'}]

        # set tree parameters
        # -- Make sure the depth of the tree is not so deep that there is not enough data
        # -- for example. If the smallest values in dims is 2**3+1, then the max depth you
        # -- can choose is 3.
        max_depth = 3
        error_threshold = 1e-2
        max_test_points = 100       # The max number of points to eval in a cell when estimating the error
        relative_error = False      # Whether or not the error threshold is absolute or relative error

        # create the emulator (should not need to modify this)
        emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes,
                                  max_test_points=max_test_points, relative_error=relative_error)

    # ------- No not edit below here ------- #
    # create folder to save files in
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # save compact emulator
    if not skip_emulator_creation:
        emulator.save(save_directory, emulator_name)

    # create C++ shared library to accompany it
    # -- Create tmp build directory (the old one will be removed)
    tmp_dir = "./tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    # -- create build files using cmake
    if sys.platform.startswith('win'):
        shell = True
    else:
        shell = False
    # -- Put include file that is needed to compile the library for the specific table
    shutil.copy(save_directory + '/' + emulator_name + "_cpp_params.h", 'cpp_emulator/emulator/table_params.h')
    # -- Create build files
    cmakeCmd = ["cmake", '-S', './cpp_emulator', '-B', tmp_dir, '-DCMAKE_BUILD_TYPE=RelWithDebInfo']
    subprocess.check_call(cmakeCmd, stderr=subprocess.STDOUT, shell=shell)
    # -- build C++ code
    cmakeCmd = ["cmake", '--build', tmp_dir, '--target', 'ND_emulator_lib']
    subprocess.check_call(cmakeCmd, stderr=subprocess.STDOUT, shell=shell)
    # -- move C++ library to install folder
    shutil.copy(tmp_dir + '/libND_emulator_lib.so', save_directory + f'/{emulator_name}_lib.so')

    # create readme file with the names of the function calls to used with the shared libraries
    with open(save_directory + '/README.md', 'w') as file:
        str = f"""# README for the *{emulator_name}* emulator
This folder should contain three file, 
    *{emulator_name}_table.hdf5*: Contains the data for the compact emulator
    *{emulator_name}_cpp_params.h*: Contains the #define's used when creating the C++ lib
    *{emulator_name}_lib.so*: C++ lib that has C extern functions that can be called to make, use, and destroy the
        emulator. 
        
The function names in *{emulator_name}_lib.so* that can be called are named based on the emulators name and are as follows:
    *{emulator_name}_emulator_setup*: Constructs an emulator C++ object.
    *{emulator_name}_emulator_interpolate*: Calls the emulator object for interpolation.
    *{emulator_name}_emulator_free*: Frees the memory allocated by *{emulator_name}_emulator_setup*.

"""
        file.write(str)

    print('done')
