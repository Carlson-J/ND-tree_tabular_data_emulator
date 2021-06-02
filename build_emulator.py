import os
from data_loading_functions.load_test_data import load_test_data
from nd_emulator import build_emulator
"""
This script is used to create a compact emulator using a tree based decomposition of the domain and
multiple model classes to fit the function in each resulting cell.

Save directory: 
"""

if __name__ == "__main__":
    # ------- Edit the details here ------- #
    # Directory where the emulator should be saved. Will be created if it does note exist.
    save_directory = "./test"
    # Name of emulator. This will be used to construct the name used when calling the compiled version and
    # -- and determining the filenames of the various saved files.
    # -- It should not contain spaces, nasty special characters or a file extension
    emulator_name = "testing"

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

    # ------- No not edit below here ------- #
    # create folder to save files in
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # create the emulator
    emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes,
                              max_test_points=max_test_points, relative_error=relative_error)

    # save compact emulator
    emulator.save(save_directory + '/' + emulator_name + '.hdf5')

    # create C++ shared library to accompany it