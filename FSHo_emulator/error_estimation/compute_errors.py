import numpy as np
import matplotlib.pyplot as plt
from nd_emulator import build_emulator
from data_loading_functions import load_SFHo_EOS

if __name__ == "__main__":
    EOS_file = "../../../tables/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5"
    vars, domain = load_SFHo_EOS(EOS_file)
    spacing = ['linear', 'linear', 'linear']  # We will do the transform ahead of time.
    # get subset of Abar
    L = 10
    abar = vars['Abar']['f'][:L, :L, :L]

    # Specify model types
    # -- add each model type to the list in the format of a dict {'type': name, ...}
    model_classes = [{'type': 'nd-linear'}]

    # set tree parameters
    # -- Make sure the depth of the tree is not so deep that there is not enough data
    # -- for example. If the smallest values in dims is 2**3+1, then the max depth you
    # -- can choose is 3.
    max_depth = 100  # as much refinement as needed.
    error_threshold = -1.0
    max_test_points = 100  # The max number of points to eval in a cell when estimating the error
    relative_error = True  # Whether or not the error threshold is absolute or relative error

    # create the emulator (should not need to modify this)
    emulator, tree = build_emulator({'f': abar}, max_depth, domain, spacing, error_threshold, model_classes,
                                    max_test_points=max_test_points, relative_error=relative_error,
                                    expand_index_domain=True, return_tree=True)

    print("DONE")
