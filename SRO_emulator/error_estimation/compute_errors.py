import numpy as np
import matplotlib.pyplot as plt
from nd_emulator import build_emulator
from data_loading_functions.load_SRO import load_SRO_EOS
import sys


if __name__ == "__main__":
    EOS_file = "../../../tables/SRO_training_rho1025_temp513_ye129_gitM6eba730_20210624.h5"
    vars, domain = load_SRO_EOS(EOS_file)
    spacing = ['linear', 'linear', 'linear']  # We will do the transform ahead of time. 
    
    relative_error_dict = {
        "Abar": True,
        "Albar": True,
        "Xa": False,
        "Xh": False,
        "Xl": False,
        "Xn": False,
        "Xp": False,
        "Zbar": True,
        "Zlbar": True,
        "cs2": True,
        "dedt": True,
        "dpderho": True,
        "dpdrhoe": True,
        "entropy": True,
        "gamma": True,
        "logenergy": True,
        "logpress": True,
        "meffn": True,
        "meffp": True,
        "mu_e": True,
        "mu_n": True,
        "mu_p": True,
        "muhat": True,
        "munu": True,
        "r": True,
        "u": True}


    assert(len(sys.argv) == 2)
    key = sys.argv[1]
    data = vars[key]['f']#[:L, :L, :L]

    # Specify model types
    # -- add each model type to the list in the format of a dict {'type': name, ...}
    model_classes = [{'type': 'nd-linear'}]
    error_type = 'RMSE'

    # set tree parameters
    # -- Make sure the depth of the tree is not so deep that there is not enough data
    # -- for example. If the smallest values in dims is 2**3+1, then the max depth you
    # -- can choose is 3.
    max_depth = 9  # as much refinement as needed.
    error_threshold = -1.0
    max_test_points = 100  # The max number of points to eval in a cell when estimating the error
    relative_error = relative_error_dict[key]  # Whether or not the error threshold is absolute or relative error

    # create the emulator (should not need to modify this)
    tree = build_emulator({'f': data}, max_depth, domain, spacing, error_threshold, model_classes,
                            max_test_points=max_test_points, relative_error=relative_error,
                            expand_index_domain=True, return_tree=True, error_type=error_type)

    max_depth = tree.achieved_depth
    leaves = tree.get_leaves()

    D =max_depth
    num_dims = tree.num_dims

    # get non-virtual leaf nodes
    E = []
    for leaf in leaves:
        if leaf['error'] is not None:
            E.append(leaf['error'])
    errors_1 = np.array(E)
    np.save(f'./error_data/errors_{key}', errors_1)
    # # save errors at each level
    # for i in range(len(E)):
    #     current_node = tree.root
    #     errors_1[i, 0] = current_node['error']
    #     for j in range(max_depth):
    #         current_node = current_node['children'][E[i]['id'][j]]
    #         errors_1[i, j+1] = current_node['error']

    # # errors
    # np.save(f'./error_data/errors_{key}', errors_1)
