import numpy as np
import matplotlib.pyplot as plt
from nd_emulator import build_emulator
from data_loading_functions import load_SFHo_EOS
import numpy as np
import matplotlib.pyplot as plt
from nd_emulator import build_emulator
from data_loading_functions import load_SFHo_EOS

if __name__ == "__main__":
    EOS_file = "../../../tables/Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5"
    vars, domain = load_SFHo_EOS(EOS_file)
    spacing = ['linear', 'linear', 'linear']  # We will do the transform ahead of time.

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

    for key in vars.keys():
        if key[0] != "X":
            continue
        # get subset of Abar
        # L = 17
        data = vars[key]['f']#[:L, :L, :L]

        # Specify model types
        # -- add each model type to the list in the format of a dict {'type': name, ...}
        model_classes = [{'type': 'nd-linear'}]
        error_type = 'RMSE'

        # set tree parameters
        # -- Make sure the depth of the tree is not so deep that there is not enough data
        # -- for example. If the smallest values in dims is 2**3+1, then the max depth you
        # -- can choose is 3.
        max_depth = 100  # as much refinement as needed.
        error_threshold = -1.0
        max_test_points = 100  # The max number of points to eval in a cell when estimating the error
        relative_error = relative_error_dict[key]  # Whether or not the error threshold is absolute or relative error

        # create the emulator (should not need to modify this)
        emulator, tree = build_emulator({'f': data}, max_depth, domain, spacing, error_threshold, model_classes,
                                        max_test_points=max_test_points, relative_error=relative_error,
                                        expand_index_domain=True, return_tree=True, error_type=error_type)

        max_depth = tree.achieved_depth
        leaves = tree.get_leaves()

        D =max_depth
        num_dims = tree.num_dims


        E = []
        for leaf in leaves:
            if leaf['error'] is not None:
                E.append(leaf)
        errors_1 = np.zeros([len(E),max_depth+1])
        EPS = 10**-16
        for i in range(len(E)):
            current_node = tree.root
            errors_1[i, 0] = current_node['error']
            for j in range(max_depth):
                current_node = current_node['children'][E[i]['id'][j]]
                errors_1[i, j+1] = current_node['error']
            h = 1
            # linear fit
            hs = np.log(np.array([1, 0.5, 0.25, 0.125, 0.125/2.]))
            fit = np.polyfit(hs[-3:-1], np.log(errors_1[i, -3:-1] + EPS), h)
            last_col = fit[0]*hs[-1] + fit[1]
            errors_1[i,-1] = np.exp(last_col)

        # errors
        np.save(f'./error_data/errors_{key}', errors_1)
