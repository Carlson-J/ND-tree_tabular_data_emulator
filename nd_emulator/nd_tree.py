import numpy as np
from .model_classes import fit_nd_linear_model, nd_linear_model


def compute_ranges(domain, spacing, dims):
    """
    Create an array for each axis that contains the intervals along that axis
    :param domain:  (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
            dimension.
    :param spacing: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :param dims: [(int),...,(int)] the number of elements in each dimension
    :return:
    """
    ranges = []
    for i in range(len(dims)):
        if spacing[i] == 'linear':
            ranges.append(np.linspace(domain[i][0], domain[i][1], dims[i]))
        elif spacing[i] == 'log':
            ranges.append(np.logspace(np.log10(domain[i][0]), np.log10(domain[i][1]), dims[i]))
        else:
            raise ValueError(f"spacing type '{spacing[i]}' unknown")
    return ranges


class ND_Tree:
    def __init__(self, data, max_depth, domain, dims, spacing, error_threshold, model_classes):
        """
        Creates an ND-tree over the given domain. The refinement of the tree is carried out until
        a specified error threshold is reached.
        :param data: (dict) A dictionary containing the data over the domain of interest.
            {   f (nd-array): function values
                df_x1_x1_...x2_x2_....x_N,x_N (nd-array) The partial derivative of f in terms of x1,x2,...xN. Each x#
                    included is an additional partial derivative, e.g., df_x1_x1_x2 is the partial derivative in terms of
                    x2 twice and x2 once. Any number of derivative combinations can be included but the order should be the
                    same. The model classes will check if all the needed derivatives are available.
            }
            The size of each dim should be (2^k)+1, where k is some integer >= max_depth.
        :param max_depth: (int) maximum depth the tree should be able to go
        :param domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
            dimension.
        :param dims: [(int),...,(int)] the number of elements in each dimension
        :param spacing: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
        :param model_classes: (list of dicts) list of model classes to use when training. Each entry should have the
            form {  'type': (string),
                    'transforms': (list of length dims of strings) None or 'log', being the transform in each dim.
                    }
        :return:
        """
        # check that data is correct format
        assert (type(data['f']) is np.ndarray)
        assert (len(dims) == data['f'].ndim)
        for key in data:
            assert (data[key].shape == data['f'].shape)
        # check if each element is it is a power of 2 after removing 1
        for n in range(len(dims)):
            a = data['f'].shape[n] - 1
            # (https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two)
            assert ((a & (a - 1) == 0) and a != 0)

        # create the root node
        self.tree = {
            'domain': domain,
            'children': None,
            'id': '0',
            'model': None,
            'mask': None,
            'error': None
        }
        # save parameters
        self.max_depth = max_depth
        self.spacing = spacing
        self.dims = dims
        self.error_threshold = error_threshold
        self.model_classes = model_classes

        # Train emulator
        self.refine_region()

    def refine_region(self):

        # train region


        # check error and depth
        return

    def fit_region(self, node):
        """
        Fit the current model to the region and determines the fit error.
        :param node: (dict) tree node, i.e., the region
        :return: fit, error
        """
        # TODO: create test for this
        current_error = np.infty
        best_fit = None
        assert (len(self.model_classes) > 0)
        # try each model class to which gives the lowest predicted error
        for model_class in self.model_classes:
            if model_class['type'] == 'nd-linear':
                # fit model
                # X = np.array([[self.]])
                fit = fit_nd_linear_model()
                # compute error
        #         f_interp =
        #
        #
        #     # check if log space model class
        #     if len(model) > 9 and model[-9:] == "_logSpace":
        #         log_space = True
        #         model = model[:-9]
        #     else:
        #         log_space = False
        #     # get domain
        #     X, Y, Z, derivatives = self.get_region_values(mask, log_space)
        #     # Choose the model that has the lowest error.
        #     if "bi-quintic_enhanced" == model:
        #         # Derivatives should be layed out in self.data.derivative like so:
        #         # [Zy, Zx, Zxy, Zyy, Zxx, Zyyx, Zxxy, Zyyxx]
        #         if derivatives is None:
        #             raise ValueError("Derivatives must be included when using bi-quintic_enhanced.")
        #         # Change X, Y, and Z to square layout.
        #         N = X.shape[0]
        #         if X.shape[1] != X.shape[0]:
        #             assert (X.shape[1] != X.shape[0])
        #         X = X.reshape([N, N])
        #         Y = Y.reshape([N, N])
        #         Z = Z.reshape([N, N])
        #         # do fit
        #         fit = create_bi_quintic_fit(X, Y, Z, derivatives)
        #         fit['model'] = model
        #         fit['log_space'] = log_space
        #         # add on sign term.
        #         if log_space:
        #             signs = mask(self.data.signs)
        #             if all((signs == -1).flatten()):
        #                 s = -1
        #             else:
        #                 # If it should stay positive or if there is a mix don't change the sign.
        #                 s = 1
        #             fit['sign_term'] = s
        #     else:
        #         A = construct_A(X, Y, model)
        #         theta = np.linalg.lstsq(A, Z, rcond=None)[0]
        #         fit = {'theta': theta, 'model': model, 'log_space': log_space}
        #
        #     # check if fit is best so far
        #     err = self._fit_lI_err(X, Y, Z, fit, log_space=log_space)
        #     if err < current_error:
        #         current_error = err
        #         best_fit = fit
        # # Return best fit and its error.
        # return best_fit, current_error