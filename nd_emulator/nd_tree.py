import numpy as np
from .model_classes import fit_nd_linear_model, nd_linear_model
from .mask import create_mask


def get_dims(mask):
    """
    Compute the dims of a given mask
    :param mask:
    :return: dims
    """
    num_dims = len(mask)
    dims = np.zeros([num_dims], dtype=int)
    for j in range(num_dims):
        dims[j] = mask[j].stop - mask[j].start
    return dims


def transform_domain(domain, spacings, reverse=False):
    """
    Transform the domain so that the spacing is linear, e.g., for log spacing we take the log. For linear we do nothing.
    The original input array is not modified
    :param domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
            dimension.
    :param spacings: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :param reverse: undo previous transforms. For linear, nothing is done, for log, we compute 10^()
    :return: transformed domain
    """
    new_domain = domain.copy()
    for i in range(len(spacings)):
        if spacings[i] == 'linear':
            continue
        elif spacings[i] == 'log':
            if reverse:
                new_domain[i] = 10**(new_domain[i])
            else:
                new_domain[i] = np.log10(new_domain[i])
    return new_domain


def compute_ranges(domain, spacing, dims):
    """
    Create an array for each axis that contains the intervals along that axis
    :param domain:  (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
            dimension.
    :param spacing: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :param dims: [(int),...,(int)] the number of elements in each dimension
    :return: (list) a list of arrays that hold the point locations along the axis corresponding to the index in the list
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


def create_children_nodes(node, spacing):
    """
    Add D more children nodes to the current node, where D is the number of dims
    :param node: {
            'domain': (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
            dimension.
            'children': None
            'id': (list of ints)
            'model'
            'mask'
            'error'
        }
    :param spacing: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :return: None
    """
    num_dims = len(node['domain'])
    assert node['children'] is None
    # do any needed transforms
    domain = transform_domain(node['domain'], spacing, reverse=False)

    # unpack corner and half the cell edge values
    c1 = np.array(domain)[:, 0]
    c2 = np.array(domain)[:, 1]
    dx = (c2 - c1) / 2.
    # add children nodes
    node['children'] = []
    for i in range(2 ** num_dims):
        # define new domain
        # # get corner with minimum values corresponding to index 0 in the new cell/node
        c1_new = np.zeros([num_dims])
        for j in range(num_dims):
            c1_new[j] = c1[j] + dx[j] * ((i >> j) & 1)
        c2_new = c1_new + dx
        # # compute new domain
        domain_new = np.zeros([num_dims, 2])
        domain_new[:, 0] = c1_new
        domain_new[:, 1] = c2_new

        # compute dims of old domain
        dims = get_dims(node['mask'])
        # compute new id
        id_new = list(node['id'].copy())
        id_new.append(i)
        # reverse transformation
        domain_new = transform_domain(domain_new, spacing, reverse=True)
        node['children'].append({
            'domain': domain_new,
            'children': None,
            'id': id_new,
            'model': None,
            'mask': create_mask(domain_new, node['domain'], dims, spacing),
            'error': None
        })


class ND_Tree:
    def __init__(self, data, max_depth, domain, spacing, error_threshold, model_classes, max_test_points=100,
                 relative_error=False):
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
        :param spacing: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
        :param model_classes: (list of dicts) list of model classes to use when training. Each entry should have the
            form {  'type': (string),
                    'transforms': (list of length dims of strings) None or 'log', being the transform in each dim.
                    }
        :param max_test_points: (int) maximum number of test points to use when computing the error in a cell.
        :param relative_error: use relative error, i.e., (f_interp - f_true)/f_true. This will have issues if you go
            to or close to zero.
        :return:
        """
        # check that data is correct format
        assert (type(data['f']) is np.ndarray)
        dims = data['f'].shape
        for key in data:
            assert (data[key].shape == dims)
        # check if each element is it is a power of 2 after removing 1
        for n in range(len(dims)):
            a = data['f'].shape[n] - 1
            # (https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two)
            assert ((a & (a - 1) == 0) and a != 0)

        # create the root node
        self.tree = {
            'domain': domain,
            'children': None,
            'id': [0],
            'model': None,
            'mask': create_mask(domain, domain, dims, spacing),
            'error': None
        }
        # save parameters
        self.max_depth = max_depth
        self.spacing = spacing
        self.dims = dims
        self.error_threshold = error_threshold
        self.model_classes = model_classes
        self.domain_spacings = compute_ranges(domain, spacing, dims)
        self.data = data
        self.max_test_points = max_test_points
        self.relative_error = relative_error

        # Train emulator
        self.refine_region(self.tree)

    def refine_region(self, node):

        # train region
        fit, error = self.fit_region(node)

        # check error and depth
        if error >= self.error_threshold:
            # create children nodes
            create_children_nodes(node, self.spacing)
            # refine children nodes
            for i in range(len(self.dims)):
                self.refine_region(node['children'][i])
        return

    def compute_error(self, true, interp):
        """
        Compute the relative norm error between two arrays
        :param true: (nd array)
        :param interp: (nd array)
        :return: (float) error
        """
        if self.relative_error:
            return np.max(abs(true-interp)/abs(true))
        else:
            return np.max(abs(true-interp))

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
        # create test set
        dims = get_dims(node['mask'])
        random_indices = np.random.choice(np.prod(dims, dtype=int), min([np.prod(dims), self.max_test_points]))
        test_indices = np.indices(dims).reshape([len(dims), np.prod(dims, dtype=int)]).T[random_indices]
        test_points = np.zeros([len(random_indices), len(dims)])
        for i in range(len(dims)):
            test_points[:, i] = self.domain_spacings[i][test_indices[:, i]]

        # try each model class to which gives the lowest predicted error
        for model_class in self.model_classes:
            if model_class['type'] == 'nd-linear':
                # fit model
                X = np.zeros([2, len(self.dims)])
                for i in range(len(self.dims)):
                    X[0, i] = self.domain_spacings[i][node['mask'][i]][0]
                    X[1, i] = self.domain_spacings[i][node['mask'][i]][-1]
                weights = fit_nd_linear_model(self.data['f'][node['mask']], X)
                fit = {'type': model_class, 'weights': weights, 'transforms': [None]*len(self.dims)}
                # compute error
                f_interp = nd_linear_model(weights, test_points)
                f_true = np.array([self.data['f'][tuple(a)] for a in test_indices])
                err = self.compute_error(f_true, f_interp)
            # check if fit is best so far
            if err < current_error:
                current_error = err
                best_fit = fit
        # Return best fit and its error.
        return best_fit, current_error
