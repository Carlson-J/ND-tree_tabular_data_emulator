import numpy as np
from .model_classes import fit_nd_linear_model, nd_linear_model
from .mask import create_mask, get_mask_dims
from .domain_functions import transform_domain


class DTree:
    def __init__(self, tree_params, data):
        """
        Creates an ND-tree over the given domain. The refinement of the tree is carried out until
        a specified error threshold is reached.
        :param tree_params: (dict) parameters for the create of the tree
        :param data: (dict) A dictionary containing the data over the domain of interest.
        {   f (nd-array): function values
            df_x1_x1_...x2_x2_....x_N,x_N (nd-array) The partial derivative of f in terms of x1,x2,...xN. Each x#
                included is an additional partial derivative, e.g., df_x1_x1_x2 is the partial derivative in terms of
                x2 twice and x2 once. Any number of derivative combinations can be included but the order should be the
                same. The model classes will check if all the needed derivatives are available.
        }
        The size of each dim should be (2^k)+1, where k is some integer >= max_depth.
        :return:
        """
        # constants needed for saving and loading things
        self.TRANSFORMS = ['linear', 'log']
        self.params = tree_params
        self.data = data

        # create the root node
        self.tree = {
            'domain': self.params['domain'],
            'children': None,
            'id': [],
            'model': None,
            'mask': create_mask(self.params['domain'], self.params['domain'], self.params['dims'], self.params['spacing']),
            'error': None
        }

        # build tree
        self.refine_region(self.tree)

    def refine_region(self, node):

        # train region
        fit, error = self.fit_region(node)

        # check error and depth
        if error >= self.params['error_threshold'] and self.params['max_depth'] > len(node['id']):
            # create children nodes
            self.create_children_nodes(node)
            if len(node['children'][0]['id']) > self.params['achieved_depth']:
                self.params['achieved_depth'] = len(node['children'][0]['id'])
            # refine children nodes
            for i in range(2**len(self.params['dims'])):
                self.refine_region(node['children'][i])
        else:
            node['model'] = fit
            node['error'] = error
        return

    def create_children_nodes(self, node):
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
        :return: None
        """
        num_dims = len(node['domain'])
        assert node['children'] is None
        # do any needed transforms
        domain = transform_domain(node['domain'], self.params['spacing'], reverse=False)

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

            # compute new id
            id_new = list(node['id'].copy())
            id_new.append(i)
            # reverse transformation
            domain_new = transform_domain(domain_new, self.params['spacing'], reverse=True)
            node['children'].append({
                'domain': domain_new,
                'children': None,
                'id': id_new,
                'model': None,
                'mask': create_mask(domain_new, self.params['domain'], self.params['dims'], self.params['spacing']),
                'error': None
            })

    def fit_region(self, node):
        """
        Fit the current model to the region and determines the fit error.
        :param node: (dict) tree node, i.e., the region
        :return: fit, error
        """
        current_error = np.infty
        best_fit = None
        assert (len(self.params['model_classes']) > 0)
        # create test set
        dims = get_mask_dims(node['mask'])
        random_indices = np.random.choice(np.prod(dims, dtype=int), min([np.prod(dims), self.params['max_test_points']]))
        test_indices = np.indices(dims).reshape([len(dims), np.prod(dims, dtype=int)]).T[random_indices]
        test_points = np.zeros([len(random_indices), len(dims)])
        for i in range(len(dims)):
            test_points[:, i] = self.params['domain_spacings'][i][test_indices[:, i]]

        # try each model class to which gives the lowest predicted error
        for model_class in self.params['model_classes']:
            if model_class['type'] == 'nd-linear':
                # fit model
                X = np.zeros([2, len(self.params['dims'])])
                for i in range(len(self.params['dims'])):
                    X[0, i] = self.params['domain_spacings'][i][node['mask'][i]][0]
                    X[1, i] = self.params['domain_spacings'][i][node['mask'][i]][-1]
                weights = fit_nd_linear_model(self.data['f'][node['mask']], X)
                fit = {'type': model_class, 'weights': weights, 'transforms': [None]*len(self.params['dims'])}
                # compute error
                f_interp = nd_linear_model(weights, test_points)
                f_true = np.array([self.data['f'][tuple(a)] for a in test_indices])
                err = self.compute_error(f_true, f_interp)
            else:
                raise ValueError("Unknown model type: {model_class}")
            # check if fit is best so far
            if err < current_error:
                current_error = err
                best_fit = fit
        # Return best fit and its error.
        return best_fit, current_error

    def compute_error(self, true, interp):
        """
        Compute the relative norm error between two arrays
        :param true: (nd array)
        :param interp: (nd array)
        :return: (float) error
        """
        if self.params['relative_error']:
            return np.max(abs(true-interp)/abs(true))
        else:
            return np.max(abs(true-interp))

    def _get_leaves(self, node, leaf_list):
        """
        A recursive function that will return the leaves of the tree
        :param node: The current node of the tree you are on
        :param leaf_list: (list) all the leaves will be appended to this list
        :return:
        """
        for child in node['children']:
            if child['children'] is None:
                leaf_list.append(child)
            else:
                self._get_leaves(child, leaf_list)

    def get_leaves(self):
        """
        Compute all the leaf nodes and put them in an ordered list
        :return: (list) of leaf nodes
        """
        leaves = []
        self._get_leaves(self.tree, leaves)
        return leaves
