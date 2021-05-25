import numpy as np
from .domain_functions import transform_domain, transform_point
from .dtree import DTree
from .compact_mapping import convert_tree
from .model_classes import nd_linear_model
from .compact_mapping import CompactMapping, save_compact_mapping
from .parameter_struct import Parameters


def build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes, max_test_points=100,
                 relative_error=False):
    """
    Creates an emulator using ND-tree decomposition over the given domain. The refinement of the tree is carried out
    until a specified error threshold is reached.
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
    :param error_threshold: (float) the error threshold for refinement in the tree based decomposition.
    :param spacing: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :param model_classes: (list of dicts) list of model classes to use when training. Each entry should have the
        form {  'type': (string),
                'transforms': (list of length dims of strings) None or 'log', being the transform in each dim.
                }
    :param max_test_points: (int) maximum number of test points to use when computing the error in a cell.
    :param relative_error: use relative error, i.e., (f_interp - f_true)/f_true. This will have issues if you go
        to or close to zero.
    :return: (Emulator)
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

    # construct a dictionary with all the needed parameters
    tree_parameters = Parameters(max_depth, np.array(spacing), np.array(dims), error_threshold, np.array(model_classes),
                                 max_test_points, relative_error, np.array(domain))

    # build tree
    tree = DTree(tree_parameters, data)

    # convert to compact storage/mapping scheme
    compact_mapping = convert_tree(tree)

    # create emulator
    emulator = Emulator(compact_mapping)

    return emulator


def load_emulator(file_path):
    """
    Loads an ND_Tree emulator from an hdf5 file
    :param file_path: (string) path to save file
    :return:
    """


class Emulator:
    def __init__(self, compact_mapping):
        """
        Build an emulator from a compact mapping data object.
        The emulator will use this data to map inputs to correct cells and perform the corresponding interpolation
        in the cell.
        :param compact_mapping: (CompactMapping)
        """
        self.encoding_array = compact_mapping.encoding_array
        self.index_array = compact_mapping.index_array
        self.offsets = compact_mapping.offsets
        self.model_arrays = compact_mapping.model_arrays
        self.params = compact_mapping.params

    def __call__(self, inputs):
        """
        Compute the  interpolated values at each point in the input array.
        :param inputs: (2d array) each row is a different point
        :return: (array) the function value at each point.
        """
        inputs = np.atleast_2d(inputs)
        ndims = len(self.params.dims)
        assert(inputs.shape[1] == ndims)
        # precompute some things
        domain = transform_domain(self.params.domain, self.params.spacing)
        dx = np.zeros([ndims])
        for i in range(ndims):
            dx[i] = (domain[i][1] - domain[i][0]) / 2**self.params.max_depth
        sol = np.zeros([inputs.shape[0]])
        for i, point in enumerate(inputs):
            # find out which model
            weights, index = self.find_model(point, dx)
            # compute fit
            if self.params.model_classes[index]['type'] == 'nd-linear':
                sol[i] = nd_linear_model(weights, point)[0]
            else:
                raise ValueError
        return sol

    def find_model(self, point, dx):
        """
        Find the model weights and the index of model class array it is
        :param point: (array) [x0, x1,...]
        :param dx: (array) points spacing of cartesian index grid in each dim
        :return: (dict) model
        """
        tree_index = self.compute_tree_index(point, dx)
        # find index in encoding array
        index = np.searchsorted(self.encoding_array, tree_index, side='right')
        model_index = self.index_array[index]
        # Determine which type of model it is by index
        if len(self.offsets) > 1:
            model_list_index = next((x for x, val in enumerate(self.offsets[1:]) if val > model_index), len(self.offsets) - 1)
        else:
            model_list_index = 0
        model_weights = self.model_arrays[model_list_index][model_index - self.offsets[model_list_index]]
        return model_weights, model_list_index

    def compute_tree_index(self, point, dx):
        """
        Compute the tree index-space index for a given point.
        The point should be within the domain of the root node.
        :param point: (array) the location of the point
        :param dx: (array) The spacing in each direction.
        :return: (int)
        """
        # do any needed transforms for spacing reasons.
        point_new = transform_point(point, self.params.spacing, reverse=False)
        domain = transform_domain(self.params.domain, self.params.spacing)
        # compute cartesian coordinate on regular grid of points
        coords_cart = np.zeros(len(self.params.dims), dtype=int)
        for i in range(len(point_new)):
            coords_cart[i] = int(np.floor((point_new[i] - domain[i][0]) / dx[i]))
            # if the right most point would index into a cell that is not their move it back a cell.
            if coords_cart[i] == 2**self.params.max_depth:
                coords_cart[i] -= 1
        # convert to tree index
        index = 0
        for i in range(self.params.max_depth):
            for j in range(len(point)):
                index = (index << 1) | ((coords_cart[::-1][j] >> (self.params.max_depth - i - 1)) & 1)
        return index

    def get_compact_mapping(self):
        """
        Package and return compact emulator
        :return:
        """
        return CompactMapping(self.encoding_array, self.index_array, self.offsets, self.model_arrays, self.params)

    def save(self, filename):
        """
        save emulator using compact encoding scheme
        :param filename: (str) location of file
        :return:
        """
        save_compact_mapping(self.get_compact_mapping(), filename)
