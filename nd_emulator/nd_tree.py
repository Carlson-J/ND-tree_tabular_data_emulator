import numpy as np
from .model_classes import fit_nd_linear_model, nd_linear_model
from .mask import create_mask
import h5py


def find_int_type(size):
    """
    Return the numpy data type for the smallest unsigned int needed for the input size
    :param size:
    :return:
    """
    UNSIGNED_INT_8BIT_SIZE = 255
    UNSIGNED_INT_16BIT_SIZE = 65535
    UNSIGNED_INT_32BIT_SIZE = 4294967295
    if size < UNSIGNED_INT_8BIT_SIZE:
        dtype = np.uint8
    elif size < UNSIGNED_INT_16BIT_SIZE:
        dtype = np.uint16
    elif size < UNSIGNED_INT_32BIT_SIZE:
        dtype = np.uint32
    else:
        dtype = np.uint64
    return dtype


def load_emulator(file_path):
    """
    Loads an ND_Tree emulator from an hdf5 file
    :param file_path: (string) path to save file
    :return:
    """


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


def transform_point(point, spacings, reverse=False):
    """
    Transform the point so that the spacing is linear, e.g., for log spacing we take the log. For linear we do nothing.
    The original input array is not modified
    :param point: (array) [x0, x1, ...]
    :param spacings: (list) The spacing, either 'linear' or 'log', of each dimension, e.g., ['linear', 'linear', 'log']
    :param reverse: (bool) undo spacing transform
    :return: transformed point
    """
    new_point = point.copy()
    for i in range(len(spacings)):
        if spacings[i] == 'linear':
            continue
        elif spacings[i] == 'log':
            if reverse:
                new_point[i] = 10 ** (new_point[i])
            else:
                new_point[i] = np.log10(new_point[i])
    return new_point


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


def create_children_nodes(node, spacing, global_domain, global_dims):
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
    :param global_domain: (2d-array) [[x0_lo, x0_hi], [x1_lo, x1_hi],... [xN_lo, xN_hi]] the upper and lower bound of each
            dimension of the global domain
    :param global_dims: (array) Number of entries in each dimension
    :return: None
    """
    num_dims = len(node['domain'])
    assert (len(global_domain) == num_dims)
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
            'mask': create_mask(domain_new, global_domain, global_dims, spacing),
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
        # constants needed for saving and loading things
        self.TRANSFORMS = ['linear', 'log']
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
            'id': [],
            'model': None,
            'mask': create_mask(domain, domain, dims, spacing),
            'error': None
        }
        # save parameters
        self.max_depth = max_depth
        self.achieved_depth = 0
        self.spacing = spacing
        self.dims = dims
        self.error_threshold = error_threshold
        self.model_classes = model_classes
        self.domain_spacings = compute_ranges(domain, spacing, dims)
        self.data = data
        self.max_test_points = max_test_points
        self.relative_error = relative_error
        self.leaves = None
        self.domain = domain

        # initialize array names
        self.encoding_array = None
        self.index_array = None
        self.offsets = None
        self.model_arrays = None

        # Train emulator
        self.refine_region(self.tree)

        # Convert emulator into computationally efficient mapping
        self.convert_tree()

    def refine_region(self, node):

        # train region
        fit, error = self.fit_region(node)

        # check error and depth
        if error >= self.error_threshold and self.max_depth > len(node['id']):
            # create children nodes
            create_children_nodes(node, self.spacing, self.domain, self.dims)
            if len(node['children'][0]['id']) > self.achieved_depth:
                self.achieved_depth = len(node['children'][0]['id'])
            # refine children nodes
            for i in range(2**len(self.dims)):
                self.refine_region(node['children'][i])
        else:
            node['model'] = fit
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
            else:
                raise ValueError("Unknown model type: {model_class}")
            # check if fit is best so far
            if err < current_error:
                current_error = err
                best_fit = fit
        # Return best fit and its error.
        return best_fit, current_error

    def get_leaves(self, node, leaf_list):
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
                self.get_leaves(child, leaf_list)

    def compute_encoding_index(self, leaf):
        """
        Compute the linear index in the nd-tree index space of the endcoding value, i.e., compute the index of the
        deepest possible node that +1 away from this node or the largest index of its decadents if it is not at max
        depth.
        Exampl1: (max depth 2 tree)
        If we get the index 0 as our leaf node we want to return that
                root
                 │
          ┌──────┴──────┐
          │             │1
          0         ┌───┴───┐
          ▲         │       │
          │        10      11
        leaf        ▲
                    │
                  encoding
                   index
        :return: (int) encoding index
        """
        index = 0
        n_dims = len(self.dims)
        leaf_depth = len(leaf['id'])
        # Use bitwise operations to find the index of the leaf node.
        for v in leaf['id']:
            index = index << n_dims
            index = index | v
        # increase the index by one and find the smallest index of all that node's children
        index += 1
        index = index << (self.achieved_depth - leaf_depth) * n_dims
        return index

    def __call__(self, inputs):
        """
        Compute the  interpolated values at each point in the input array.
        :param inputs: (2d array) each row is a different point
        :return: (array) the function value at each point.
        """
        inputs = np.atleast_2d(inputs)
        assert(inputs.shape[1] == len(self.dims))
        # precompute some things
        domain = transform_domain(self.domain, self.spacing)
        dx = np.zeros([len(self.dims)])
        for i in range(len(self.dims)):
            dx[i] = (domain[i][1] - domain[i][0]) / 2**(self.achieved_depth*len(self.dims))
        sol = np.zeros([inputs.shape[0]])
        for i, point in enumerate(inputs):
            # find out which model
            weights, index = self.find_model(point, dx)
            # compute fit
            if self.model_classes[index]['type'] == 'nd-linear':
                sol[i] = nd_linear_model(weights, point)[0]
            else:
                raise ValueError
        return sol

    def find_model(self, point, dx):
        """
        Find the model weights and the index of model class array it is
        :param point: (array) [x0, x1,...]
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
        point_new = transform_point(point, self.spacing, reverse=False)
        domain = transform_domain(self.domain, self.spacing)
        # compute cartesian coordinate on regular grid of points
        coords_cart = np.zeros(len(self.dims), dtype=int)
        for i in range(len(point_new)):
            coords_cart[i] = int(np.floor((point_new[i] - domain[i][0]) / dx[i]))
            # if the right most point would index into a cell that is not their move it back a cell.
            if coords_cart[i] == 2**self.achieved_depth:
                coords_cart[i] -= 1
        # convert to tree index
        index = 0
        for i in range(self.achieved_depth):
            for j in range(len(point)):
                index = (index << 1) | ((coords_cart[::-1][j] >> (self.achieved_depth - i - 1)) & 1)
        return index

    def convert_tree(self):
        """
        Convert the tree to a computationally efficient mapping scheme that can easily be saved and queried.
        See quadtree paper (Carlson:2021)
        :return:
        """
        # Create model arrays
        model_arrays = [[]]*len(self.model_classes)
        if self.leaves is None:
            if self.achieved_depth == 0:
                self.leaves = [self.tree]
            else:
                self.leaves = []
                self.get_leaves(self.tree, self.leaves)
        for leaf in self.leaves:
            model_arrays[self.model_classes.index(leaf['model']['type'])].append(leaf['model']['weights'])

        # create encoding array
        self.encoding_array = np.zeros([len(self.leaves)], dtype=int)
        self.index_array = np.zeros([len(self.leaves)], dtype=int)
        counters = np.zeros([len(model_arrays)])
        self.offsets = [0] + [len(model_arrays[i]) for i in range(len(model_arrays) - 1)]
        for i in range(len(self.leaves)):
            # compute encoding array index
            self.encoding_array[i] = self.compute_encoding_index(self.leaves[i])
            # compute index-array index
            # # determine model type index
            type_index = self.model_classes.index(self.leaves[i]['model']['type'])
            self.index_array[i] = counters[type_index] + self.offsets[type_index]
            counters[type_index] += 1

        # save arrays
        self.model_arrays = []
        for model_array in model_arrays:
            self.model_arrays.append(np.array(model_array))

    def save(self, filename):
        # Save the mapping using the smallest int size needed.
        encode_dtype = find_int_type(2**(len(self.dims)*self.achieved_depth))
        value_dtype = find_int_type(len(self.leaves))
        encoding_compressed = np.ndarray.astype(self.encoding_array, dtype=encode_dtype)
        index_compressed = np.ndarray.astype(self.index_array, dtype=value_dtype)

        # Save arrays as hdf5 files
        with h5py.File(filename, 'w') as file:
            # Save models
            model_group = file.create_group('models')
            for j, models_array in enumerate(self.model_arrays):
                if len(models_array) > 0:
                    dset_model = model_group.create_dataset(self.model_classes[j]['type']+f'_{j}'
                                                            , models_array.shape, dtype='d')
                    dset_model[...] = models_array[...]
                    dset_model.attrs['model_type'] = self.model_classes[j]['type'].encode("ascii")
                    # save transforms as a string
                    transforms_index = []
                    for t in self.model_classes[j]['transforms']:
                        transforms_index.append(self.TRANSFORMS.index(t))
                    dset_model.attrs['transforms'] = transforms_index
                    dset_model.attrs['offset'] = self.offsets[j]
            # Save mapping
            mapping_group = file.create_group('mapping')
            dset_encoding = mapping_group.create_dataset("encoding", data=encoding_compressed, dtype=encode_dtype)
            dset_value = mapping_group.create_dataset("indexing", data=index_compressed, dtype=value_dtype)
            # save domain
            domain_group = file.create_group('emulator_properties')
            domain_group.create_dataset("depth", data=self.achieved_depth)
            domain_group.create_dataset("domain", data=self.domain)
            spacing_index = [self.TRANSFORMS.index(t) for t in self.spacing]
            domain_group.create_dataset("spacing", data=spacing_index)
            file.close()