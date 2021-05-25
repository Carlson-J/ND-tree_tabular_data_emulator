import numpy as np


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


class CompactMapping:
    def __init__(self):
        # initialize array names
        self.encoding_array = None
        self.index_array = None
        self.offsets = None
        self.model_arrays = None


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