import numpy as np
from dataclasses import dataclass
import h5py
from os import path
from .parameter_struct import Parameters

# mapping from strings to ints for saving in hdf5 file
SPACING_TYPES = ['linear', 'log']
MODEL_CLASS_TYPES = ['nd-linear']
TRANSFORMS = ['linear', 'log']


def type_header_conversion(type):
    """
    Convert the numpy type to a string type that will be used in the C++ emulator
    :param type: np unsigned int type
    :return:
    """
    if type is np.uint8:
        return "unsigned char"
    elif type is np.uint16:
        return "unsigned short"
    elif type is np.uint32:
        return "unsigned long"
    elif type is np.uint64:
        return "unsigned long long"
    else:
        raise ValueError("Unknown type")


def save_header_file(folder_path, emulator_name, encoding_type, indexing_type, num_dims, num_model_classes, num_models, model_array_size):

    # create file to hold define constants
    filename = folder_path + '/' + emulator_name
    if filename[-3:] == '.h5':
        save_name = filename[:-3] + "_cpp_params.h"
    elif filename[-5:] == '.hdf5':
        save_name = filename[:-5] + "_cpp_params.h"
    else:
        save_name = filename + "_cpp_params.h"

    # save template parameters for the given table
    with open(save_name, 'w') as file:
        file.write("#undef ND_TREE_EMULATOR_TYPE\n")
        file.write("#define ND_TREE_EMULATOR_TYPE ")
        file.write(f"{type_header_conversion(encoding_type)}, ")
        file.write(f"{type_header_conversion(indexing_type)}, ")
        file.write(f"{num_model_classes}, ")
        file.write(f"{num_dims}, ")
        file.write(f"{num_models}, ")
        file.write(f"{model_array_size}\n")
        # create function name stuff
        file.write(f"#define ND_TREE_EMULATOR_NAME_SETUP {emulator_name}_emulator_setup\n")
        file.write(f"#define ND_TREE_EMULATOR_NAME_INTERPOLATE {emulator_name}_emulator_interpolate\n")
        file.write(f"#define ND_TREE_EMULATOR_NAME_FREE {emulator_name}_emulator_free\n")


def save_compact_mapping(compact_mapping, folder_path, emulator_name):
    """
    Save the compact mapping array in an hdf5 file.
    :param compact_mapping: (CompactMapping)
    :param folder_path: (str)
    :param emulator_name: (str)
    :return:
    """
    # Save the mapping using the smallest int size needed.
    encode_dtype = find_int_type(np.max(compact_mapping.encoding_array))
    index_dtype = find_int_type(np.max(compact_mapping.index_array))
    encoding_compressed = np.ndarray.astype(compact_mapping.encoding_array, dtype=encode_dtype)
    index_compressed = np.ndarray.astype(compact_mapping.index_array, dtype=index_dtype)

    # save header file for C++ compiler
    save_header_file(folder_path, emulator_name, encode_dtype, index_dtype, len(compact_mapping.params.dims),
                     len(compact_mapping.model_arrays), sum([len(v) for v in compact_mapping.model_arrays]),
                     sum([sum([len(model) for model in v]) for v in compact_mapping.model_arrays]))

    # Save arrays as hdf5 files
    filename = folder_path + '/' + emulator_name + '_table.hdf5'
    with h5py.File(filename, 'w') as file:
        # Save models
        model_group = file.create_group('models')
        for j, models_array in enumerate(compact_mapping.model_arrays):
            if len(models_array) > 0:
                dset_model = model_group.create_dataset(f'{j}_' + compact_mapping.params.model_classes[j]['type']
                                                        , models_array.shape, dtype='d')
                dset_model[...] = models_array[...]
                dset_model.attrs['model_type'] = compact_mapping.params.model_classes[j]['type'].encode("ascii")
                # # save transforms as a string
                # transforms_index = []
                # for t in compact_mapping.params.model_classes[j]['transforms']:
                #     transforms_index.append(TRANSFORMS.index(t))
                # dset_model.attrs['transforms'] = transforms_index
        # Save mapping
        mapping_group = file.create_group('mapping')
        mapping_group.create_dataset("encoding", data=encoding_compressed, dtype=encode_dtype)
        mapping_group.create_dataset("indexing", data=index_compressed, dtype=index_dtype)
        mapping_group.create_dataset("offsets", data=compact_mapping.offsets)

        # save parameters
        file.attrs['max_depth'] = compact_mapping.params.max_depth
        file.attrs['spacing'] = [SPACING_TYPES.index(s) for s in compact_mapping.params.spacing]
        file.attrs['dims'] = compact_mapping.params.dims
        file.attrs['error_threshold'] = compact_mapping.params.error_threshold
        file.attrs['model_classes'] = [MODEL_CLASS_TYPES.index(s['type']) for s in compact_mapping.params.model_classes]
        file.attrs['max_test_points'] = compact_mapping.params.max_test_points
        file.attrs['relative_error'] = compact_mapping.params.relative_error
        file.attrs['domain'] = compact_mapping.params.domain
        #
        # parameter_group = file.create_group('parameters')
        # parameter_group.create_dataset('domain', data=compact_mapping.params.domain, dtype=float)
        #
        # parameter_group.create_dataset('spacing', data=compact_mapping.params.spacing, dtype=float)
        #
        # domain_group = file.create_group('emulator_properties')
        # domain_group.create_dataset("depth", data=compact_mapping.params.max_depth)
        # domain_group.create_dataset("domain", data=compact_mapping.params.domain)
        # spacing_index = [TRANSFORMS.index(t) for t in compact_mapping.params.spacing]
        # domain_group.create_dataset("spacing", data=spacing_index)
        file.close()


def load_compact_mapping(filename, return_file_size=False):
    """
    Loads the compact mapping from an hdf5 file
    :param filename: (str)
    :return: (CompactMapping)
    """
    assert (path.exists(filename))
    # save file size
    file_size_bytes = path.getsize(filename)
    # load data
    with h5py.File(filename, 'r') as file:
        # load model arrays
        model_arrays = []
        model_classes = []
        for key in file['models'].keys():
            model_arrays.append(file['models'][key][...])
            model_classes.append(file['models'][key].attrs['model_type'])

        # load mapping arrays
        encoding_array = file['mapping']['encoding'][...]
        index_array = file['mapping']['indexing'][...]
        offsets = file['mapping']['offsets'][...]

        # load parameters
        max_depth = file.attrs['max_depth']
        spacing = [SPACING_TYPES[s] for s in file.attrs['spacing']]
        dims = file.attrs['dims']
        error_threshold = file.attrs['error_threshold']
        model_classes = [{'type':MODEL_CLASS_TYPES[s]} for s in file.attrs['model_classes']]
        max_test_points = file.attrs['max_test_points']
        relative_error = file.attrs['relative_error']
        domain = file.attrs['domain']
        params = Parameters(max_depth, np.array(spacing), np.array(dims), error_threshold, np.array(model_classes),
                            max_test_points, relative_error, np.array(domain))
        file.close()

    # Return compact emulator
    compact_mapping = CompactMapping(encoding_array, index_array, offsets, model_arrays, params)
    if return_file_size:
        return compact_mapping, file_size_bytes
    else:
        return compact_mapping


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


def compute_encoding_index(leaf, params):
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
    n_dims = len(params.dims)
    leaf_depth = len(leaf['id'])
    # Use bitwise operations to find the index of the leaf node.
    for v in leaf['id']:
        index = index << n_dims
        index = index | v
    # increase the index by one and find the smallest index of all that node's children
    index += 1
    index = index << (params.max_depth - leaf_depth) * n_dims
    return index


@dataclass
class CompactMapping:
    encoding_array: np.array
    index_array: np.array
    offsets: np.array
    model_arrays: list
    params: Parameters


def convert_tree(tree):
    """
    Convert the tree to a computationally efficient mapping scheme that can easily be saved and queried.
    See quadtree paper (Carlson:2021)
    :param tree: (DTree)
    :return: (CompactMapping) a dataclass that holds the compact mapping
    """
    # compute leaf nodes
    leaves = tree.get_leaves()
    params = tree.get_params()
    # Create model arrays
    model_arrays = [[]] * len(tree.params.model_classes)

    for leaf in leaves:
        model_arrays[list(params.model_classes).index(leaf['model']['type'])].append(leaf['model']['weights'])

    # create encoding array
    encoding_array = np.zeros([len(leaves)], dtype=int)
    index_array = np.zeros([len(leaves)], dtype=int)
    counters = np.zeros([len(model_arrays)])
    offsets = [0] + [len(model_arrays[i]) for i in range(len(model_arrays) - 1)]
    for i in range(len(leaves)):
        # compute encoding array index
        encoding_array[i] = compute_encoding_index(leaves[i], params)
        # compute index-array index
        # # determine model type index
        type_index = list(params.model_classes).index(leaves[i]['model']['type'])
        index_array[i] = counters[type_index] + offsets[type_index]
        counters[type_index] += 1

    # convert model arrays to a list of np arrays
    for i in range(len(model_arrays)):
        model_arrays[i] = np.array(model_arrays[i])

    # save in data object and return
    return CompactMapping(encoding_array, index_array, offsets, model_arrays, params)
