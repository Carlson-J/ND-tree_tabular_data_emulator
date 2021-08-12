import numpy as np
from dataclasses import dataclass
import h5py
from os import path
from .parameter_struct import Parameters

# mapping from strings to ints for saving in hdf5 file
SPACING_TYPES = ['linear', 'log']
MODEL_CLASS_TYPES = ['nd-linear']
TRANSFORMS = [None, 'log']


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


def save_header_file(folder_path, emulator_name, indexing_type, num_dims, num_model_classes,
                     mapping_array_size, encoding_array_size):
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
        file.write("#undef ND_TREE_EMULATOR_NAME_SETUP\n")
        file.write("#undef ND_TREE_EMULATOR_NAME_INTERPOLATE\n")
        file.write("#undef ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE\n")
        file.write("#undef ND_TREE_EMULATOR_NAME_FREE\n")
        file.write("#define ND_TREE_EMULATOR_TYPE ")
        file.write(f"{type_header_conversion(indexing_type)}, ")
        file.write(f"{num_model_classes}, ")
        file.write(f"{num_dims}, ")
        # file.write(f"{model_array_size}\n, ")
        file.write(f"{mapping_array_size}, ")
        file.write(f"{encoding_array_size}\n")
        # create function name stuff
        file.write(f"#define ND_TREE_EMULATOR_NAME_SETUP {emulator_name}_emulator_setup\n")
        file.write(f"#define ND_TREE_EMULATOR_NAME_INTERPOLATE {emulator_name}_emulator_interpolate\n")
        file.write(f"#define ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE {emulator_name}_emulator_interpolate_single\n")
        file.write(f"#define ND_TREE_EMULATOR_NAME_FREE {emulator_name}_emulator_free\n")


def save_compact_mapping(compact_mapping, folder_path, emulator_name, return_file_size=False):
    """
    Save the compact mapping array in an hdf5 file.
    :param compact_mapping: (CompactMapping)
    :param folder_path: (str)
    :param emulator_name: (str)
    :param return_file_size: (bool) return the size of the saved mapping
    :return:
    """
    # convert mapping dict into 2 arrays
    index_array = []
    node_values = []
    for key in compact_mapping.point_map.keys():
        index_array.append(int(key))
        node_values.append(compact_mapping.point_map[key])
    index_array = np.array(index_array)
    node_values = np.array(node_values)
    # Save the mapping using the smallest int size needed.
    index_dtype = find_int_type(np.max(index_array))
    index_compressed = np.ndarray.astype(index_array, dtype=index_dtype)

    # save header file for C++ compiler
    save_header_file(folder_path, emulator_name, index_dtype, len(compact_mapping.params.dims),
                     1,  # TODO: Remove this hard coded answer of 1 for number of model classes
                     len(node_values), len(compact_mapping.encoding_array))

    # Save arrays as hdf5 files
    filename = folder_path + '/' + emulator_name + '_table.hdf5'
    with h5py.File(filename, 'w') as file:
        # Save models
        model_group = file.create_group('models')
        # for j, models_array in enumerate(compact_mapping.model_arrays):
        #     if len(models_array) > 0:
        #         dset_model = model_group.create_dataset(f'{j}_' + compact_mapping.params.model_classes[j]['type']
        #                                                 , models_array.shape, dtype='d')
        #         dset_model[...] = models_array[...]
        #         dset_model.attrs['model_type'] = compact_mapping.params.model_classes[j]['type'].encode("ascii")
        #         # # save transforms as a string
        #         # transforms_index = []
        #         # for t in compact_mapping.params.model_classes[j]['transforms']:
        #         #     transforms_index.append(TRANSFORMS.index(t))
        #         # dset_model.attrs['transforms'] = transforms_index
        # Save mapping
        mapping_group = file.create_group('mapping')
        mapping_group.create_dataset("node_values", data=node_values, dtype=np.double)
        mapping_group.create_dataset("encoding", data=compact_mapping.encoding_array, dtype=np.byte)
        mapping_group.create_dataset("indexing", data=index_compressed, dtype=index_dtype)

        # save parameters
        file.attrs['max_depth'] = compact_mapping.params.max_depth
        file.attrs['spacing'] = [SPACING_TYPES.index(s) for s in compact_mapping.params.spacing]
        file.attrs['error_threshold'] = compact_mapping.params.error_threshold
        file.attrs['model_classes'] = [MODEL_CLASS_TYPES.index(s['type']) for s in compact_mapping.params.model_classes]
        file.attrs['transforms'] = [TRANSFORMS.index(s['transforms']) for s in compact_mapping.params.model_classes]
        file.attrs['max_test_points'] = compact_mapping.params.max_test_points
        file.attrs['relative_error'] = compact_mapping.params.relative_error
        file.attrs['domain'] = compact_mapping.params.domain
        file.attrs['index_domain'] = compact_mapping.params.index_domain
        file.attrs['expand_index_domain'] = compact_mapping.params.expand_index_domain
        file.attrs['dims'] = compact_mapping.params.dims
        file.close()
    if return_file_size:
        return path.getsize(filename)


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
        error_threshold = file.attrs['error_threshold']
        model_classes = [{'type': MODEL_CLASS_TYPES[m], 'transforms': TRANSFORMS[t]} for m, t in
                         zip(file.attrs['model_classes'], file.attrs['transforms'])]
        max_test_points = file.attrs['max_test_points']
        relative_error = file.attrs['relative_error']
        domain = file.attrs['domain']
        index_domain = file.attrs['index_domain']
        expand_index_domain = file.attrs['expand_index_domain']
        params = Parameters(max_depth, np.array(spacing), np.array([2 ** max_depth for i in range(len(spacing))]),
                            error_threshold, np.array(model_classes), max_test_points, relative_error, np.array(domain),
                            np.array(index_domain), expand_index_domain)
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
    UNSIGNED_INT_64BIT_SIZE = 18446744073709551615
    if size < UNSIGNED_INT_8BIT_SIZE:
        dtype = np.uint8
    elif size < UNSIGNED_INT_16BIT_SIZE:
        dtype = np.uint16
    elif size < UNSIGNED_INT_32BIT_SIZE:
        dtype = np.uint32
    elif size < UNSIGNED_INT_64BIT_SIZE:
        dtype = np.uint64
    else:
        raise ValueError(f"Number of models exceeds largest int! Size needed = {size}")
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
    point_map: dict
    params: Parameters


def convert_model_class_to_string(model_class):
    """
    Dictionary containing the model class type and transforms
    :param model_class: (dict)
    :return: (string) unique name of model class
    """
    if 'transforms' in model_class and model_class['transforms'] is not None:
        return model_class['type'] + "___" + model_class['transforms']
    else:
        return model_class['type']


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

    # change dims if max depth was not achieved
    depth_diff = tree.max_index_depth - tree.achieved_depth
    dims_cells = np.array([np.ceil((dim-1)/float(2**depth_diff)) for dim in tree.params.dims], dtype=np.uint)
    dims_points = dims_cells + 1

    # create encoding array
    encoding_array = np.empty(dims_cells, dtype=np.byte)
    encoding_array[...] = 0

    # Create model arrays
    model_classes_str = []
    for model_class in tree.params.model_classes:
        model_classes_str.append(convert_model_class_to_string(model_class))

    # fill arrays
    point_map = {}
    for leaf in leaves:
        if leaf['model']['type'] is not None:
            model_string = convert_model_class_to_string(leaf['model']['type'])
            # offset mask by one and adjust step for max depth
            mask = list(leaf['mask'])
            encoding_mask = []
            for i in range(len(mask)):
                mask[i] = slice(mask[i].start, mask[i].stop - 1, 2 ** depth_diff)
                encoding_mask.append(slice(int(np.ceil(mask[i].start/2 ** depth_diff)),
                                           int(np.ceil(mask[i].stop/2 ** depth_diff))))
            mask = tuple(mask)
            encoding_mask = tuple(encoding_mask)
            # save cartesian index of all points and their values is separate arrays
            for i in range(2 ** tree.num_dims):
                cart_indices = np.zeros(tree.num_dims, dtype=np.int)
                data_indices = np.zeros(tree.num_dims, dtype=np.int)
                for j, s in enumerate(f'{i:0{tree.num_dims}b}'[::-1]):
                    if s == '0':
                        # assert (mask[j].start % mask[j].step == 0)
                        cart_indices[j] = mask[j].start / mask[j].step
                        data_indices[j] = mask[j].start
                    elif s == '1':
                        # assert (mask[j].stop % mask[j].step == 0)
                        cart_indices[j] = max(mask[j].stop / mask[j].step, 1)
                        data_indices[j] = mask[j].stop
                    else:
                        raise ValueError
                global_index = compute_global_index(cart_indices, dims_points)
                if f'{global_index}' in point_map:
                    assert(point_map[f'{global_index}'] == tree.data['f'][tuple(data_indices)])
                else:
                    point_map[f'{global_index}'] = tree.data['f'][tuple(data_indices)]
            # determine byte for given size and type
            c = (list(model_classes_str).index(model_string) << 4) | len(leaf['id'])
            encoding_array[encoding_mask] = c
            # to decode use depth = ord(c) & 0b00001111; type = ord(c) >> 4
    # change params to reflect the new index space
    new_params = Parameters(params.max_depth, params.spacing, dims_points, params.error_threshold,
                            params.model_classes, params.max_test_points, params.relative_error,
                            params.domain, params.index_domain, params.expand_index_domain)

    # save in data object and return
    return CompactMapping(encoding_array.flatten(), point_map, new_params)


def compute_global_index(cart_indices, dims):
    """
    Compute the global index given the dims and nd-index
    :param cart_indices: list of coordinates
    :param dims: shape of global array
    :return:
    """
    tmp = cart_indices.copy()
    for j in range(1, len(dims)):
        tmp[:-j] *= dims[-j]
    return sum(tmp)


def unpack_global_index(global_index, dims):
    """
    Compute the cartesian index based on the global one
    :param global_index: (uint)
    :param dims: list (uint)
    :return:
    """
    cart_indices = np.zeros_like(dims)
    cart_indices[-1] = global_index % dims[-1]
    for i in range(1, len(dims)):
        cart_indices[i] = (global_index // np.prod(dims[-i:])) % dims[-i]
    return cart_indices
