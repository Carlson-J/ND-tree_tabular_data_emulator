import numpy as np
from .domain_functions import transform_domain, transform_point
from .dtree import DTree
from .compact_mapping import convert_tree, compute_global_index, unpack_global_index, MODEL_CLASS_TYPES
from .model_classes import nd_linear_model, compute_log_transform_weight
from .compact_mapping import CompactMapping, save_compact_mapping, load_compact_mapping
from .parameter_struct import Parameters
import ctypes
from numpy.ctypeslib import ndpointer
import subprocess
import sys
import shutil
import os
import h5py


def build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes, max_test_points=100,
                   relative_error=False, expand_index_domain=True, return_tree=False, error_type='max'):
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
                'transforms': (string) None or 'log', being the transform on the function f. f is first conditioned to
                    be all positive before the log is taken. A variable that saves the transform is also saved.
                }
    :param max_test_points: (int) maximum number of test points to use when computing the error in a cell.
    :param relative_error: use relative error, i.e., (f_interp - f_true)/f_true. This will have issues if you go
        to or close to zero.
    :param expand_index_domain: (bool) Expand the index domain so that the number of points in each dim are equal
        and are of the form (2^k) + 1. This will allow for efficient mapping with no extra cost except that the
        integers needed to represent the values in the encoding array will increase in bytes size.
    :param return_tree: (bool) return tree
    :param error_type: (str) type of error to use. Options: ['L1', 'RMSE', 'max']
    :return: (Emulator, optional-> DTree)
    """
    # check that data is correct format
    assert (type(data['f']) is np.ndarray)
    dims = data['f'].shape
    for key in data:
        assert (data[key].shape == dims)

    # construct a dictionary with all the needed parameters
    tree_parameters = Parameters(max_depth, np.array(spacing), np.array(dims, dtype=int), error_threshold,
                                 np.array(model_classes), max_test_points, relative_error,
                                 np.array(domain, dtype=float), np.array([None]), expand_index_domain)

    # build tree
    tree = DTree(tree_parameters, data, error_type=error_type)
    if return_tree:
        return tree

    # convert to compact storage/mapping scheme
    compact_mapping = convert_tree(tree)

    # create emulator
    emulator = Emulator(compact_mapping)

    return emulator

def load_emulator(filename):
    """
    Load the emulator from a saved compact mapping.
    :param filename: (string) location of the hdf5 file
    :return: (Emulator)
    """
    # load compact mapping
    compact_mapping = load_compact_mapping(filename)
    # create emulator
    return Emulator(compact_mapping)


class Emulator:
    def __init__(self, compact_mapping):
        """
        Build an emulator from a compact mapping data object.
        The emulator will use this data to map inputs to correct cells and perform the corresponding interpolation
        in the cell.
        :param compact_mapping: (CompactMapping)
        """
        self.encoding_array = compact_mapping.encoding_array
        self.point_map = compact_mapping.point_map
        self.params = compact_mapping.params

        # precompute some things
        self.num_dims = len(self.params.dims)
        self.domain = transform_domain(self.params.domain, self.params.spacing)
        self.index_domain = transform_domain(self.params.index_domain, self.params.spacing)
        self.dx = np.zeros([self.num_dims])
        for i in range(self.num_dims):
            self.dx[i] = (self.index_domain[i][1] - self.index_domain[i][0]) / 2 ** self.params.max_depth

    def __call__(self, inputs):
        """
        Compute the  interpolated values at each point in the input array.
        :param inputs: (2d array) each row is a different point
        :return: (array) the function value at each point.
        """
        inputs = np.atleast_2d(inputs)
        assert (inputs.shape[1] == self.num_dims)

        sol = np.zeros([inputs.shape[0]])
        for i, point in enumerate(inputs):
            # find out which model
            weights, index = self.find_model(point)
            # compute fit
            if self.params.model_classes[index]['type'] == 'nd-linear':
                if self.params.model_classes[index]['transforms'] is not None:
                    sol[i] = nd_linear_model(weights, point, transform=self.params.model_classes[index]['transforms'])[0]
                else:
                    sol[i] = nd_linear_model(weights, point)[0]
            else:
                raise ValueError
        return sol

    def unpack_encoding(self, index):
        """
        Unpack the encoding array to get the depth and model type
        :param index (int):
        :return: depth, model type
        """
        c = self.encoding_array[index]
        depth = np.uint64(c & 0b00001111)
        type = np.uint64(c >> 4)
        return depth, type

    # from numba import jit
    # @jit
    def find_model(self, point):
        """
        Find the model weights and the index of model class array it is
        :param point: (array) [x0, x1,...]
        :return: (array) model weights, (int) model type
        """
        # compute cartesian index
        cart_indices = self.compute_cartesian_index(point)
        # determine model type and location
        global_index = compute_global_index(cart_indices, self.params.dims - 1)  # the cells are 1 smaller in each dim
        depth, model_type = self.unpack_encoding(global_index)
        # add extra transform if needed
        if self.params.model_classes[model_type]['transforms'] == "log":
            transform_weights = 1
        else:
            transform_weights = 0
        # compute owning cell's index
        depth_diff = np.zeros(self.num_dims, dtype=np.uint64)
        for i in range(self.num_dims):
            depth_diff[i] = self.params.max_depth - depth
            cart_indices[i] = (cart_indices[i] >> depth_diff[i]) << depth_diff[i]
        # get model weights
        weights = np.zeros(2 ** self.num_dims + 2*self.num_dims + transform_weights)
        # -- Go over each corner of the hyper-cube
        for i in range(2 ** self.num_dims):
            point_coords = cart_indices.copy()
            for j, d in enumerate(f'{i:0{self.num_dims}b}'[::-1]):
                point_coords[j] += 2**depth_diff[j] * int(d)  # make sure it is in the domain
                point_coords[j] = min(point_coords[j], self.params.dims[j] - 1)  # make sure it is in the domain
            # add domain coords
            if i == 0:
                weights[-2*self.num_dims:-self.num_dims] = self.compute_domain_from_indices(point_coords)
            elif i == 2 ** self.num_dims - 1:
                weights[-self.num_dims:] = self.compute_domain_from_indices(point_coords)
            global_index = compute_global_index(point_coords, self.params.dims)
            weights[transform_weights+i] = self.point_map[f'{global_index}']
        if self.params.model_classes[model_type]['transforms'] == "log":
            weights[:transform_weights] = compute_log_transform_weight(weights[transform_weights:2**self.num_dims+transform_weights])
            weights[transform_weights:2**self.num_dims+transform_weights] = np.log10(weights[transform_weights:2**self.num_dims+transform_weights])
        return weights, model_type

    def compute_domain_from_indices(self, point_coords):
        """
        Compute (x0,x1,..,xN) based on the index
        :param point_coords:
        :return:
        """
        output = np.zeros(self.num_dims)
        for i in range(self.num_dims):
            output[i] = self.domain[i][0] + self.dx[i]*point_coords[i]
        return transform_point(output, self.params.spacing, reverse=True)

    def compute_cartesian_index(self, point):
        """
        Compute the cartesian indices
        :param point: (array) the location of the point
        :return: (int)
        """
        EPS = 1e-10
        # do any needed transforms for spacing reasons.
        point_new = transform_point(point, self.params.spacing, reverse=False)
        # compute cartesian coordinate on regular grid of points in the index domain
        coords_cart = np.zeros(self.num_dims, dtype=np.uint)
        for i in range(len(point_new)):
            coords_cart[i] = int(np.floor((point_new[i] - self.index_domain[i][0]) / self.dx[i]))
            # restrict cart index to cells in the domain
            coords_cart[i] = min(max(coords_cart[i], 0), self.params.dims[i]-2)
        return coords_cart

    def get_compact_mapping(self):
        """
        Package and return compact emulator
        :return:
        """
        return CompactMapping(self.encoding_array, self.point_map, self.params)

    def save(self, folder_path, emulator_name, return_file_size=False):
        """
        save emulator using compact encoding scheme
        :param folder_path: (str) location of folder to save emulator in
        :param emulator_name: (str) name of emulator
        :param return_file_size: (bool) return saved file siaze
        :return:
        """
        save_compact_mapping(self.get_compact_mapping(), folder_path, emulator_name, return_file_size=return_file_size)

    def get_cell_locations(self, include_model_type=False):
        """
        Compute the locations of all the corner of the cells.
        :param include_model_type: (bool) return model type at each point as well.
        :return:
        """
        assert(self.num_dims <= 2)
        points = []
        values = []
        model_types = []
        for key in self.point_map.keys():
            points.append(unpack_global_index(int(key), self.params.dims))
            # convert point index to cell index
            tmp = np.array(points[-1])
            tmp = np.array([v if v == 0 else v-1 for v in tmp])
            global_cell_index = compute_global_index(tmp, self.params.dims - 1)
            model_types.append(self.unpack_encoding(global_cell_index)[1])

        # covert point indices to values
        point_indices = np.array(points)
        point_values = []
        for point_index in point_indices:
            point_values.append(self.compute_domain_from_indices(point_index))

        if include_model_type:
            return np.array(point_values), model_types
        return np.array(point_values)


class EmulatorCpp:
    def __init__(self, filename, mapping_filename, extern_name, extern_lib_location):
        """
        A wrapper class for the C++ version of the emulator.
        An extern function needs to be built for each emulator.
        The emulator will use this data to map inputs to correct cells and perform the corresponding interpolation
        in the cell.
        :param filename: (string) Location of hdf5 file that holds the info needed to construct the emulator.
        :param mapping_filename: (string) Location of bin file containing the point mapping.
        :param extern_name: (string) name of the extern C function that goes along with the emulator.
        """
        with h5py.File(filename, 'r') as file:
            self.num_dims = len(file.attrs['dims'])

        # Load C++ emulator
        self.lib = ctypes.cdll.LoadLibrary(extern_lib_location)
        self.setup_emulator = eval(f'self.lib.{extern_name}_emulator_setup')
        self.interpolate = eval(f'self.lib.{extern_name}_emulator_interpolate')
        # self.interpolate_single = eval(f'self.lib.{extern_name}_emulator_interpolate_single')
        # self.interpolate_single_dx1 = eval(f'self.lib.{extern_name}_emulator_interpolate_single_dx1')
        self.free = eval(f'self.lib.{extern_name}_emulator_free')

        # set input and output type
        self.setup_emulator.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.interpolate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        # -- add the number of args
        for i in range(self.num_dims):
            self.interpolate.argtypes.append(ctypes.POINTER(ctypes.c_double))
        self.interpolate.argtypes.append(ctypes.POINTER(ctypes.c_size_t))
        self.interpolate.argtypes.append(ctypes.POINTER(ctypes.c_double))
        self.setup_emulator.restype = ctypes.POINTER(ctypes.c_void_p)

        # load emulator
        self.emulator = ctypes.c_void_p()
        self.setup_emulator(filename.encode("UTF-8"), mapping_filename.encode("UTF-8"), self.emulator)

    def __del__(self):
        """
        Destructor for emulator. This is needed to free the memory used by the C++ objects.
        :return:
        """
        try:
            self.free(self.emulator)
        except AttributeError:
            return

    def __call__(self, inputs):
        """
        Compute the  interpolated values at each point in the input array.
        :param inputs: (2d array) each row is a different point
        :return: (array) the function value at each point.
        """
        inputs = np.array(inputs).T
        # allocate memory to work in
        num_points = inputs.shape[1]
        output = np.zeros(num_points)
        # construct 2d array for C++ function
        # transform inputs
        ptrs = [inputs[i, :].ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for i in range(inputs.shape[0])]

        # do interpolation
        self.interpolate(self.emulator, *ptrs, ctypes.byref(ctypes.c_size_t(num_points)),
                         output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        return output


def make_cpp_emulator(save_directory, emulator_name, cpp_source_dir):
    """
    Craete the cpp library for the emulator. The save directory should have the *_cpp_params.h file in it already.
    :param save_directory: (str) where the params and table files are located and where to save the lib file
    :param emulator_name: (str) name of the emulator
    :param cpp_source_dir: (str) the location of the cpp_emulator directory
    :return:
    """
    assert (os.path.isfile(save_directory + '/' + emulator_name + "_cpp_params.h"))
    # create C++ shared library to accompany it
    # -- Create tmp build directory (the old one will be removed)
    tmp_dir = "./tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    # -- create build files using cmake
    if sys.platform.startswith('win'):
        shell = True
    else:
        shell = False
    # -- Put include file that is needed to compile the library for the specific table
    shutil.copy(save_directory + '/' + emulator_name + "_cpp_params.h", cpp_source_dir + '/emulator/table_params.h')
    # -- Setup env
    new_env = dict(os.environ)
    new_env['EMULATOR_NAME'] = emulator_name
    # -- Create build files
    current_dir = os.getcwd()
    os.chdir(tmp_dir)
    cmakeCmd = ["cmake", f"{current_dir+'/'+cpp_source_dir}", '.', '-DCMAKE_BUILD_TYPE=RelWithDebInfo'] #RelWithDebInfo
    subprocess.check_call(cmakeCmd, stderr=subprocess.STDOUT, shell=shell, env=new_env)
    # -- build C++ code
    cmakeCmd = ["cmake", '--build', '.', '--target', 'ND_emulator_lib', '--verbose']
    subprocess.check_call(cmakeCmd, stderr=subprocess.STDOUT, shell=shell,  env=new_env)
    cmakeCmd = ["cmake", '--build', '.', '--target', 'make_mapping', '--verbose']
    subprocess.check_call(cmakeCmd, stderr=subprocess.STDOUT, shell=shell,  env=new_env)
    # -- build mapping
    os.chdir(current_dir)
    with h5py.File(save_directory + '/' + emulator_name + "_table.hdf5") as file:
        num_points = file["/mapping/indexing"].shape[0]
    cmakeCmd = [tmp_dir+"/make_mapping",save_directory + '/' + emulator_name + "_table.hdf5" , save_directory + '/' + emulator_name +"_mapping.bin", f'{num_points}']
    subprocess.check_call(cmakeCmd, stderr=subprocess.STDOUT, shell=shell,  env=new_env)
    # -- move C++ library to install folder
    shutil.copy(tmp_dir + f'/{emulator_name}.so', save_directory + f'/{emulator_name}.so')

    # create readme file with the names of the function calls to used with the shared libraries
    with open(save_directory + '/README.md', 'w') as file:
        str = f"""# README for the *{emulator_name}* emulator
    This folder should contain three file, 
        *{emulator_name}_table.hdf5*: Contains the data for the compact emulator
        *{emulator_name}_cpp_params.h*: Contains the #define's used when creating the C++ lib
        *{emulator_name}_lib.so*: C++ lib that has C extern functions that can be called to make, use, and destroy the
            emulator. 

    The function names in *{emulator_name}_lib.so* that can be called are named based on the emulators name and are as follows:
        *{emulator_name}_emulator_setup*: Constructs an emulator C++ object.
        *{emulator_name}_emulator_interpolate*: Calls the emulator object for interpolation.
        *{emulator_name}_emulator_free*: Frees the memory allocated by *{emulator_name}_emulator_setup*.

    """
        file.write(str)
