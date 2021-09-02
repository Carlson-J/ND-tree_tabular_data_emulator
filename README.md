# ND-tree_tabular_data_emulator
A memory efficient emulator for tabular data, i.e. lookup tables, that utilizes an ND-tree domain decomposition. Typical lookup tables save function values at regularly spaced intervals, e.g., linearly-spaced or log-spaced. These tables are then interpolated over during runtime to return an approximate value at the desired location. Because the point spacing is fixed over the domain this can lead to varying degrees of interpolation accuracy throughout the table's domain.

To reduce the variability of interpolation error over the domain and reduce the size of the table we use an ND-tree decomposition of the domain. The decomposition is do such that the error over the domain is as close to some desired threshold as possible, given the constraint of the input data and type of interpolation performed.

The emulator is built using the python, but there is a python and C++ version of the ND-emulator for calling a built emulator. The C++ version is build for speed and is thread safe. Furthermore, there are python and fortran bindings for the C++ version. To use them, simply build the C++ emulator into a shared library and follow the call the C extern functions in the desired application.

Different types of interpolation methods are possible and will be chosen on a cell-by-cell basis based on which gives a lower expected error. Currently a nd-linear interpolation in log-space and linear-space are implemented in the python versions, and only the linear-space nd-linear interpolation is available in the c++ version.

## Initializing the repo
The emulator uses multiple submodule, which must be loaded when the repo is cloned. Load with
```
git submodule update --init --recursive --remote
```
# Using the ND-Emulator
An example of how the `build_emulator` function is called is shown in the `build_emulator.py` file. 

## Building an Emulator
In order to create an emulator training data is needed. 
We store the data in a python dictionary, with the format:
```
data = {'f': nd_data_array}
```
If other data, such as derivative information, needs to be included for a given model class, i.e. type of interpolation scheme, you can simply add it to the dictionary, e.g.,
```
data = {'f': nd_data_array,
        'dfdx': nd_data_array}
```
To build the emulator you must also specify the maximum depth of the tree, the domain over which the data is defined, the spacing of the data (it must be regularly spaced), the desired error threshold, and the model classes you wish to use.
```
max_depth = 5                           # There will be at most 5 refinements
domain = [[0.0, 1.0], [0.0, 2.0]]       # a 2d rectangular domain with corners at (0,1) and (0,2)
spacing = ['linear', 'log']             # points are spaced linearly in the first dim and in log-space in the second dim 
error_threshold = 1e-2                  # target interpolation error
model_classes = [{'type': 'nd-linear'}, {'type': 'nd-linear', transforms='log'}]        # The nd-linear model class done normally and after doing a log transform

```
You then build the emulator by calling
```
emulator = build_emulator(data, max_depth, domain, spacing, error_threshold, model_classes)
```

## Saving the emulator

## Building a C++ library for the Emulator

## Using an Emulator
### Python
Loading an emulator in python is simple. You can either load it directly into python or load a C++ library.


The emulator can be called directly by passing an nd-array of inputs to the emulator, as shown below,
```
# 2d example of calling emulator
x = np.linspace(0,1,10)
y = np.linspace(0,1,10)
inputs = np.array([x,y]).T
output = emulator(inputs)
```
This works for either an emulator 
### C++

### Fortran

## Calling the emulator


