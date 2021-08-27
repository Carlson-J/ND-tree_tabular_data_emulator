# ND-tree_tabular_data_emulator
A memory efficient emulator for tabular data that utilizes an ND-tree domain decomposition. 

## Initializing the repo
The emulator uses multiple submodule, which must be loaded when the repo is cloned. Load with
```
git submodule update --init --recursive --remote
```
# Building an Emulator
## Loading the Training Data
In order to create an emulator training data is needed. 
We store the data in a python dictionary, with the format:
```
data = {'f': nd_data_array}
```
If other data, such as derivative information, needs to be included for the model classes you are using simply add it to the dictionary, e.g.,
```
data = {'f': nd_data_array,
        'dfdx': nd_data_array}
```

## Calling the emulator
An example of how the `build_emulator` function is called is shown in the `build_emulator.py` file. 
