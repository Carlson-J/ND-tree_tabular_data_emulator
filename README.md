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
Currently, this data needs to be in a very specific format:
