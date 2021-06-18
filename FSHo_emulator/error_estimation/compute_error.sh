#!/bin/bash

source ../../../../load_env

conda activate nd_emulator

export PYTHONPATH=${PYTHONPATH}:/mnt/home/carls502/ANL/ND-tree_tabular_data_emulator/

python3 compute_errors.py


