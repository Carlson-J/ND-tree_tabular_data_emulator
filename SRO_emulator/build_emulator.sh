#!/bin/bash
#SBATCH --time=02:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 1 processor core(s) per node 
#SBATCH --mem=40G   # maximum memory per node
#SBATCH --job-name="build_emulator"
#SBATCH --output="build_emulator_%j.out" # job standard output file (%j replaced by job id)

# setup env
source ../load_env_intel
export PYTHONPATH=${PYTHONPATH}:/mnt/home/carls502/ANL/ND-tree_tabular_data_emulator/

# do computation
python3 build_SRO_emulator.py ${1}