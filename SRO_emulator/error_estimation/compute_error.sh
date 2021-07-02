#!/bin/bash
#SBATCH --time=06:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 1 processor core(s) per node 
#SBATCH --mem=30G   # maximum memory per node
#SBATCH --job-name="error_computation"
#SBATCH --output="error_computation_%j.out" # job standard output file (%j replaced by job id)

source ../../../load_env

conda activate nd_emulator

export PYTHONPATH=${PYTHONPATH}:/mnt/home/carls502/ANL/ND-tree_tabular_data_emulator/

python3 compute_errors.py $1


