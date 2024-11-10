#!/bin/bash -l
#SBATCH --job-name=NC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 16G
#SBATCH --time=2-0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
##SBATCH --account=hpc-topo-sensedynamics

module purge
module load gcc mvapich2 py-torch
source ~/venv-gcc/bin/activate
module list
#srun python3 Cases_test.py ${1} ${2} ${3} ${4} ${5} > log_${5}_${1}_${2}and${3}_${4}
srun ${1} > ${2}
deactivate
