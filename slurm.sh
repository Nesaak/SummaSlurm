#!/bin/bash

#SBATCH --job-name=Summa_test_1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=49
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=Summa_test_1.log

cd $SLURM_SUBMIT_DIR
cat $SLURM_JOB_NODELIST

module load openmpi/5.0.1

make clean
make
mpirun -np 49 ./summa
