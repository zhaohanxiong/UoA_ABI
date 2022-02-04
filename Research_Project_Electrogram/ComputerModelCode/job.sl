#!/bin/bash
#SBATCH -J MPI_JOB
#SBATCH -A nesi00385 # Project Account
#SBATCH --time=24:00:00 # Walltime
#SBATCH --ntasks=50 # number of tasks
#SBATCH --mem-per-cpu=4096 
module load OpenMPI/1.6.5-GCC-4.8.2
export I_MPI_FABRICS=shm:ofa
export I_MPI_DAPL_PROVIDER=ofa-v2-mlx4_0-1
export /nesi/project/uoa02715/jzha437/
echo "Nodes: ${SLURM_JOB_NODELIST}"
srun /nesi/project/uoa02715/jzha437/2DSim.out