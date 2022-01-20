#!/bin/bash
# 02614 - High-Performance Computing, January 2022
# 
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J mm_batch_gpu
#BSUB -o mm_batch_gpu%J.out
#BSUB -e mm_batch_gpu%J.err
#BSUB -q hpcintrogpu
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -n 16 -R "span[hosts=1] affinity[socket(1)]"

# Load the cuda module
module load cuda/11.5.1
module load gcc/10.3.0-binutils-2.36.1

# define the driver name to use
EXECUTABLE=matmult_f.nvcc

# define the permutation type in PERM
#
PERM="lib"

# enable(1)/disable(0) result checking
export MATMULT_COMPARE=0            # - control result comparison (def: 1)
# export MATMULT_RESULTS={[0]|1}    # - print result matrices (in Matlab format, def: 0)
# export MFLOPS_MIN_T=[3.0]         # - the minimum run-time (def: 3.0 s)
# export MFLOPS_MAX_IT=0            # - max. no of iterations; set if you want to do profiling.
export MKL_NUM_THREADS=16           # - For running on one core on CPU

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=16

lscpu
cd ..
echo -e "\n###DATASTART"
./$EXECUTABLE $PERM 64 64 64
./$EXECUTABLE $PERM 128 128 128
./$EXECUTABLE $PERM 256 256 256
./$EXECUTABLE $PERM 512 512 512
./$EXECUTABLE $PERM 1024 1024 1024
./$EXECUTABLE $PERM 2048 2048 2048
./$EXECUTABLE $PERM 4096 4096 4096
./$EXECUTABLE $PERM 8192 8192 8192
./$EXECUTABLE $PERM 16384 16384 16384