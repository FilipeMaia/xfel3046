#!/bin/bash

#SBATCH --array=91
#SBATCH --time=04:00:00
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_003046
#SBATCH --export=ALL
#SBATCH -J laseron
#SBATCH -o .%j.out
#SBATCH -e .%j.out

# Change the runs to process using the --array option on line 3

PREFIX=/gpfs/exfel/exp/SPB/202202/p003046

source /etc/profile.d/modules.sh
source ${PREFIX}/scratch/filipe/xfel3046/source_this_at_euxfel


python ../laser_on.py ${SLURM_ARRAY_TASK_ID} 

