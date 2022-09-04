#!/bin/bash

##SBATCH --array=0004-0005
#SBATCH --time=01:00:00
#SBATCH --partition=upex
##SBATCH --partition=upex-beamtime
##SBATCH --reservation=upex_003046
#SBATCH --export=ALL
#SBATCH -J vds
#SBATCH -o .%j.out
#SBATCH -e .%j.out

# Change the runs to process using the --array option on line 3

PREFIX=/gpfs/exfel/exp/SPB/202202/p003046/

source /etc/profile.d/modules.sh
source ${PREFIX}/usr/Shared/alfredo/xfel3046/source_this_at_euxfel

run=`printf %.4d "${SLURM_ARRAY_TASK_ID}"`
extra-data-make-virtual-cxi ${PREFIX}/proc/r${run} -o ${PREFIX}/scratch/vds/proc/r${run}_proc.cxi
#extra-data-make-virtual-cxi ${PREFIX}/raw/r${run} -o ${PREFIX}/scratch/vds/r${run}_raw.cxi --exc-suspect-trains
