#!/bin/bash
#SBATCH --array=0-10
#SBATCH --time=01:00:00
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_003046
##SBATCH --export=ALL
#SBATCH -J vds
#SBATCH -o .%j.out

PREFIX=/gpfs/exfel/exp/SPB/202202/p003046/

source /etc/profile.d/modules.sh
source ${PREFIX}usr/Shared/alfredo/xfel3046/source_this_at_euxfel
module load exfel exfel_anaconda3

run=`printf %.4d "${SLURM_ARRAY_TASK_ID}"`
#echo ${SLURM_ARRAY_TASK_ID}
python ${PREFIX}/usr/Shared/alfredo/xfel3046/offline/slurm/azimuthal_integration.py ${SLURM_ARRAY_TASK_ID} $1
