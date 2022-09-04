#!/bin/env python  
##SBATCH --array=0
##SBATCH --job-name=AzimuthalIntegration
##SBATCH --reservation=upex_003046
##SBATCH --partition=upex-beamtime
##SBATCH -o %j-%a-output.out
##SBATCH --export=ALL

#print('Step 1',flush=True)

import extra_geom
#print(extra_geom.__file__,flush=True)
from pyFAI import azimuthalIntegrator
import numpy as np
import h5py as h5
import sys
import os

#def radial_profile(data, center):
#    y,x = np.indices((data.shape)) # first determine radii of all pixels
#    r = np.sqrt((x-center[0])**2+(y-center[1])**2)
#    ind = np.argsort(r.flat) # get sorted indices
#    sr = r.flat[ind] # sorted radii
#    sim = data.flat[ind] # image values sorted by radii
#    ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
#    # determining distance between changes
#    deltar = ri[1:] - ri[:-1] # assume all radii represented
#    rind = np.where(deltar)[0] # location of changed radius
#    nr = rind[1:] - rind[:-1] # number in radius bin
#    csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
#    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
#    radialprofile = tbin/nr # the answer
#    return radialprofile

chunk = int(sys.argv[1]) #int(os.environ['SLURM_ARRAY_TASK_ID'])
run = int(sys.argv[2]) #92
print('Start', flush =True)

path = f'/gpfs/exfel/exp/SPB/202202/p003046/scratch/vds/proc/r{run:04}_proc.cxi'                        
output_path = '/home/alfredo/p003046/usr/Shared/alfredo/xfel3046/offline/slurm/radial_integration/'

os.makedirs(output_path + f'r00{run}', exist_ok = True)

chunk_tot = 10000       #int(os.environ['SLURM_ARRAY_TASK_COUNT'])

pixel_size = 200e-6
detector_distance = 0.217
wavelength = 0.15498e-9 #8 keV
npt = 128

#print('Step 2',flush=True)

#_____________________:loop for chunks of data 
with h5.File(path,'r') as vds_file:
    dataset = vds_file['entry_1']['data_1']['data']
    n_pulses = dataset.shape
    chunk_start = chunk*(n_pulses[0]//chunk_tot)
    chunk_end = (chunk+1)*(n_pulses[0]//chunk_tot)
    data = dataset[chunk_start:chunk_end,:,:,:]
    #print(chunk_start)
    #print(chunk_end)


mask_file = '/home/alfredo/p003046/usr/Shared/alfredo/xfel3046/mask/newer_mask.h5'

with h5.File(mask_file,'r') as f:
    mask = np.asarray(f['combined_mask'])

geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('/home/alfredo/p003046/usr/Shared/alfredo/xfel3046/geometry/agipd_september_2022_v03.geom')

#print(data.shape)
#print(mask.shape)

i = np.zeros((chunk_end-chunk_start,npt))
Q = None
degrees = 360
i_2d = np.zeros((chunk_end-chunk_start,degrees,npt))
variance = np.zeros((chunk_end-chunk_start,npt))

for index in range(chunk_end-chunk_start):
    #print(chunk_start)
    #print(chunk_end)
    #print(i)
    #print('Step 3',flush=True)
    assemask, centremask = geom.position_modules_fast(mask)


    assem, centre = geom.position_modules_fast(data[index,:,:])

    ai = azimuthalIntegrator.AzimuthalIntegrator(dist = detector_distance,
                                   poni1 = centremask[0] * pixel_size,
                                   poni2 = centremask[1] * pixel_size,
                                   pixel1 = pixel_size,
                                   pixel2 = pixel_size,
                                   rot1 = 0, rot2=0, rot3=0,
                                   #N.B: assembled vs dissassmbled: detector = geom.to_pyfai_detector(),
                                   wavelength = wavelength)

    ai_2d = azimuthalIntegrator.AzimuthalIntegrator(dist = detector_distance,
                                   poni1 = centremask[0] * pixel_size,
                                   poni2 = centremask[1] * pixel_size,
                                   pixel1 = pixel_size,
                                   pixel2 = pixel_size,
                                   rot1 = 0, rot2=0, rot3=0,
                                   #N.B: assembled vs dissassmbled: detector = geom.to_pyfai_detector(),                
                                   wavelength = wavelength)

    #print(assem.shape)
    #print(assemask.shape)
    assem[assemask != 1] = np.nan

    #print(assem.shape,flush=True)
    #print(npt,flush=True)
    #print(assem,flush=True)
    
    #print('Step 3.1',flush=True)
    
    #center = assem.shape[0]//2, assem.shape[1]//2
    #print(assemask.shape)
    #print(assemask.shape[0]-centre)
    #print(assemask.shape[1]-centre)
    #print(assem.shape)
    #i[index] = radial_profile(assem, centre)

    #print('Test', flush = True)
    
    Q, i[index] = ai.integrate1d(assem,
                                 npt,
                                 method = "BBox",
                                 #mask = (assemask | (assem[10] = 0)),
                                 radial_range = (0.05, 2.5),
                                 correctSolidAngle = True,
                                 polarization_factor = 1,
                                 unit = "q_A^-1")
    #print(variance.shape, flush=True)
    i_2d[index,:,:], _, _ = ai_2d.integrate2d(assem,
                                npt,
                                method = "BBox",
                                correctSolidAngle = True,
                                polarization_factor = 1,
                                unit = "q_A^-1")
    #print(variance.shape, flush=True)
    variance[index,:] = (i_2d[index,:,:].std(axis=0))**2

    #print('Step 3.2',flush=True)
    #print(i.shape,flush=True)
    #print(i,flush=True)

with h5.File(output_path + f'r{run:04}/r{run:04}_{chunk}.azimuth', 'w') as integrated:
    integrated.create_dataset('radial_profile', data = i[:])
    integrated.create_dataset('variance', data = variance[:])
    print('End', flush =True)
    #print('Step 4',flush=True)

 
