#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1.2/bin/python
#SBATCH -p upex
#SBATCH -t 4:00:00
#SBATCH -J LPDfai
#SBATCH --mem=754000
#>>>Please change XXXX to run number<<<
#SBATCH --output=/gpfs/exfel/exp/FXE/202201/p002787/scratch/slurmout/LPDfai_run0042_id_%j_user_%u.out

"""
Created on Aug 23 21:33:23 2017

@author: khakhulin
@contributors: kluyver, uemura, danilevski, checchia
"""

import h5py
import sys, os, datetime
import numpy as np
import time, math
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
from extra_data import open_run, by_id
from extra_data.components import LPD1M
from extra_geom.lpd_old import LPDGeometry

# Tell pyFAI not to use OpenCL, prevents errors loading it on machines w/out GPU
import pyFAI
pyFAI.use_opencl = False
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

# To read data from user experiment proposal put respective cycle, experiment ID and paths, for exmample:
#ExpCycle = '202122'
#ExpNum = '2958'
#ExpScratchDir = '/gpfs/exfel/exp/FXE/'+ExpCycle+'/p00'+ExpNum+'/scratch/'

## Settings
#ExpCycle = '202150'
#ExpNum = '700002'
#ExpScratchDir = '/gpfs/exfel/exp/XMPL/'+ExpCycle+'/p'+ExpNum+'/scratch/'

#ExpCycle = '202150'
ExpCycle = '202201'
ExpNum = '002787'
#ExpScratchDir = '/gpfs/exfel/exp/XMPL/'+ExpCycle+'/p'+ExpNum+'/scratch/'
ExpScratchDir = '/gpfs/exfel/exp/FXE/'+ExpCycle+'/p'+ExpNum+'/scratch/'

darks_dir = ExpScratchDir+'Darks'
lpd_params_dir = ExpScratchDir+'LPD_params/'
mask_file = lpd_params_dir+'Mask_99.npy'
lpd_geom_file = lpd_params_dir+'lpd_mar_18.h5'
out_dir = ExpScratchDir+'Reduced'

"""
[Parameters]
runNums : run number(s), array/int
trains_slice: Select only part of the run to process, by slicing trains.
              e.g. np.s_[:1000] -> first 1000 trains
DarkRuns: specify dark run numbers: [High, Med, Low]
scan_device: Scan types: 'delay' or 'wave plate'
all_trains: Take all the trains (within trains_slice) or not
factor: if all_trains is not True, this factor affects how many trains you take.
        The value should be between 1 and 10. Larger factor => More trains
"""

#### Parameters ####
#DarkRuns = [67, 68, 69] #High, Med, Low 
DarkRuns = [112, 113, 114] #High, Med, Low 
DisableMask = False # Put True when a new mask is needed

if len(sys.argv)<2: 
    runNums = [ 104, 105 ]
    print(f'Reducing data from runs predefined in code: {runNums} in [PROPOSAL, CYCLE] = [{ExpNum}, {ExpCycle}]')
elif sys.argv[1]=='-h' or sys.argv[1]=='--help':
    print(f'Usage: sbatch '+ sys.argv[0] + '[ RunNumber ]')
    exit()
elif len(sys.argv)<3:
    runNums = np.asarray([int(sys.argv[1])])
    trains_slice = np.s_[:]
    
    print(f'Reducing data from run {runNums} in [PROPOSAL, CYCLE] = [{ExpNum}, {ExpCycle}]')
else:
    runNums = np.asarray([int(sys.argv[1])])
    NumTrains = int(sys.argv[2])
    trains_slice = np.s_[:NumTrains]
    print(f'Reducing first {NumTrains} trains from run {runNums} in [PROPOSAL, CYCLE] = [{ExpNum}, {ExpCycle}]')

saving = True
scan_device1 = 'delay'
scan_device2 = 'wave-plate'
all_trains = True
factor = 4

npt = 512  # Number of points for azimuthal integration
dist = 250 * 1e-3  # Sample-detector distance (m)
center_X = 575  # Oct 2021
center_Y = 618

pixel_size = 0.5e-3
energy = 9.3  # Photon Energy [keV]
wavelength_lambda = 12.3984 / energy * 1e-10  # m

quadpos = [(0, -312), (26, -24), (-238, 3), (-264.5, -285.5)]  # MAR 2020

print(lpd_geom_file)

SM_unit = 256   # Pixels across a supermodule (same in both dimensions)
RAMlimit = 250  # [GB] of max RAM avalible for image data
Nworkers = 64

plotfinal = False  # Show some plots at the end

# q normalization range, currently not used
Qnorm_min = 0.5
Qnorm_max = 3
RadialRange = (0.01, 7.0)

# Directories

print(darks_dir)
print (f"Using Dark Runs numbers [High, Medium, Low] = [{DarkRuns[0]}, {DarkRuns[1]}, {DarkRuns[2]}]")

#device (scan)
device_src = {
    'delay': ('FXE_AUXT_LIC/DOOCS/PPODL','actualPosition.value'),
    'wave-plate': ('FXE_SMS_USR/MOTOR/UM13','actualPosition.value'),
}
#%%
def Seq_integr(chunk_no, run_chunk):
    ai = AzimuthalIntegrator(dist=dist,
                   poni1=center_Y*pixel_size,
                   poni2=center_X*pixel_size,
                   pixel1=pixel_size,
                   pixel2=pixel_size,
                   rot1=0,rot2=0,rot3=0,
                   wavelength=wavelength_lambda)  

    t0 = time.monotonic()
    lpd = LPD1M(run_chunk)

    data = lpd.get_array('image.data', unstack_pulses=False)
    t1 = time.monotonic()
    print(f"Loaded LPD data with shape {data.shape}, in {t1 - t0:.2f}s", flush=True)
    nframes = data.shape[1]


    LowGain = np.asarray(
        [73,73.5,67.8,66.2,73,74.4,67.8,68.6,67.0,73,66.3,65.4,65.4,67.8,65.4,67.9]
    ).reshape((16, 1, 1))

    # Make an empty frame to sum up images into
    TotalIm, _ = geom.position_all_modules(np.zeros((16, SM_unit, SM_unit)))
    TotalIm[np.isnan(TotalIm)] = 0

    # Arrays to hold integration results
    Q = np.zeros(npt, np.float64)
    Sq_sa0 = np.zeros((npt, nframes), dtype=np.float32)

    for i in range(nframes):
        raw_image = data[:, i].values
        pixels_med_gain = (raw_image >= 4096) & (raw_image < 8192)
        pixels_low_gain = raw_image >= 8192

        # Correct data (high, medium, and low gain in that order)
        DarkPulse = i % PulsesPerTrain

        # CorrAr_Offs_* aren't currently loaded, so we skip subtracting them to speed things up.
        #Corr_hg = ((raw_image - DarkGain100[:, DarkPulse]) - CorrAr_Offs_high) / CorrAr_Gain_high
        
#        print(f"Raw: {raw_image.shape} Dark: {DarkGain100.shape}")
        Corr_hg = (raw_image - DarkGain100[:, DarkPulse]) / CorrAr_Gain_high
        Curr_image = Corr_hg#.astype('int64', copy=True)

        #Corr_mg = ((raw_image - DarkGain10[:, DarkPulse]) - CorrAr_Offs_med) / CorrAr_Gain_med
        Corr_mg = (raw_image - DarkGain10[:, DarkPulse]) / CorrAr_Gain_med
        Curr_image[pixels_med_gain] = Corr_mg[pixels_med_gain]

        Corr_lg = (raw_image - DarkGain1[:, DarkPulse]) * LowGain  # MedGain*7.44
        Curr_image[pixels_low_gain] = Corr_lg[pixels_low_gain]

        # Masking - replace invalid values with 0
        Curr_image = Curr_image * CorrBP
        Curr_image[:, :, 127:129] = 0
        for ypix in range(32, 255, 32):
            Curr_image[:, ypix - 1:ypix + 1, :] = 0
        Curr_image[:, 0:2, :] = 0
        Curr_image[:,254:256, :] = 0
        Curr_image[:,:, 0:2] = 0
        Curr_image[:,:, 254:256] = 0
        Curr_image[1, 128:160, 128:256] = 0

        Curr_image[Curr_image < 0] = 0
        Curr_image[Curr_image > 1e6] = 0

        # Assemble modules into a single 2D image
        assembled, _ = geom.position_all_modules(Curr_image.astype(np.float64))

        # Convert NaN values to zero
        assembled[np.isnan(assembled)] = 0

        TotalIm += assembled * mask_comb_inv  # mask_comb_inv is 0 where invalid

        # PyFAI masks are 0 for valid, 1 for invalid
        Q, i_unc_sa0 = ai.integrate1d(assembled,
                                      npt, method="BBox",
                                      mask=(mask_comb | (assembled == 0)),
                                      radial_range=RadialRange,
                                      correctSolidAngle=True,
                                      polarization_factor=1,
                                      unit="q_A^-1")
        Sq_sa0[:, i] = i_unc_sa0

    t2 = time.monotonic()
    print(f'Chunk {chunk_no}/{Nchunks}: '
          f'Preparation and azimuthal integration of {nframes} frames took '
          f'{t2 - t1:.2f}s ({(t2 - t1) * 1000 / nframes:.1f} ms per frame)',
          flush=True)

    trainIDs = data.coords['train'].values
    pulseIDs = data.coords['pulse'].values
    cellIDs_allmods = lpd.get_array('image.cellId', unstack_pulses=False)
    cellIDs = cellIDs_allmods.max(dim='module').values

    # Result fields for 2D azimuthal integration ('cake') - not currently used
    qu = chi = 0
    Sdphi = np.zeros((1, 1, nframes), dtype=np.float32)

    return TotalIm, Q[:, None] , Sq_sa0, qu, chi, Sdphi, trainIDs, cellIDs, pulseIDs


def load_constants():
    global CorrAr_Gain_high, CorrAr_Gain_med
    global CorrAr_Offs_high, CorrAr_Offs_med
    global CorrBP

    global DarkGain1, DarkGain10, DarkGain100

    CorrAr_Gain_high = np.ones((16, SM_unit, SM_unit), dtype=float)
    CorrAr_Gain_med = np.ones((16, SM_unit, SM_unit), dtype=float) * 0.115 
    CorrAr_Offs_high = np.zeros((16, SM_unit, SM_unit), dtype=float)
    CorrAr_Offs_med = np.zeros((16, SM_unit, SM_unit), dtype=float)
    CorrBP = np.zeros((16, SM_unit, SM_unit), dtype=bool)

    DarkGain1 = np.zeros((16, PulsesPerTrain, SM_unit, SM_unit), dtype=int)
    DarkGain10 = np.zeros((16, PulsesPerTrain, SM_unit, SM_unit), dtype=int)
    DarkGain100 = np.zeros((16, PulsesPerTrain, SM_unit, SM_unit), dtype=int)

    missing_darks = 0
    for i in range(16):
        try:
            dark100_file = h5py.File(
                f'{darks_dir}/DarksHighGain_p{ExpNum}_MeanCell_all_LPD{i:02}R{DarkRuns[0]:04}.h5', 'r'
            )
            dark10_file = h5py.File(
                f'{darks_dir}/DarksMedGain_p{ExpNum}_MeanCell_all_LPD{i:02}R{DarkRuns[1]:04}.h5', 'r'
            )
            dark1_file = h5py.File(
                f'{darks_dir}/DarksLowGain_p{ExpNum}_MeanCell_all_LPD{i:02}R{DarkRuns[2]:04}.h5', 'r'
            )
        except Exception:
            print(f"No darks found for module {i}")
            missing_darks += 1
        else:
            A100 = dark100_file['DarkImages']
            A10 = dark10_file['DarkImages']
            A1 = dark1_file['DarkImages']
            if A100.shape[0]==PulsesPerTrain:
                DarkGain100[i] = A100
                DarkGain10[i] = A10
                DarkGain1[i] = A1
            elif  A100.shape[0] == 510:
                DarkGain100[i] = np.take(A100,cellID_stack0[:PulsesPerTrain],axis=0)
                DarkGain10[i] = np.take(A10,cellID_stack0[:PulsesPerTrain],axis=0)
                DarkGain1[i] = np.take(A1,cellID_stack0[:PulsesPerTrain],axis=0)                   
            else:
                print(f"Wrong dimensions of dark image array for module {i}")
                exit(1)
            dark100_file.close()
            dark10_file.close()
            dark1_file.close()

        try:
            CorrAr_file = h5py.File(
                f'{lpd_params_dir}/HDF/CorrectionArrays_SM{i}', 'r'
            )
        except Exception:
            print(f"No correction arrays for module {i}")
        else:
            CorrAr_Gain_high[i] = CorrAr_file['/High_Gain_Arrays/Gain_Array']
            CorrAr_Gain_med[i] = CorrAr_file['/Medium_Gain_Arrays/Gain_Array']
            #CorrAr_Offs_high [i] = CorrAr_file['/High_Gain_Arrays/Offset_Array']
            #CorrAr_Offs_med [i] = CorrAr_file['/Medium_Gain_Arrays/Offset_Array']
            CorrAr_file.close()

        try:
            CorrBP_file = h5py.File(
                f'{lpd_params_dir}/1M_bad_pixel_masks/BadPixelBitMask_SM{i}.h5', 'r'
            )
        except Exception:
            print(f"No bad pixel info for module {i}")
        else:
            badpix_data = CorrBP_file['PixelMask'][8, :, :]
            # Convert to 1 for valid pixels - we'll multiply by the mask
            CorrBP[i] = (badpix_data == 0)
            CorrBP_file.close()

    if missing_darks == 16:
        print('Failed to read dark runs, please check. Quitting processing. \n')
        exit(1)

    # Zeros in these arrays cause numpy to spit out divide-by-zero warnings.
    # Replace with NaN - these pixels will be masked out in the images.
    CorrAr_Gain_high[CorrAr_Gain_high == 0] = np.nan
    CorrAr_Gain_med[CorrAr_Gain_med == 0] = np.nan


if __name__ == '__main__':
    
    stTime = time.time()

    with h5py.File(lpd_geom_file, 'r') as f:
        geom = LPDGeometry.from_h5_file_and_quad_positions(f, quadpos)

    # Mask file contains 0 for pixels to keep, 1 to exclude.
    # It describes a single assembled (2D) image
    mask_comb = np.load(mask_file).astype(bool)
    mask_comb_inv = np.invert(mask_comb) | DisableMask

    now = datetime.datetime.now()
    start_dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    for runNum in runNums:
        print (f"########## PROCESS START: [PROPOSAL, RunNumber]=[{ExpNum}, {runNum}] ########## ")
        run = open_run(ExpNum, runNum).select_trains(trains_slice)

        cellID_stack0 = run[
            'FXE_DET_LPD1M-1/DET/8CH0:xtdf', 'image.cellId'
        ].drop_empty_trains().select_trains(0).ndarray()
        cellID_stack0 = np.squeeze(cellID_stack0)
        PulsesPerTrain = cellID_stack0.shape[0]
        print(cellID_stack0)

        # 1 raw frame with 16 supermodules is 2 MiB
        MBytesPerTrain = PulsesPerTrain * 2
        # Pick our chunk size so that loading data uses at most 10% of RAMlimit
        # across all workers, leaving plenty of room for processing.
        TrainsPerChunk = RAMlimit * 1024 // 10 // Nworkers // MBytesPerTrain

        load_constants()  # Constants are added to the global namespace

        # Select what data to process
        sel_data = LPD1M(run).data
        tids_lpd = sel_data.train_ids
        tids_device1 = run.select(*device_src[scan_device1], require_all=True).train_ids
        delays = run[device_src[scan_device1]].ndarray()
        tids_device2 = run.select(*device_src[scan_device2], require_all=True).train_ids
        waveplate = run[device_src[scan_device2]].ndarray()
        

        if not all_trains:          
            _TrainIds = np.intersect1d(tids_lpd,tids_device1)
            udelays = list(np.unique(delays))
            uwaveplate = list(np.unique(waveplate))
            step = len(delays) // (len(udelays)*(factor*10))
            if step < 0:
                print("!INFO! train step is smaller than 1. The step is set to 100")
                step = 100
            print(f"The number of trains: {len(delays)}, Unique delays(positions): {len(udelays)}, Step: {step}")
            sel_data = sel_data.select_trains(by_id[_TrainIds[::step]])           

        # Split selected data into chunks
        chunks = list(sel_data.split_trains(parts=Nworkers, trains_per_part=TrainsPerChunk))
        Nchunks = len(chunks)

        # Processing
        print(f"Processing {len(sel_data.train_ids)} trains "
              f"in {Nchunks} chunks of <={TrainsPerChunk} trains", flush=True)
        with Pool(processes=Nworkers) as pool:
            results = pool.starmap(Seq_integr, enumerate(chunks))

        # Gather results
        Im = np.dstack([r[0] for r in results])
        q = results[0][1]
        Sq = np.concatenate([r[2] for r in results], axis=1)
        qu = results[0][3]
        chi = results[0][4]
        Sq_2D = np.concatenate([r[5] for r in results], axis=2)
        trainIDs = np.concatenate([r[6] for r in results])
        cellIDs = np.concatenate([r[7] for r in results])
        pulseIDs = np.concatenate([r[8] for r in results])

        TotTime = ((time.time()-stTime))
    
        print('Whole run', str(runNum), 'took', str(math.ceil(TotTime)),
              ' s or about',str(math.ceil(TotTime/np.size(trainIDs)*1000)), ' ms per image',
              flush=True)

        if (saving == 1) :
            t0_save = time.monotonic()
            os.makedirs(out_dir, exist_ok=True)
            full = all_trains and (trains_slice == np.s_[:])
            full_or_part = 'full' if full else 'part'
            fileStr = f'{out_dir}/ChiRunRAW_Corr_{full_or_part}_{runNum}_testing.h5'

            h5f = h5py.File(fileStr,'w')
            h5f.create_dataset('q', data=q)
            h5f.create_dataset('qu', data=qu)
            h5f.create_dataset('chi', data=chi)
            h5f.create_dataset('Sq_sa0', data=Sq)
            h5f.create_dataset('trainID', data=trainIDs[:, None])
            h5f.create_dataset('cellID', data=cellIDs[:, None])
            h5f.create_dataset('pulseID', data=pulseIDs[:, None])
            mean_image = np.sum(Im,axis=2)/np.size(trainIDs)
            h5f.create_dataset('MeanImage', data=mean_image)
            h5f.create_dataset('Sq2D', data=Sq_2D)
            h5f.create_dataset('TidsRunLPD', data=tids_lpd)
            h5f.create_dataset('TidsRunDelay', data=tids_device1)
            h5f.create_dataset('TidsRunWaveplate', data=tids_device2) 
            h5f.create_dataset('DelayRun', data=delays) 
            h5f.create_dataset('WaveplateRun', data=waveplate)
#            h5f.create_dataset('MeanImageSq2D', data=np.mean(Sq_2D,axis=0))
            h5f.close() 
            print('File ' + fileStr + ' saved')

            im = Image.fromarray(mean_image)
            im.save(f'{out_dir}/SumImage_Corr{runNum}.tiff')

            print(f"Saving output files took {time.monotonic() - t0_save:.2f} s")

        if (plotfinal) :
            plt.figure(figsize=(14,14))
        #    plt.imshow(TotalIm) 
            plt.imshow(np.sum(Im,axis=2)/np.size(trainIDs),vmin=0, vmax=2.0e4)
            plt.colorbar()

            plt.figure(figsize=(14,14))
        #    plt.imshow(TotalIm) 
            plt.imshow(np.sum(Sq_2D,axis=2)/np.size(trainIDs),vmin=-0e2, vmax=3.0e3)
            plt.ylabel("Azimuthal angle, phi [deg]")
            plt.xlabel("Momentum transfer, q [1/A]")
    #        plt.imshow(np.sum(Im[500:800,50:250,:],axis=2)/np.size(trainIDs),vmin=7e2, vmax=2e3)
    #        plt.figure(figsize=(14,14))
    #        plt.imshow(np.sum(Im[500:800,450:750,:],axis=2)/np.size(trainIDs),vmin=0e2, vmax=2e3)
    #        plt.imshow(np.sum(Im,axis=2)/np.size(trainIDs))
    #        plt.title('Average image Run # ' + str(runNum))
            plt.colorbar()
            plt.figure(figsize=(14,7))
        #    plt.plot(q,np.median(Sq,axis=1),label='median')
        #    plt.plot(q,np.mean(Sq,axis=1),label='mean')
            plt.plot(q,np.median(Sq,axis=1)/np.max(np.median(Sq,axis=1)),label='median')
            plt.plot(q,np.mean(Sq,axis=1)/np.max(np.mean(Sq,axis=1)),label='mean')
            plt.legend()
            plt.grid()
            plt.xlim((0.05,6.0))
            plt.xlabel("Momentum transfer, q [1/A]")
            plt.ylabel("S(q) [ADU]")
    #        plt.title('Azimuthally integrated Run # ' + str(runNum) + '. Mean and median S(q)')
            #fig.set_size_inches((16,9))
    #        plt.savefig('/gpfs/exfel/exp/FXE/201701/p002026/scratch/ReducedScans/June2018/img_r'+str(runNum).zfill(4)+'.png')
            plt.show()
    now = datetime.datetime.now()
    end_dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    print ("############PROCESS: End###############")
    print ('start:' + start_dt_string)
    print ('end:' + end_dt_string)
