#!/usr/bin/env python

'''Calculate lit azimuthal average per pulse for a run using the VDS files'''

import sys
import os.path as op
import time
import glob
import multiprocessing as mp
import ctypes
import subprocess

import h5py
import numpy as np
import extra_geom
import pyFAI.azimuthalIntegrator

PREFIX = '/gpfs/exfel/exp/SPB/202202/p003046/'
ADU_PER_PHOTON = 5

class RadialAverage():
    def __init__(self, run, nproc=0, chunk_size=32, n_images=0):
        vds_file = PREFIX+'scratch/vds/proc/r%.4d_proc.cxi' %run
        print('Calculating radial average pixels from', vds_file)

        self.vds_file = vds_file
        self.chunk_size = chunk_size # Needs to be multiple of 32 for raw data
        if self.chunk_size % 32 != 0:
            print('WARNING: Performance is best with a multiple of 32 chunk_size')
        if nproc == 0:
            self.nproc = int(subprocess.check_output('nproc').decode().strip())
        else:
            self.nproc = nproc
        print('Using %d processes' % self.nproc)

        with h5py.File(vds_file, 'r') as f:
            self.dset_name = 'entry_1/instrument_1/detector_1/data'
            self.dshape = f[self.dset_name].shape        

            self.azi_avg_fname = PREFIX+'scratch/events/r%.4d_azimuth_avg.h5'%run
            
        self.load_geom()
        if(n_images != 0):
            self.n_images = min(n_images,self.dshape[0])
        else:
            self.n_images = self.dshape[0]

    def load_geom(self):
        geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(PREFIX+'/scratch/geom/agipd_september_2022_v03.geom')
        pixel_size = 200e-6
        detector_distance = 0.217
        wavelength = 0.15498e-9 #8 keV
        assem, centre = geom.position_modules_fast(np.empty(geom.expected_data_shape))

        ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist = detector_distance,
                                                           poni1 = centre[0] * pixel_size,
                                                           poni2 = centre[1] * pixel_size,
                                                           pixel1 = pixel_size,
                                                           pixel2 = pixel_size,
                                                           rot1 = 0, rot2=0, rot3=0,
                                                           wavelength = wavelength)
        
        self.solid_angle = ai.solidAngleArray(shape=assem.shape)
        self.polarization = ai.polarization(shape=assem.shape,factor=1)
        self.q = ai.qArray(shape=assem.shape)
        self.npts = 256
        self.intrad = np.floor(self.q/(np.max(self.q)+1e-7)*256).astype(int)
        self.max_rad = self.npts
        self.geom = geom
        mask_file = PREFIX+'/scratch/geom/newer_mask.h5'
        with h5py.File(mask_file,'r') as f:
            self.good_pixels = (np.asarray(f['combined_mask']))
            good_pixels, centre = geom.position_modules_fast(self.good_pixels)
            self.good_pixels = good_pixels
        self.ai = ai
            
    def run(self):
        sys.stdout.write('Calculating radial average %d frames\n'%(self.n_images))
        sys.stdout.flush()
        # Radial average for each frame. Store the sum and the number of pixels and variance per radius
        radialavg = mp.Array(ctypes.c_float, int(self.n_images*self.max_rad*3))
        jobs = []
        for c in range(self.nproc):
            p = mp.Process(target=self._part_worker, args=(c, radialavg))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        self.radialavg = np.frombuffer(radialavg.get_obj(), dtype=np.float32)
        return self.radialavg

    def _part_worker(self, p, radialavg):
        np_radialavg = np.frombuffer(radialavg.get_obj(), dtype=np.float32).reshape((self.n_images,self.max_rad, 3))

        nframes = self.n_images
        my_start = (nframes // self.nproc) * p
        my_end = min((nframes // self.nproc) * (p+1), nframes)

        stime = time.time()
        f_vds = h5py.File(self.vds_file, 'r')
        idx = my_start
        intrad = self.intrad
        while(idx < my_end):
            data = f_vds[self.dset_name][idx]
            assem, centre = self.geom.position_modules_fast(data)
            data = assem
            if 0:
                Q, i = self.ai.integrate1d(data,
                                           self.npts,
                                           method = "BBox",
                                           mask = self.assemask,
                                           radial_range = (0.1, 2.4),
                                           correctSolidAngle = True,
                                           polarization_factor = 1,
                                           unit = "q_A^-1")

            radcount = np.zeros(self.max_rad)
            radavg = np.zeros_like(radcount)
            radvar = np.zeros_like(radcount)
            mymask = ((self.good_pixels & ~np.isnan(data)) == 1)
            radcount = np.bincount(intrad[mymask])
            #np.add.at(radcount, intrad[mymask], 1)
            #np.add.at(radavg, intrad[mymask], data[mymask])
            radavg = np.bincount(intrad[mymask], weights=data[mymask])
            with np.errstate(divide='ignore', invalid='ignore'):
                #np.add.at(radvar, intrad[mymask], (data[mymask] - (radavg/radcount)[intrad[mymask]])**2)
                radvar = np.bincount(intrad[mymask], weights=(data[mymask] - (radavg/radcount)[intrad[mymask]])**2)
            np_radialavg[idx,:len(radavg),0] = radavg
            np_radialavg[idx,:len(radcount),1] = radcount
            np_radialavg[idx,:len(radvar),2] = radvar

            idx += 1
            etime = time.time()
            if p == 0:
                sys.stdout.write('\r%.4d/%.4d: %.2f Hz' % (idx+1, my_end-my_start, (idx+1-my_start)/(etime-stime)*self.nproc))
                sys.stdout.flush()
        if p == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

def copy_ids(fname, fptr):
    print('Copying IDs from VDS file')
    sys.stdout.flush()

    f_vds = h5py.File(fname, 'r')
    if 'entry_1/trainId' in fptr: del fptr['entry_1/trainId']
    if 'entry_1/cellId' in fptr: del fptr['entry_1/cellId']
    if 'entry_1/pulseId' in fptr: del fptr['entry_1/pulseId']

    fptr['entry_1/trainId'] = f_vds['entry_1/trainId'][:]
    fptr['entry_1/cellId'] = f_vds['entry_1/cellId'][:]
    fptr['entry_1/pulseId'] = f_vds['entry_1/pulseId'][:]

    f_vds.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Radial average calculator')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('-n', '--nproc', 
                        help='Number of processes to use',
                        type=int, default=0)
    parser.add_argument('-i', '--images', help='Run on only the first i images',
                        type=int, default=0)
    parser.add_argument('-o', '--out_folder', 
                        help='Path of output folder (default=%s/scratch/data/)'%PREFIX,
                        default=PREFIX+'scratch/data/')
    args = parser.parse_args()


    l = RadialAverage(args.run, nproc=args.nproc, n_images=args.images)

    radialavg = l.run().reshape(l.n_images,l.max_rad,3)
    # Sum across all modules. This is wrong for the variance, but close enough
    
    out_fname = args.out_folder + op.splitext(op.basename(l.vds_file))[0] + '_radavg.h5'
    with h5py.File(out_fname, 'a') as outf:
        dset_name = 'entry_1/radialsum'
        if dset_name in outf: del outf[dset_name]
        outf[dset_name] = radialavg[:,:,0]
        dset_name = 'entry_1/radialcount'
        if dset_name in outf: del outf[dset_name]
        outf[dset_name] = radialavg[:,:,1]
        dset_name = 'entry_1/radialavg'
        if dset_name in outf: del outf[dset_name]
        with np.errstate(divide='ignore', invalid='ignore'):
            outf[dset_name] = radialavg[:,:,0]/radialavg[:,:,1]
        dset_name = 'entry_1/radialvar'
        if dset_name in outf: del outf[dset_name]
        with np.errstate(divide='ignore', invalid='ignore'):
            outf[dset_name] = radialavg[:,:,2]/radialavg[:,:,1]

        dset_name = 'entry_1/q'
        if dset_name in outf: del outf[dset_name]
        outf[dset_name] = np.arange(l.max_rad)/l.max_rad*np.max(l.q)
        copy_ids(l.vds_file, outf)
    print('DONE')
                
if __name__ == '__main__':
    main()
