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
import pyFAI

PREFIX = '/gpfs/exfel/exp/SQS/202202/p003046/'
ADU_PER_PHOTON = 5

class RadialAverage():
    def __init__(self, run, dark_run, nproc=0, chunk_size=32, n_images=0, litpixel_threshold=0):
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

    def load_geom(self):
        geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(PREFIX+'/scratch/geom/agipd_september_2022_v03.geom')
        pixel_size = 200e-6
        detector_distance = 0.334
        wavelength = 0.15498e-9 #8 keV
        ai = azimuthalIntegrator.AzimuthalIntegrator(detector=geom.to_pyfai_detector(),
                                                     wavelength=wavelength,
                                                     dist = detector_distance)
        self.solid_angle = ai.solidAngleArray(geom.expected_data_shape)
        self.polarization = ai.polarization(shape=geom.expected_data.shape,factor=1)
        self.q = ai.qArray()
            
    def run_module(self, module):
        sys.stdout.write('Calculating radial average in module %d for %d frames\n'%(module, self.dshape[0]))
        sys.stdout.flush()
        # Radial average for each module and each frame. Store both the sum and the number of pixels per radius
        radialavg = mp.Array(ctypes.c_float, int(self.nhits*self.max_rad*3))
        jobs = []
        for c in range(self.nproc):
            p = mp.Process(target=self._part_worker, args=(c, module, radialavg))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        self.radialavg = np.frombuffer(radialavg.get_obj(), dtype=np.float32)
        return self.radialavg

    def _parse_darks(self, module):
        with h5py.File(PREFIX + '/scratch/dark/r%.4d_dark.h5'%self.dark_run, 'r') as f:
            # Get dark for central 4 ASICs of module
            dark = f['data/mean'][module,:,:,:]
            cells = f['data/cellId'][:]
            sigma = f['data/sigma'][module]
        return dark, cells, sigma            

    def _part_worker(self, p, m, radialavg):
        np_radialavg = np.frombuffer(radialavg.get_obj(), dtype=np.float32).reshape((self.nhits,self.max_rad, 3))

        nframes = self.nhits
        my_start = (nframes // self.nproc) * p
        my_end = min((nframes // self.nproc) * (p+1), nframes)

        dark, cells, sigma = self._parse_darks(m)
        mask = ~(sigma.mean(0) < 0.5) | (sigma.mean(0) > 1.5)

        stime = time.time()
        f_vds = h5py.File(self.vds_file, 'r')
        idx = my_start
        intrad = self.intrad[m]
        while(idx < my_end):
            data = f_vds[self.dset_name][self.hits[idx], m, 0, :, :]
            # Assume all cells are the same
            cids = f_vds['entry_1/cellId'][self.hits[idx], 0]
            data = data - dark[cids]

            data[data < 4.] = 0            
            data /= ADU_PER_PHOTON
            radcount = np.zeros(self.intrad.max()+1) 
            radavg = np.zeros_like(radcount)
            radvar = np.zeros_like(radcount)
            mymask = mask & ~np.isnan(data)
            np.add.at(radcount, intrad[mymask], 1)
            np.add.at(radavg, intrad[mymask], data[mymask])
            with np.errstate(divide='ignore', invalid='ignore'):
                np.add.at(radvar, intrad[mymask], (data[mymask] - (radavg/radcount)[intrad[mymask]])**2)
            np_radialavg[idx,:,0] = radavg
            np_radialavg[idx,:,1] = radcount
            np_radialavg[idx,:,2] = radvar

            idx += 1
            etime = time.time()
            if p == 0:
                sys.stdout.write('\r%.4d/%.4d: %.2f Hz' % (idx+1, my_end-my_start, (idx+1-my_start)/(etime-stime)*self.nproc))
                sys.stdout.flush()
        if p == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

def copy_ids(fname, hits, fptr):
    print('Copying IDs from VDS and litpixel files')
    sys.stdout.flush()

    f_vds = h5py.File(fname, 'r')
    if 'entry_1/trainId' in fptr: del fptr['entry_1/trainId']
    if 'entry_1/cellId' in fptr: del fptr['entry_1/cellId']
    if 'entry_1/pulseId' in fptr: del fptr['entry_1/pulseId']

    fptr['entry_1/trainId'] = f_vds['entry_1/trainId'][:]
    fptr['entry_1/cellId'] = f_vds['entry_1/cellId'][:]
    fptr['entry_1/pulseId'] = f_vds['entry_1/pulseId'][:]

    if 'entry_1/hit_indices' in fptr: del fptr['entry_1/hit_indices']    
    fptr['entry_1/hit_indices'] = hits
    f_vds.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Radial average calculator')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('dark_run', type=int, help='Dark run number')
    parser.add_argument('-n', '--nproc', 
                        help='Number of processes to use',
                        type=int, default=0)
    parser.add_argument('-m', '--module', nargs='+', 
                        help='Run on only these modules',
                        type=int, default=[0,7,8,15])
    parser.add_argument('-i', '--images', help='Run on only the first i images',
                        type=int, default=0)
    parser.add_argument('-l', '--litpixels', help='Run only on images with more than the given number of litpixels',
                        type=int, default=0)    
    parser.add_argument('-o', '--out_folder', 
                        help='Path of output folder (default=%s/scratch/data/)'%PREFIX,
                        default=PREFIX+'scratch/data/')
    args = parser.parse_args()


    l = RadialAverage(args.run, args.dark_run, nproc=args.nproc, n_images=args.images, litpixel_threshold=args.litpixels)
    print('Running on the following modules:', args.module)

    radialavg = np.array([l.run_module(module) for module in args.module]).reshape((len(args.module),l.nhits,l.max_rad, 3))
    # Sum across all modules. This is wrong for the variance, but close enough
    radialavg = np.sum(radialavg,axis=0)
    print(radialavg.shape)
    
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

        if 'entry_1/modules' in outf: del outf['entry_1/modules']
        outf['entry_1/modules'] = args.module
        copy_ids(l.vds_file, l.hits, outf)
    print('DONE')
                
if __name__ == '__main__':
    main()
