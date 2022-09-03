#!/usr/bin/env python

'''Calculate lit azimuthal average per pulse for a run using the VDS files'''

import h5py
import numpy as np
from extra_data import open_run
import os.path as op
import sys

PREFIX = '/gpfs/exfel/exp/SPB/202202/p003046/'

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Radial average calculator')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('-o', '--out_folder', 
                        help='Path of output folder (default=%s/scratch/data/)'%PREFIX,
                        default=PREFIX+'scratch/data/')
    args = parser.parse_args()


    run = open_run(proposal=3046, run=args.run)
    volt = run['SPB_RR_SYS/ADC/UTC1-2:channel_0.output']['data.rawDataVolt'].ndarray()
    train = run['SPB_RR_SYS/ADC/UTC1-2:channel_0.output']['data.trainId'].ndarray()
    laser_on = (np.max(volt,axis=1) > 0)
    vds_file = PREFIX+'scratch/vds/proc/r%.4d_proc.cxi' %args.run
    
    out_fname = args.out_folder + op.splitext(op.basename(vds_file))[0] + '_laser.h5'
    with h5py.File(out_fname, 'a') as outf:
        dset_name = 'entry_1/laser_on'
        if dset_name in outf: del outf[dset_name]
        outf[dset_name] = laser_on
        dset_name = 'entry_1/trainId'
        if dset_name in outf: del outf[dset_name]
        outf[dset_name] = train
    print('DONE')
                
if __name__ == '__main__':
    main()
