"""
@author: Asmus Ougaard Dohn
"""

import h5py
import numpy as np
import scipy 
import matplotlib.pyplot as plt
from extra_data import open_run
from tqdm.auto import tqdm
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

import multiprocessing as mp 
cpus = int(mp.cpu_count() / 2)

def make_colors(c, colmap='viridis'):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colmap)
    colors = [cmap(1. * i / c) for i in range(c)]
    return colors


class Diff:
    def __init__(self, run, dt_offset=0, mode='Pulse_On/Off', qnorm=[1, 4],
                 kick_trains=False, threshold=0.01, lbl='', do_plot=True, 
                 offset_angle=110,parallel=True,
                 pf_height=20, pf_threshold=None, pf_distance=220, 
                 path='/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/'):
        self.run = run  # run number  
        self.dt_offset = dt_offset  # time offset
        self.mode = mode  # Laser On/Off mode
        self.qnorm = qnorm  # Normalization range    
        self.path = path   # path of h5 files
        self.prefix = 'ChiRunRAW_Corr_full_' 
        self.suffix = '_testing.h5'  # so full fn: prefix+run+suffix  
        self.slcs = 17  # number of azimuthal slices 
        self.lbl = lbl  # extra outname
        self.do_plot = do_plot
        self.offset_angle = offset_angle
        self.proposal_number = 2787  # change for each experiment
        self.parallel = parallel

        # Peakfinder parameters (for matching digitizer output to pulses)
        self.pf_height = pf_height
        self.pf_threshold = pf_threshold
        self.pf_distance = pf_distance

        # kick_trains: Switches pulse/train mode: 
        # False: ALL trains have LaserOn (101010...101010...)
        # 0: The FIRST train has LaserOn, the second has not:
        #        101010...000000...101010...000000
        # 1: The SECOND train has LaserOn:
        #        000000...101010...000000...101010
        self.kt = kick_trains

        self.threshold = threshold

        self.q = None  # q vector
        self.n_pulses = None # no. of pulses per train 
        self.N_trains = None # no. of trains
        self.tid = None  # Train ID
        self.delays = None  # Time delays
        self.dt = None  # unique time delays
        self._data = None


    def diff(self, filtering=(0.1, 0.2)):
        ''' Run the calculation 
            filtering, (min, max) or None'''

        if self._data is None:
            self.load_h5()
        
        self.intensity_filter(filtering)
        self.make_diff()
        self.save()

    def load_h5(self):
        ''' Populate obj with data from h5 ''' 
        fname = self.prefix + str(self.run) + self.suffix
        data = h5py.File(self.path + fname,'r')

        q  = data['q'][:, 0]
        q_mask = (q > self.qnorm[0]) & (q < self.qnorm[1])
        
        pulses =  np.unique(data['pulseID'][:])
        n_pulses = pulses.size
        
        N_trains = int(data['pulseID'][:].size / n_pulses)
        
        q = data['q'][:,0]
        S = np.reshape(data['Sq_sa0'],[q.size, N_trains, n_pulses])

        TID = np.reshape(data['trainID'][:,0],[N_trains, n_pulses])
        delays = data['DelayRun'][:] - self.dt_offset
        dt = np.unique(delays)

        self._data = data
        self.q = q
        self.n_pulses = n_pulses
        self.N_trains = N_trains
        self.TID = TID 
        self.delays = delays
        self.dt = dt
        self.S = S  # scattering array of shape (q, trains, pulses)
        self.S_mask = np.ones(S.shape, bool)
        self.q_mask = q_mask
        self.filtering = None
        self.total = 0
    
    def simple_intensity_filter(self, imin, imax):
        S = np.copy(self.S)
        tot_S = np.sum(S, axis=0)
        tot_S_flat = tot_S.reshape(-1)
        mask = (tot_S_flat > imin) & (tot_S_flat < imax)
        mask = np.reshape(mask, tot_S.shape)
        self.S_mask *= mask 
        if self.do_plot:
            self.plot_hist(tot_S_flat[mask.reshape(-1)])

    def median_filter(self, filtering):
        S = self.S
        tot_S = np.sum(S, axis=0)
        tot_S_flat = np.reshape(tot_S, tot_S.size)
        m = np.nanmedian(tot_S_flat)
        lb = m - m * filtering[0]
        ub = m + m * filtering[1]
        mask = (tot_S_flat > lb) & (tot_S_flat < ub)
        mask = np.reshape(mask, tot_S.shape)
        S[:, ~mask] = np.nan 
        S =  np.reshape(S, [self.q.size, self.N_trains, self.n_pulses])
        self.total = -np.sum(mask)
        if self.do_plot:
            self.plot_hist(tot_S_flat, m, lb, ub)

        self.filtering = filtering
        self.S = S
        return m, lb, ub 

    def make_clean_ds(self):
        S = np.copy(self.S)
        ds = np.empty((self.dt.shape[0], self.q.shape[0]))
        unique_trains = np.unique(self.TID)
        diffs = np.empty([self.q.size, unique_trains.size])

        def md_et(S, T, TID, q_mask, filter_mask):
            ''' Meaning over entire train '''
            train_mask = T == TID
            S_on  = np.mean((S[:, train_mask][:, ::2][filter_mask[:, train_mask][:, ::2]]).reshape((len(q_mask), -1)), axis=1)
            S_off = np.mean((S[:, train_mask][:, 1::2][filter_mask[:, train_mask][:, 1::2]]).reshape((len(q_mask), -1)), axis=1)
            S_on_norm = S_on / np.mean(S_on[q_mask]) 
            S_off_norm = S_off / np.mean(S_off[q_mask]) 
            return S_on_norm - S_off_norm

        def md_nn(S, T, TID, q_mask, filter_mask):
            unique_tids =  np.unique(TID)
            train_mask = T == TID
            S_on  = (S[:, train_mask][:, ::2][filter_mask[:, train_mask][:, ::2]]).reshape((len(q_mask), -1))
            if np.sum(S_on) == 0:
                return np.zeros(len(q_mask)) * np.nan
            else:
                S_off = (S[:, train_mask][:, 1::2][filter_mask[:, train_mask][:, 1::2]]).reshape((len(q_mask), -1))
                S_on_norm = S_on / np.mean(S_on[q_mask][None, :], axis=1) 
                S_off_norm = S_off / np.mean(S_off[q_mask][None, :], axis=1) 
                this_diff = np.zeros(q_mask.shape)
                num_ons = S_on_norm.shape[1]
                num_offs = S_off_norm.shape[1]
                if (num_ons < 5) | (num_offs < 5):  # Too few good pulses in this train
                    return np.zeros(len(q_mask)) * np.nan
                else:
                    # loop over all ons and find closests offs
                    this_train_diffs = np.zeros((len(q_mask), S_on_norm.shape[1]))
                    for jj, this_on in enumerate(S_on_norm.T):
                        if (jj == 0) & (num_offs > 3):
                            this_off = np.mean(S_off_norm[:, :3], axis=1)
                        elif (jj == 0) & (num_offs < 3):
                            this_off = S_off_norm[:, 0]
                        elif (jj < num_offs) & (jj >=  1):
                            this_off = np.mean(S_off_norm[:, jj - 1:jj + 2], axis=1)
                        elif jj == num_offs:
                            this_off = np.mean(S_off_norm[:, -4:-1], axis=1)
                        elif jj >= num_offs:
                            this_off = np.mean(S_off_norm[:, -3:], axis=1) # last couple shots.. bad. 
                        else:
                            print(f'[{jj} COULD NOT FIND OFF LOL')
            
                        this_train_diffs[:, jj] = this_on - this_off

                    return np.median(this_train_diffs, axis=1)     
        # Serial style:
        #for ii, T in tqdm(enumerate(unique_trains)):
        #    this_diff = md_nn(S, T, self.TID, self.q_mask, self.S_mask)
        #    diffs[:, ii] = this_diff 
        # Parallel style
        D = Parallel(n_jobs=cpus)(delayed(md_nn)(S, T, 
                                              self.TID, 
                                              self.q_mask, 
                                              self.S_mask) for ii, T in enumerate(unique_trains))
        for ii, this_diff in enumerate(D):
            diffs[:, ii] = this_diff 
        self.clean_diffs = diffs 

        # average all curves in each time bin (nominal)
        bad_t = np.array([])
        self.bin_avg = S[0, :, :].size / self.dt.size
        for ii, t in tqdm(enumerate(self.dt)):
            mask = (t == self.delays)
            ds[ii, :] = np.nanmean(diffs[:, mask], axis=1)
            # If we do not have enough trains at this delay, throw it away.
            if self.bin_avg * self.threshold > np.sum(mask):
                bad_t = np.append(bad_t, int(np.where(self.dt==t)[0]))
                #print('dt = ' + str(t) +' is thrown away, number of pulses:' + str(np.sum(mask) * self.n_pulses))
            else:
                self.total += np.sum(mask * self.n_pulses)
        dt_corr = np.delete(self.dt, np.array(bad_t, dtype=int))
        ds_corr = np.delete(ds, np.array(bad_t, dtype=int), axis=0)
        if len(bad_t) != 0:
            print('Discarded delays: ' + ', '.join(str(b) for b in bad_t))

        self.dt_corr = dt_corr
        self.ds_corr = ds_corr
        if self.do_plot:
            self.plot_lines(self.q, ds_corr)
        
        self.ds = ds

    def make_diff(self):
        S = self.S
        ds = np.empty((self.dt.shape[0], self.q.shape[0]))
        self.bin_avg = S[0, :, :].size / self.dt.size
        bad_t = np.array([])
        diffs = np.empty([self.q.size, np.unique(self.TID).size])

        unique_tids = np.unique(self.TID)

        if self.kt is not False:
            unique_tids = unique_tids[self.kt::2] 

        for ii, T in tqdm(enumerate(unique_tids), total=len(unique_tids)):  # loop over trains
            mask = (T == self.TID) 
            # make on and off, based on EVERY SECOND PULSE: 101010
            S_on = np.nanmean(S[:, mask][:,::2], axis=1)
            S_off = np.nanmean(S[:, mask][:, 1::2], axis=1)
            # Normalize
            S_on_norm = S_on / np.nansum(S_on[self.q_mask], axis=0)
            S_off_norm = S_off / np.nansum(S_off[self.q_mask], axis=0)
            diff = S_on_norm - S_off_norm
            diffs[:, ii] = diff
        
        self.diffs = diffs 

        # average all curves in each time bin (nominal)
        for ii, t in tqdm(enumerate(self.dt)):
            mask = (t == self.delays)
            ds[ii, :] = np.nanmean(diffs[:, mask], axis=1)

            # If we do not have enough trains at this delay, throw it away.
            if self.bin_avg * self.threshold > np.sum(mask):
                bad_t = np.append(bad_t, int(np.where(self.dt==t)[0]))
                print('dt = ' + str(t) +' is thrown away, number of pulses:' + str(np.sum(mask) * self.n_pulses))
            else:
                self.total += np.sum(mask * self.n_pulses)
        dt_corr = np.delete(self.dt, np.array(bad_t, dtype=int))
        ds_corr = np.delete(ds, np.array(bad_t, dtype=int), axis=0)

        self.dt_corr = dt_corr
        self.ds_corr = ds_corr
        if self.do_plot:
            self.plot_lines(self.q, ds_corr)
        
        self.ds = ds

    def save(self):
        info = {}
        info['q_norm'] = self.qnorm
        info['kick_trains'] = self.kt
        info['dt_offset'] = self.dt_offset
        info['filtering'] = self.filtering
        info['threshold'] = self.threshold
        info['run'] = self.run
        
        savedict = {
            'Sazi': self.ds_corr.T,
            'S0': self.ds_corr.T,
            'S2': self.ds.T,
            'q' : self.q,
            'scanvar' : self.dt,
            'info' : info}

        savepath = self.path + f'../Diffs/Run{self.run:04d}_Reduced_azi{self.lbl}.mat'
        scipy.io.savemat(savepath, savedict)
        print('binned phi signal saved to file: ' + savepath +'\n' )
        print('Total number of curves used in diff signal: ' + str(self.total * 2))

    def plot_lines(self, x, y, xlbl='q  [$\AA^{-1}$]', ylbl='$\Delta S_{azi}$'):
        fig, ax = plt.subplots()
        col = make_colors(y.shape[0])
        for ii, t in enumerate(y):
            ax.plot(self.q, y[ii, :], color=col[ii])
            ax.set_xlabel(xlbl)
            ax.set_ylabel(ylbl)
            fig.tight_layout()
        return fig, ax

    def plot_hist(self, y, m=None, lb=None, ub=None, xlbl='$\Sigma S$', ylbl='#Curves'):
        fig, ax = plt.subplots()
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title('Intensity distribution \n run '+ str(self.run) + ' with filtering bounds')
        ax.hist(y, 100)
        if m != None:
            ax.plot([ub, ub],[0, y.size/30], 'k--')
            ax.plot([lb, lb],[0, y.size/30], 'k--')
            ax.plot([m, m],[0, y.size/30], 'r--')
        fig.tight_layout()
        plt.draw()

    def digistream_to_pulse(self, stream):
        ''' Uses peakfind to connect pulse peaks to pulse# 
            uses joblib to parallelise if self.parallel == True

            Assumes that if NO pulses are found in a train, the train is bad, 
            and will be masked out. 
        '''

        height = self.pf_height
        threshold = self.pf_threshold
        distance = self.pf_distance
        
        pulse_vals = np.ones([self.n_pulses, self.N_trains])
        if not self.parallel:
            for ii in tqdm(range(self.N_trains)):
                peaks = find_peaks(np.abs(stream[ii, :]), height=height, threshold=threshold, distance=distance, prominence=1)
                # if peakfinder doesn't find all pulses per train every time, the below lines break
                # this is good, because we need to be aware of this
                if len(peaks[0]) == 0:  
                    print(f'Train {self.trainID[p]} had no pulses!')
                    self.S_mask[:, ii, :] = False
                else: 
                    pulse_vals[:, ii] = stream[ii, peaks[0]]
        else:  
            def pf(stream, ii, height, threshold, distance):
                peaks = find_peaks(np.abs(stream[ii, :]), height=height, threshold=threshold, distance=distance, prominence=1)
                return peaks[0]
            peaks = Parallel(n_jobs=cpus)(delayed(pf)(stream, ii, height, threshold, distance) for ii in range(self.N_trains))
            for p, peak in enumerate(peaks):
                if peak.size == 0:
                    print(f'Train {self.trainID[p]} had no pulses!')
                    self.S_mask[:, p, :] = False
                else: 
                    pulse_vals[:, p] = stream[p, peak]

        self.pulse_vals = pulse_vals
        return pulse_vals

    def dbscan_filter(self, eps=0.15, min_samples=200, reload_pulses=True, 
                     diag_path=('FXE_RR_DAQ/ADC/1:network','digitizers.channel_2_B.raw.samples')):
        S = np.copy(self.S)
        # work on total scattering
        tot_S = np.sum(S, axis=0)
        tot_S_flat = tot_S.reshape(-1)

        if reload_pulses:
            run = open_run(self.proposal_number, self.run)
            pulses = run[diag_path].ndarray()
            pulse_vals = self.digistream_to_pulse(pulses) 
        else:
            pulse_vals = self.pulse_vals

        # Make correlation-array X with previous masks applied
        pv_flat = np.reshape(np.abs(pulse_vals.T), tot_S.size)
        X = np.vstack((pv_flat[(self.S_mask[0, :, :]).reshape(-1)], 
                    tot_S_flat[(self.S_mask[0, :, :]).reshape(-1)])).T

        # Preproces
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        # Density-find best shots 
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=cpus)
        yhat = model.fit_predict(X_scaled)

        # apply new filter 
        full_flat_mask = self.S_mask[0, :, :].reshape(-1)
        sub_flat_mask = yhat == 0
        new_flat_mask = np.zeros(full_flat_mask.shape, bool)
        new_flat_mask[full_flat_mask] = sub_flat_mask  # apply new filter to same subselection as before
        new_mask = new_flat_mask.reshape(self.S_mask[0, :, :].shape)  # over a single Q 
        for i in range(self.q.size):
            self.S_mask[i, :, :] = new_mask  # over all Qs, same value

        clusters = np.unique(yhat) #-1 is all, 0-... is clusters
        print(f'Clusters found: {len(clusters) - 1}')
        if self.do_plot:
            fig, ax = plt.subplots()
            for c, cluster in enumerate(clusters):
                row_ix = np.where(yhat == cluster)[0]
                ax.plot(X[row_ix, 0], X[row_ix, 1], f'C{c}.', alpha=.2)
            fig.tight_layout()

        return X, yhat 
        