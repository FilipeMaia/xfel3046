# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:21:12 2022

@author: mhaubro

inputs:
    run: run number
    
    dt_offset: time zero offset
    
    mode: 'Pulse_On/Off' or 'Train_On/Off', denoting the two exciation schemess used at the FXE
    
    qnorm: normalisation range for calculating difference scattering
    
    plotting: True or False, chose whether results should be plotted
    
    saving: Chose whether results should be saved
    
    filtering: either None or [lb, ub], intensity based filtering is then implemented filtering out all curves with total inensity below m - m*lb and above m + m*ub, where m is the mean of the intensity
    
    threshold: controls whether or not time bins with very few curves are thrown out, 0 keeps everything, to get rid of them 0.01 is an appropriate number. Numbers close to one and will remove everything
    
outputs:
    If saving = True, difference scattering array is saved in a .mat file
    If plotting = True, intensity distribution and resulting difference curves are plotted

"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
import scipy.io
from tqdm import tqdm
import glob
#from rescaleQ import rescaleQ 
from scipy.signal import find_peaks
from extra_data import open_run
from lmfit import Model

font = {        'size'   : 20}
matplotlib.rc('font', **font)

def lin(x,a,b):
    return x*a+b

def calc_diff(run, dt_offset = 0, mode = 'Pulse_On/Off',
              qnorm = [0.5,4], plotting = True, saving = True,
              filtering = [0.1,0.2],threshold = 0.1, hybrid=False,correlation =  True,corr_bound=0.05):

    Run = str(run)
    

    file = 'ChiRunRAW_Corr_full_'+Run+'_testing.h5'
    path = '/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/'
    
    data = h5py.File(path + file,'r')
    
    q  = data['q'][:,0]
    #q = rescaleQ(q,9.3e3,25)
    q_mask = (q > qnorm[0]) & (q < qnorm[1])
    mask_q = (q>0.32) & (q<4.3)
    
    pulses =  np.unique(data['pulseID'][:])
    n_pulses = pulses.size
    
    N_trains = int(data['pulseID'][:].size/n_pulses)
    # pulses_all = np.reshape(data['pulseID'][:],[N_trains,num_pulse])
    
    S = np.reshape(data['Sq_sa0'],[q.size,N_trains,n_pulses])

    
    #S = S[mask_q,:,:]
    #q = q[mask_q]
    
    TID = np.reshape(data['trainID'][:,0],[N_trains,n_pulses])
    
    delays = data['DelayRun'][:] - dt_offset
    
    dt = np.unique(delays)
    
    if correlation:
        print('correlation filtering is applied')
        tot_S = np.nansum(S,axis=0)
        tot_S_flat = np.reshape(tot_S, tot_S.size)
        
        run = open_run(2787,run)
        I0 = run['FXE_RR_DAQ/ADC/1:network','digitizers.channel_2_B.raw.samples'].ndarray()
        inds = np.ones([n_pulses,N_trains])
        for ii in range(N_trains):
            peaks = find_peaks(np.abs(I0[ii,:]),height=20, threshold=None, distance=220,prominence=1)
            inds[:,ii] = I0[ii,peaks[0]]
        
        I0_flat = np.reshape(np.abs(inds.T), tot_S.size)
        if plotting:
            fig,ax = plt.subplots()
            ax.plot(I0_flat,tot_S_flat,'.')
            ax.set_xlabel('I0 diode')
            ax.set_ylabel('tot abs scattering')
        
        rat = I0_flat/tot_S_flat
        mask = rat> np.nanmean(rat)/1.3
        

        model = Model(lin)
        params = model.make_params(a=0,b=0)
        result = model.fit(tot_S_flat,x = I0_flat,params=params)
        if plotting:
            ax.plot(I0_flat[mask],tot_S_flat[mask],'.')
            ax.plot(I0_flat,result.best_fit,'-')
        
        mask = np.reshape(mask,tot_S.shape)
        S[:,~mask] = np.nan 
        S =  np.reshape(S,[q.size,N_trains,n_pulses])
      
    
        mask = (tot_S_flat < (1+corr_bound)*result.best_fit) & (tot_S_flat > (1-corr_bound)*result.best_fit)
        if plotting:
            ax.plot(I0_flat[mask],tot_S_flat[mask],'.')
        
        mask = np.reshape(mask,tot_S.shape)
        S[:,~mask] = np.nan 
        S =  np.reshape(S,[q.size,N_trains,n_pulses])
        total = -np.sum(mask)
        
        
    if filtering == None:
             print('No intensity filtering is applied')
             total = 0
    else:
        tot_S = np.nansum(S,axis=0)
        tot_S_flat = np.reshape(tot_S, tot_S.size)
        m = np.nanmedian(tot_S_flat[tot_S_flat>0.2e6])
        lb = m-m*filtering[0]
        ub = m+m*filtering[1]

        mask = (tot_S_flat > lb) & (tot_S_flat < ub)
        mask = np.reshape(mask,tot_S.shape)
        S[:,~mask] = np.nan
        S =  np.reshape(S,[q.size,N_trains,n_pulses])
        total = -np.sum(mask)
        
        if plotting:
            fig,ax = plt.subplots()
            ax.set_xlabel('$\Sigma S$')
            ax.set_ylabel('#Curves')
            ax.set_title('Intensity distribution \n run '+ Run + ' with filtering bounds')
            ax.hist(tot_S_flat,100)
            ax.plot([ub,ub],[0,tot_S_flat.size/30],'k--')
            ax.plot([lb,lb],[0,tot_S_flat.size/30],'k--')
            ax.plot([m,m],[0,tot_S_flat.size/30],'r--')
            fig.tight_layout()

    
        
    
    
    bin_avg = S[0,:,:].size / dt.size
    #print(bin_avg)
    
    if mode == 'Pulse_On/Off':    
        ds = np.ones((dt.shape[0],q.shape[0]))*0
        bad_t = np.array([])
        diffs = np.ones([q.size,np.unique(TID).size])
        S_norm = np.ones_like(diffs)
        unique_tids = np.unique(TID)
        if not (np.diff(unique_tids) == 1).all():
            print('***********WE MIGHT HAVE LOST A TRAIN!**********')

        if hybrid:
            kicked_trains = unique_tids[1::2] 
            unique_tids = unique_tids[::2]
            #diffs = np.ones([q.size,unique_tids.size])
            kicked_diffs = np.ones([q.size,np.unique(TID).size])
        for ii,T in enumerate(unique_tids):
            mask = (T == TID)

            S_on = S[:,mask][:,::2]
            S_off = S[:,mask][:,1::2]
            #S_on = S[:,mask][:,0]
            #S_off = S[:,mask][:, 1]
            #S_on_norm = S_on / np.nansum(S_on[q_mask])
            #S_off_norm = S_off / np.nansum(S_off[q_mask])
            S_on_norm = S_on / np.nansum(S_on[q_mask,:],axis=0)[np.newaxis,:]
            S_off_norm = S_off / np.nansum(S_off[q_mask,:],axis=0)[np.newaxis,:]
            diff = np.nanmedian(S_on_norm-S_off_norm,axis=1)
            diffs[:,ii] = diff
            S_norm[:,ii] = np.median((S_on_norm+S_off_norm),axis=1)/2
        for ii,t in enumerate(dt):
            mask = t == delays
            ds[ii,:] = (np.nanmean(diffs[:,mask],axis=1))
            if bin_avg*threshold > np.sum(mask):
                bad_t = np.append(bad_t,int(np.where(dt==t)[0]))
                print('dt = ' + str(t) +' is thrown away, number of pulses:' + str(np.sum(mask)*n_pulses))
            else:
                total += np.sum(mask*n_pulses)
        ds = ds[:,mask_q]
        q = q[mask_q]
        dt_corr = np.delete(dt,np.array(bad_t,dtype=int))
        ds_corr = np.delete(ds,np.array(bad_t,dtype=int),axis=0)

        if hybrid: 
            ds_kicked = np.ones((dt.shape[0], q.shape[0])) * 0
            for ii,T in enumerate(kicked_trains):
                mask = (T == TID)
                S_on = np.nanmean(S[:,mask][:,::2],axis=1)
                S_off = np.nanmean(S[:,mask][:, 1::2],axis=1)
                S_on_norm = S_on / np.nansum(S_on[q_mask],axis=0)
                S_off_norm = S_off / np.nansum(S_off[q_mask],axis=0)
                diff = S_on_norm - S_off_norm
                kicked_diffs[:,ii] = diff
            for ii,t in enumerate(dt):
                mask = t == delays
                ds_kicked[ii,:] = (np.nanmean(kicked_diffs[:,mask],axis=1))
                if bin_avg*threshold > np.sum(mask):
                    bad_t = np.append(bad_t,int(np.where(dt==t)[0]))
                    print('dt = ' + str(t) +' is thrown away, number of pulses:' + str(np.sum(mask)*n_pulses))
    
    
    
    
    
    elif mode == 'Train_On/Off':
        
        bad_t = np.array([])
        ds = np.ones((dt.shape[0],q.shape[0]))*0
        
        for ii,t in enumerate(dt):
            mask_off =  (t == delays ) & (TID[:,0] %2!=0)
            mask_on = (t == delays ) & (TID[:,0] %2==0)
            
            S_on = np.nanmean(S[:,mask_on],axis=(1))[:,0]
            S_off = np.nanmean(S[:,mask_off],axis=(1))[:,0]
            
            S_on_norm = S_on / np.nansum(S_on[q_mask])
            S_off_norm = S_off / np.nansum(S_off[q_mask])

            ds[ii,:] = S_on_norm-S_off_norm
            if bin_avg*threshold > np.sum(mask_on*2):
               bad_t = np.append(bad_t,int(np.where(dt==t)[0]))
               print('dt = ' + str(t) +' is thrown away, number of pulses:' + str(np.sum([mask_on,mask_off])*n_pulses) + '\n')
            else:
                total += np.sum([mask_on,mask_off])*n_pulses
        mask_q = (q>0.5) & (q<5)
        ds = ds[:,mask_q]
        q = q[mask_q]
        dt_corr = np.delete(dt,np.array(bad_t,dtype=int))
        ds_corr = np.delete(ds,np.array(bad_t,dtype=int),axis=0)
        
    
    else:
        print('Error: mode should be either Train_On/Off or Pulse_On/Off ')

    if plotting:
        fig,ax=plt.subplots()
        #ax.set_title('Train On/Off')
        for ii,t in enumerate(dt_corr):
            ax.plot(q,ds_corr[ii,:])
        ax.set_xlabel('q  [$\AA^{-1}$]')
        ax.set_ylabel('$\Delta S_{azi}$')
        fig.tight_layout
    
    if saving == True:
        savedict = {
        'Sazi': ds_corr.T,
        'S2': ds_corr.T*np.nan,
        'S0': ds_corr.T*np.nan,
        'q' : q,
        'scanvar' : dt_corr,
        'avrS': np.nanmean(S_norm,axis=1)[mask_q]
        }
        if hybrid:
            savedict['S_kicked'] = ds_kicked
        savepath = '/gpfs/exfel/u/scratch/FXE/202201/p002787/Diffs/Run' + Run + '_Reduced' +'.mat'
        scipy.io.savemat(savepath,savedict)
        print('Diff signal saved to file: ' + savepath +'\n' )
    else:
        print('Diff signal is not saved \n')
        
    
    return print('Total number of curves used in diff signal: ' + str(total*2))