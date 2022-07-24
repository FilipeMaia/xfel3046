# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:21:12 2022

@author: mhaubro


@ 2022_03_22 plenzen created this fork to generate 2D images for S0/S2 decomposition
    at the time of creation only the pulseon/Off is working. Train support is to be implemented

inputs:
    run: run number
    
    dt_offset: time zero offset
    
    mode: 'Pulse_On/Off' or 'Train_On/Off', denoting the two exciation schemess used at the FXE
    
    qnorm: normalisation range for calculating difference scattering
    
    plotting: True or False, chose whether results should be plotted
    
    saving: Chose whether results should be saved
    
    filtering: either None or [lb, ub], intensity based filtering is then implemented filtering out all curves with total inensity below m - m*lb and above m + m*ub, where m is the mean of the intensity
    
    threshold: controls whether or not time bins with very few curves are thrown out, 0 keeps everything, to get rid of them 0.01 is an appropriate number. Numbers close to one and will remove everything
    
    offsetangle: Angle between X-ray and laser polarization in degrees. Determine it with the function "test_offset_angle"

    test_offset_angle: If true, returns a cake sliced image integrated over the last timebin. to be used for test_offset_angle
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
from scipy.stats.mstats import theilslopes

font = {        'size'   : 20}
matplotlib.rc('font', **font)



def calc_diff_2D(run, dt_offset = 0, mode = 'Pulse_On/Off',qnorm = [1,4],
                 plotting = True, saving = True,filtering = [0.1, 0.2],
                 threshold = 0.01, offsetangle=67, return_diff_image = False):

    Run = str(run)

    file = 'ChiRunRAW_Corr_part_'+Run+'_testing_2D.h5'
    file = 'ChiRunRAW_Corr_full_'+Run+'_testing_2D.h5'
    path = '/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/'
    
    data = h5py.File(path + file,'r')
    
    q  = data['q'][:,0]
    q_mask = (q > qnorm[0]) & (q < qnorm[1])
    
    pulses =  np.unique(data['pulseID'][:])
    n_pulses = pulses.size
    
    N_trains = int(data['pulseID'][:].size/n_pulses)
    # pulses_all = np.reshape(data['pulseID'][:],[N_trains,num_pulse])
    
    q = data['q'][:,0]
    swapped = np.swapaxes(data['Sq_sa0'], 1,2)
    #S = np.reshape(data['Sq_sa0'],[q.size,17,N_trains,n_pulses])
    S = np.reshape(swapped ,[q.size,17,N_trains,n_pulses])

    #S = np.reshape(data['Sq_sa0'],[17,q.size,N_trains,n_pulses])
    
    TID = np.reshape(data['trainID'][:,0],[N_trains,n_pulses])
    
    delays = data['DelayRun'][:] - dt_offset
    
    dt = np.unique(delays)
    
    
    if filtering == None:
         print('No intensity filtering is applied')
         total = 0
    else:
        tot_S = np.sum(S,axis=0)
        tot_S_flat = np.reshape(tot_S, tot_S.size)
        m = tot_S_flat.mean()
        lb = m-m*filtering[0]
        ub = m+m*filtering[1]
    
        mask = (tot_S_flat > lb) & (tot_S_flat < ub)
        mask = np.reshape(mask,tot_S.shape)
        S[:,~mask] = np.nan 
        S =  np.reshape(S,[q.size,17,N_trains,n_pulses])
        #S =  np.reshape(S,[17,q.size,N_trains,n_pulses])
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
    
    if mode == 'Pulse_On/Off':    
        ''' Makes diff curves from trains that have the shape 10101010... 10101010...  
            (1: LaserOn, 2: LaserOff)
        '''
        ds = np.empty((dt.shape[0], q.shape[0], 17))
        bad_t = np.array([])
        diffs = np.empty([q.size, 17, np.unique(TID).size])
        if return_diff_image:
            T = np.unique(TID)[-2] 
            mask = (T == TID)
            S_on = np.nanmean(S[:, :, mask][:, :, ::2], axis=2)
            return S_on

            S_off = np.nanmean(S[:, :, mask][:, :, 1::2], axis=2)
            S_on_norm = S_on / np.nanmean(S[:, :, mask][q_mask,:, ::2])
            S_off_norm = S_off / np.nanmean(S[:, :, mask][q_mask,:, 1::2])
            diff = S_on_norm-S_off_norm
            diff[np.where(diff == 0)] = np.nan
            print("shape S_on: ", np.shape(S_on))
            print("S_on isnan: ", sum(np.isnan(S_on)))
            print("shape mask: ", np.shape(mask))
            print("shap S: ", np.shape(S))
            print("isnan in S: ", np.sum(np.isnan(S)))
            print("size s: ", np.size(S))

            return diff
        else:        
            for ii, T in enumerate(np.unique(TID)):
                mask = (T == TID)
                
                S_on = np.nanmean(S[:,:,mask][:,:,::2],axis=2)
                S_off = np.nanmean(S[:,:,mask][:,:, 1::2],axis=2)

                S_on_norm = S_on / np.nanmean(S[:,:,mask][q_mask,:, ::2])
                S_off_norm = S_off / np.nanmean(S[:,:,mask][q_mask,:, 1::2])

                diff = S_on_norm-S_off_norm
                diff[np.where(diff == 0)] = np.nan  # XXX we can get rid of np.where here right? 
                diffs[:,:,ii] = diff

            for ii,t in enumerate(dt):
                mask = t == delays
                ds[ii,:,:] = (np.nanmean(diffs[:,:,mask],axis=2))
                if bin_avg*threshold > np.sum(mask):
                    bad_t = np.append(bad_t,int(np.where(dt==t)[0]))
                    print('dt = ' + str(t) +' is thrown away, number of pulses:' + str(np.sum(mask)*44))
                else:
                    total += np.sum(mask*44)
            dt_corr = np.delete(dt,np.array(bad_t,dtype=int))
            ds_corr = np.delete(ds,np.array(bad_t,dtype=int),axis=0)
    
    elif mode == 'Train_On/Off':
        ''' Makes diff curves from trains that have the shape 11111111... 00000000...  
            (1: LaserOn, 2: LaserOff)
            Mode will probably never be used, as its for XES and the reprate will be so 
                high that the jet is destroyed and the scattering is unusable.
        '''
        bad_t = np.array([])
        ds = np.empy((dt.shape[0],q.shape[0]))
        
        for ii,t in enumerate(dt):

            ## old
            mask_off =  (t == delays ) & (TID[:,0] %2!=0)
            mask_on = (t == delays ) & (TID[:,0] %2==0)
            
            S_on = np.nanmean(data['Sq_sa0'][:,mask_on],axis=1)
            S_off = np.nanmean(data['Sq_sa0'][:,mask_off],axis=1)
            
            S_on_norm = S_on / np.nansum(S_on[q_mask])
            S_off_norm = S_off / np.nansum(S_off[q_mask])

            ## new
            S_on = np.nanmean(swapped[:,:,mask_on],axis=2)
            S_off = np.nanmean(swapped[:,:,mask_off],axis=2)

            S_on_norm =  S_on / np.nanmean(swapped[:,:,mask_on])
            S_off_norm = S_off / np.nanmean(swapped[:,:,mask_off])

            diff = S_on_norm-S_off_norm
            diff[np.where(diff == 0)] = np.nan
            ds[ii] = diff

            if bin_avg*threshold > np.sum(mask_on*2):
               bad_t = np.append(bad_t,int(np.where(dt==t)[0]))
               print('dt = ' + str(t) +' is thrown away, number of pulses:' + str(np.sum([mask_on,mask_off])*44) + '\n')
            else:
                total += np.sum([mask_on,mask_off])*44
                
        dt_corr = np.delete(dt,np.array(bad_t,dtype=int))
        ds_corr = np.delete(ds,np.array(bad_t,dtype=int),axis=0)
        
    ## decompose S0/S2
    print(dt.shape[0],q.shape[0],3)
    S_total = np.empty((dt.shape[0],q.shape[0],3))
    
    phi_angles = np.deg2rad(np.linspace(0,360,18)[:-1]+90 +offsetangle)
    print("image shape: ", np.shape(np.nanmean(ds, (0))))
    for i in range(len(dt)):
        #S_total[i,:] = decompose_S0_S2_Az(np.nanmean(ds, (0)), phi_angles,q)
        S_total[i,:] = decompose_S0_S2_Az(ds[i,:,:], phi_angles,q)    
    else:
        print('Error: mode should be either Train_On/Off or Pulse_On/Off ')
        
    if plotting:
        fig,ax=plt.subplots()
        ax.set_title('Train On/Off')
        for ii,t in enumerate(dt_corr):
            ax.plot(q,ds_corr[ii,:])
        ax.set_xlabel('q  [$\AA^{-1}$]')
        ax.set_ylabel('$\Delta S_{azi}$')
        fig.tight_layout
    
    if saving == True:
        info = {}
        info['q_norm'] = qnorm
        info['mode'] = mode
        info['dt_offset'] = dt_offset
        info['filtering'] = filtering
        info['threshold'] = threshold
        info['run'] = run
        
        savedict = {
        'Sphi': ds.T,
        'q' : q,
        'scanvar' : dt,
        'info' : info,
        #'avrS': np.nanmean(data['Sq_sa0'],axis=1)
        }
        savepath = '/gpfs/exfel/u/scratch/FXE/202201/p002787/Diffs/Run' + Run + '_Reduced_binnedphi' +'.mat'
        scipy.io.savemat(savepath,savedict)
        print('binned phi signal saved to file: ' + savepath +'\n' )
        
        info = {}
        info['q_norm'] = qnorm
        info['mode'] = mode
        info['dt_offset'] = dt_offset
        info['filtering'] = filtering
        info['threshold'] = threshold
        info['offsetangle'] = offsetangle
        info['run'] = run
        
        print("shape S_total", np.shape(S_total))
        
        savedict = {
        'Sazi': S_total[:,:,0].T,
        'S0': S_total[:,:,1].T,
        'S2': S_total[:,:,2].T,
        'q' : q,
        'scanvar' : dt,
        'info' : info,
        #'avrS': np.nanmean(data['Sq_sa0'],axis=1)
        }
        savepath = '/gpfs/exfel/u/scratch/FXE/202201/p002787/Diffs/Run' + Run + '_Reduced_decomposed' +'.mat'
        scipy.io.savemat(savepath,savedict)
        print('binned phi signal saved to file: ' + savepath +'\n' )
        
    else:
        print('Diff signal is not saved \n')
        
    
    return print('Total number of curves used in diff signal: ' + str(total*2))


def decompose_S0_S2_Az(image, phi_angles,q):
    S = np.empty((len(q),3))
    for q_index in range(len(q)):
        if sum(~np.isnan(image[q_index,:])) > 2:
            
            non_nan_indeces = ~np.isnan(image[q_index,:])
            #print("q_index: ", q_index,image[q_index,:], ", non_nan_indeces: ", non_nan_indeces)
            Az_slice = image[q_index,:]
            x = -np.cos(phi_angles)
            P2 = 0.5 * ((3*x**2)-1)
            #print(Az_slice[non_nan_indeces], P2[non_nan_indeces])
            A,B,C,D = theilslopes(Az_slice[non_nan_indeces], P2[non_nan_indeces])
            #print(A,B)
            S[q_index,1] = B
            S[q_index,2] = -A
        S[q_index,0] = np.nanmedian(image[q_index,:])    
    return S

def test_offset_angle(image):
    '''
        This function tests which offset angle produces the largest S2 signal
    '''
    S2_list = []
    offset_list = np.linspace(0,180,45)
    for offset in offset_list:
        phi_angles = np.deg2rad(np.linspace(0,360,18)[:-1]+90 +offset)
        S = decompose_S0_S2_Az(image, phi_angles,np.arange(512))
        S2_list.append(np.nanmedian(abs(S[:,2])))
    return offset_list, S2_list

