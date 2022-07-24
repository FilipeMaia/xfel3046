# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:46:37 2021

@author: mhaubro
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import h5py
from scipy.signal import savgol_filter
import matplotlib
import scipy.io
from tqdm import tqdm
from ase.io import read
from debye import Debye
from scipy.interpolate import interp1d,interp2d
import warnings
import glob
from lmfit import Model, Parameters
from rescaleQ import rescaleQ

plt.close('all')
run=104

Run = str(run)

file = '\ChiRunRaw_Corr_full_' + Run + '.h5'
path = '..\Test_Data'

data = h5py.File(path + file,'r')

num_pulse = np.unique(data['pulseID'][:]).size
pulse_num =  np.unique(data['pulseID'][:])
N_trains = int(data['pulseID'][:].size/num_pulse)
pulses = np.reshape(data['pulseID'][:],[N_trains,num_pulse])
q = data['q'][:,0]
S = np.reshape(data['Sq_sa0'],[q.size,N_trains,num_pulse])
dt = data['DelayRun'][:]

TID = data['trainID']

tot_S = np.sum(S,axis=0)

tot_S_flat = np.reshape(tot_S, tot_S.size)
m = tot_S_flat.mean()
mask = (tot_S_flat > m - m*100) & (tot_S_flat < m + m*100)
mask = np.reshape(mask,tot_S.shape)
S[:,~mask] = np.nan 
S =  np.reshape(S,[q.size,N_trains,num_pulse])
#%%

q_mask = (q>0) &(q<5)
S = S[q_mask,:,:]
q = q[q_mask]

#q= rescaleQ(q, 9.3e3, -0.8)
scatt = Debye(q)


#dt = np.abs(dt-dt.max())-5



mask_q = (q>1) &(q<4.5)

#%%
# mask_q = qq<3
kk = 1
S_train = np.nanmean(S,axis=1)#np.sum(S_train[:,mask_q],axis=1)[:,np.newaxis]
S_train_norm = S_train/np.nansum(S_train[mask_q,:],axis=0)[np.newaxis,:]
fig5,ax5=plt.subplots()
c=ax5.pcolormesh(pulse_num,q,(S_train_norm-S_train_norm[:,kk][:,np.newaxis]))
ax5.set_xlabel('Pulse No.')
ax5.set_ylabel('$q \ \mathrm{[\AA^{-1}]}$')
ax5.set_title('Avg. scattering diff. $P_n-P_{}$'.format(pulse_num[kk]))
        
fig5.tight_layout()
fig6,ax6 = plt.subplots()

for ii in range(pulse_num.size):
    ax6.plot(q,S_train_norm[:,ii])
#ax6.plot(q,S_train_norm[:,0],'r',label='Pulse 0')
#ax6.plot(q,S_train_norm[:,1],'k',label='Pulse 1')
#ax6.legend()
ax6.set_xlabel('q / Å$^{-1}$')
ax6.set_ylabel('$S$')
ax6.set_title('Normalised scattering') 
fig6.tight_layout()
I = np.sum(S_train,axis=0)
Var_I =np.std(S_train,axis=0)
fig9,ax9 = plt.subplots()
ax9.set_xlabel('pulse')
ax9.set_ylabel('$\Sigma S$')    
ax9.plot(pulse_num,I,'.-')
ax9.fill_between(pulse_num, I-Var_I, I+Var_I,alpha=0.6)
ax9.plot(pulse_num[50:99],I[50:99])
ax9.set_title('Average scattering intensity, first 50 pulses')
fig9.tight_layout()
#%%

Abs = np.array([])
Var = np.array([])
fig,ax = plt.subplots(figsize=[5,10])
ax.set_xlabel('$q$  [Å$^{-1}$]')
ax.set_ylabel('$\Delta S$')
for pulse in pulses:
    Sp = S[:,pulse,:]
    dtp = dt[:,pulse]
    TIDp = TID[:,pulse]
    mask = TIDp%2 == 0
    Son = np.nanmean(Sp[mask,:],axis=0)
    Soff = np.nanmean(Sp[~mask,:],axis=0)
    ds = Son/np.nanmean(Son[mask_q])-Soff/np.nanmean(Soff[mask_q])
    ax.plot(ds+pulse*0.008,'.')
    Abs = np.append(Abs,np.nansum(np.abs(ds)))
    Var = np.append(Var,np.nanstd(np.abs(ds)))

fig.tight_layout()
fig1,ax1 = plt.subplots()
ax1.errorbar(pulse_num,Abs,yerr=Var,linestyle='',marker='.')
ax1.set_xlabel('Pulse No.')
ax1.set_ylabel('$\Sigma \Delta S$')
fig1.tight_layout()

mask = Abs>0.3
ind_sig = pulse_num[mask]

#%%
tempdat = np.loadtxt('../LCLS2015/KSK_WaterdSdT_Scaled.txt')
# tempdat = np.loadtxt('ACNheating.txt',skiprows=1)
temp_interp = interp1d(tempdat[:,0], tempdat[:,1], kind='cubic')

MC3 = read('../Structures/FebpyCN4_3mc.xyz')
MLCT3 = read('../Structures/FebpyCN4_3mlct.xyz')
# 
GS = read('../Structures/FebpyCN4.xyz')
scatt = Debye(q)
DS = scatt.debye_numba(MC3)-scatt.debye_numba(GS)

S_on = np.ones([delay.size,q.size])*0
S_off = np.ones([delay.size,q.size])*0
ds = np.ones([delay.size,q.size])*0
Sum = 0
dT = np.ones((delay.size,pulses.size))*0;MC = np.ones((delay.size,pulses.size))*0
jj=0;kk=0;dd=0
def fit2(q,MC,dT):
    return MC*DS+dT*temp_interp(q)
# fig10,ax10 = plt.subplots(figsize=[20,10],nrows=5,ncols=10)
for pulse in pulses:
    Sp = S[:,pulse,:]
    dtp = dt[:,pulse]
    TIDp = TID[:,pulse]
    ii=0
    for T in delay:
        mask = TIDp%2==0
        S_odd = Sp[~mask,:]
        S_even = Sp[mask,:]
        
        dt_odd = dtp[~mask]
        dt_even = dtp[mask]
        
        mask_odd = (T-0.4< dt_odd) & (T+0.4> dt_odd)
        mask_even = (T-0.4< dt_even) & (T+0.4> dt_even)
        S_on[ii,:] = np.nanmean(S_even[mask_even,:],axis=0)
        S_on[ii,:] = S_on[ii,:]/np.nansum(S_on[ii,mask_q])
        
        S_off[ii,:] = np.nanmean(S_odd[mask_odd,:],axis=0)
        # S_off[ii,:] = S_off[ii,:]/np.sum(S_off[ii,mask_q])
        def fit(q,a):
            return a*S_off[ii,:]
        
        model = Model(fit)
        params = model.make_params(a = 1)
        result = model.fit(S_on[ii,:],params,q=q,nan_policy='propagate')
        S_off[ii,:] = result.best_fit
        ds[ii,:] = S_on[ii,:]- S_off[ii,:]
        model = Model(fit2)
        params = model.make_params(dT = 0,MC=0)
        result = model.fit(ds[ii,:],params,q=q,nan_policy='propagate')
        dT[ii,jj] = result.best_values['dT']
        MC[ii,jj] = result.best_values['MC']
        ii+=1
    # ax10[dd,kk].pcolormesh(delay,q,ds.T)
    jj+=1
    dd+=1
    if dd==5:
        dd=0
        kk+=1

#%%
for ii , T in enumerate(delay):
    mask_odd = (T-0.05< dt_odd) & (T+0.05> dt_odd)
    mask_even = (T-0.05< dt_even) & (T+0.05> dt_even)
    S_on[ii,:] = np.nanmean(S_even[mask_even,:],axis=0)
    S_on[ii,:] = S_on[ii,:]/np.nansum(S_on[ii,mask_q])
    
    S_off[ii,:] = np.nanmean(S_odd[mask_odd,:],axis=0)
    # S_off[ii,:] = S_off[ii,:]/np.sum(S_off[ii,mask_q])
    def fit(q,a):
        return a*S_off[ii,mask_q]
    model = Model(fit)
    params = model.make_params(a = 1)
    result = model.fit( S_on[ii,mask_q],params,q=q,nan_policy='propagate')
    S_off[ii,:] = result.best_values['a']*S_off[ii,:]
    ds[ii,:] = S_on[ii,:]- S_off[ii,:]
    
plt.pcolormesh(delay,q,ds.T)
#%%
fig99,ax99 = plt.subplots()
for jj in range(pulses.size):
    ax99.plot(delay,dT[:,jj])
    # plt.plot(delay,MC[:,jj])
    
#%%
fig11,ax11 = plt.subplots()
ax11.plot(I0_pulse,Abs,'.')
ax11.set_xlabel('$\Sigma S$')
ax11.set_ylabel('$\Sigma |\Delta S|$')

#%%
fig10,ax10 = plt.subplots()
ax10.plot(pulse_num,Abs,'.-')
ax10.set_xlabel('Pulse No.')
ax10.set_ylabel('$\Sigma |\Delta S|$')

#%%
# fig6,ax6 = plt.subplots()
fig9,ax9 = plt.subplots()
ax9.set_xlabel('pulse')
ax9.set_ylabel('$\Sigma S$')
col =plt.get_cmap('Purples')
# val_cm = cm(dist/dist.max())
col = col(np.linspace(0,31,32)/31)
for ii in range(32):
    S_train = np.nanmean(S[ii*100:((ii+1)*(100)-1)],axis=0)#np.sum(S_train[:,mask_q],axis=1)[:,np.newaxis]
    # S_train_norm = S_train/np.nansum(S_train[:,mask_q],axis=1)[:,np.newaxis]
    # fig5,ax5=plt.subplots()
    # c=ax5.pcolormesh(pulse_num,q,(S_train-S_train[kk,:]).T)
    # ax5.set_xlabel('Pulse No.')
    # ax5.set_ylabel('$q \ \mathrm{[\AA^{-1}]}$')
    # ax5.set_title('Avg. scattering diff. $P_n-P_{}$'.format(pulse_num[kk]))
            
    # fig5.tight_layout()
    I = np.sum(S_train,axis=1)
    # Var_I =np.std(S_train,axis=1)
 
    ax9.plot(pulse_num,I,'.-',color = col[ii,:],label = 'Train ' + str(ii*100) + ' to ' + str((ii+1)*100-1))
    # ax9.fill_between(pulse_num, I-Var_I, I+Var_I,alpha=0.6)
    # ax9.plot(pulses[50:99],I[50:99])
    # ax9.set_title('Average scattering intensity, first 50 pulses')
fig9.legend()
fig9.tight_layout()
    