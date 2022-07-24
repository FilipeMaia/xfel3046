# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:49:07 2021

@author: mhaubro
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import scipy.io
import matplotlib
from lmfit import Model
plt.close('all')
data  = scipy.io.loadmat('Stacked_Runs_0044_0045_0050_0051_0052.mat')
q = np.squeeze(data['q'])
dt = -(np.squeeze(data['scanvar'])-2250)

S0 = data['S0']
S2 = data['S2']
Sazi = data['Sazi']
t_mask = (dt<2) &(dt>-0.65)
S0 = S0[:,t_mask]
S2 = S2[:,t_mask]
Sazi = Sazi[:,t_mask]
dt = dt[t_mask]
cmin= -1e-6
cmax= -cmin
figs,[ax,ax_corr]=plt.subplots(1,2,sharey=True)
ax.pcolormesh(dt,q,Sazi,vmin = cmin,vmax = cmax)
ax.set_xlabel('$\Delta t$ [ps]')
ax.set_ylabel('$q \ [\AA^{-1}]$')

ax_corr.set_xlabel('$\Delta t$ [ps]')
ax.set_title('Sazi')
ax_corr.set_title('corr')

#%%
mask = dt<-0.2
dt_bckg = dt[mask]
data_bckg = Sazi[:,mask]

fig,ax = plt.subplots(1,2)
ax[0].pcolormesh(dt_bckg,q,data_bckg,vmin = cmin,vmax = cmax)

font = {      
        'size'   : 22}

#plt.rcParams.update({
 #   "text.usetex": True
#})
matplotlib.rc('font', **font)





offset = 1e-5

u,s,vh = svd(data_bckg,full_matrices=False)
fig9,ax9 = plt.subplots(1,4,figsize =[20,8] )
ax9[0].pcolor(dt_bckg,q,u@np.diag(s)@vh)
num_comp = 5
for ii in range(num_comp):
    
    ThisSign= np.sign(np.sum(np.sign(vh[ii,:])));
    ax9[1].plot(ii+1,s[ii],'.',Markersize=25)
    
    ax9[2].plot(q,s[ii]*u[:,ii]*ThisSign+ii*offset,LineWidth=3)
    ax9[2].plot([q[0],q[-1]],[offset*ii,offset*ii],'k--',LineWidth=3)    
    
    ax9[3].plot(dt_bckg,s[ii]*vh[ii,:]*ThisSign+ii*offset,LineWidth=3)
    ax9[3].plot([dt_bckg[0],dt_bckg[-1]],[offset*ii,offset*ii],'k--',LineWidth=3)
    
ax9[0].set_title('U*S*V')
ax9[0].set_xlabel('$\Delta t$ / ps')
ax9[0].set_ylabel('$q$ / $\AA^{-1}$')

ax9[1].set_title('$S_{i,i}$ (magnitude)')
ax9[1].set_xlabel('i')

ax9[2].set_title('$S_{i,i}*U_i$  (typogram)')
ax9[2].set_xlabel('$q$ / $\AA^{-1}$')

ax9[3].set_title('$S_{i,i}*V_i$  (chronogram)')
ax9[3].set_xlabel('$\Delta t$ / ps')
fig9.tight_layout()
plt.show
S0[np.isnan(S0)] = 0
S2[np.isnan(S2)] = 0
Sazi[np.isnan(Sazi)] = 0
def func(q,alpha0,alpha1,alpha2,alpha3,alpha4,alpha5):
    return alpha0 * u[:,1] #+alpha1 * u[:,1]+alpha2 * u[:,2]+alpha3 * u[:,3] +alpha4 * u[:,4]+alpha5 * u[:,5]

model = Model(func)
params = model.make_params(alpha0=1e-7,alpha1=0,alpha2=0,alpha3=0,alpha4=0,alpha5=0)

fig1,ax1 = plt.subplots()
ax1.set_xlabel('dt')
ax1.set_ylabel('alphas')

for ii,t in enumerate(dt): 
    result = model.fit(Sazi[:,ii],params,q=q)
    Sazi[:,ii]-= result.best_fit
    ax1.plot(t,result.best_values['alpha0'],'r.')
    # ax1.plot(t,result.best_values['alpha1'],'b.')
    # ax1.plot(t,result.best_values['alpha2'],'k.')
    # ax1.plot(t,result.best_values['alpha3'],'g.')
    # ax1.plot(t,result.best_values['alpha4'],'m.')
    # ax1.plot(t,result.best_values['alpha5'],'x')
    
#%%
ax_corr.pcolormesh(dt,q,Sazi,vmin = cmin,vmax = cmax)

#%%


offset = 1e-5

u,s,vh = svd(Sazi,full_matrices=False)
fig9,ax9 = plt.subplots(1,4,figsize =[20,8] )
ax9[0].pcolor(dt,q,S2,vmin=cmin,vmax=cmax)
num_comp = 5
for ii in range(num_comp):
    
    ThisSign= np.sign(np.sum(np.sign(vh[ii,:])));
    ax9[1].plot(ii+1,s[ii],'.',Markersize=25)
    
    ax9[2].plot(q,s[ii]*u[:,ii]*ThisSign+ii*offset,LineWidth=3)
    ax9[2].plot([q[0],q[-1]],[offset*ii,offset*ii],'k--',LineWidth=3)    
    
    ax9[3].plot(dt,s[ii]*vh[ii,:]*ThisSign+ii*offset,LineWidth=3)
    ax9[3].plot([dt[0],dt[-1]],[offset*ii,offset*ii],'k--',LineWidth=3)


ax9[0].set_title('$\Delta S_2$')
ax9[0].set_xlabel('$\Delta t$ / ps')
ax9[0].set_ylabel('$q$ / $\AA^{-1}$')

ax9[1].set_title('$S_{i,i}$ (magnitude)')
ax9[1].set_xlabel('i')

ax9[2].set_title('$S_{i,i}*U_i$  (typogram)')
ax9[2].set_xlabel('$q$ / $\AA^{-1}$')

ax9[3].set_title('$S_{i,i}*V_i$  (chronogram)')
ax9[3].set_xlabel('$\Delta t$ / ps')
fig9.tight_layout()