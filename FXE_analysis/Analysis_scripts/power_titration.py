# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:36:54 2022

@author: mhaubro
"""


import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import matplotlib
import lcls
from lcls import Scan
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def power_titration(runs, powers,offset = 1e-4 ):

    rhodat = np.loadtxt('../References/KSK_WaterdSdrho_Scaled.txt')
    tempdat = np.loadtxt('../References/KSK_WaterdSdT_Scaled.txt')

    temp_interp = interp1d(tempdat[:,0], tempdat[:,1], kind='cubic',fill_value=0,bounds_error=False)
    dens_interp = interp1d(rhodat[:,0], rhodat[:,1], kind='cubic',fill_value=0,bounds_error=False)

    def fit(q,dT,drho):
        return temp_interp(q)*dT + dens_interp(q)*drho*0

    path = '/gpfs/exfel/u/scratch/FXE/202201/p002787/Diffs/'
    Laser_powers = powers
    run_numbers = runs




    fig1,ax1=plt.subplots()
    ax1.set_xlabel('q [$\AA^{-1}$]')
    ax1.set_ylabel('$\Delta S$')



    fig,ax=plt.subplots()
    ax.set_xlabel('q')
    ax.set_ylabel('$\Delta S_{norm}$')



    fig4,ax4=plt.subplots(1,3,sharey=True,figsize=[14,7])
    ax4[0].set_xlabel('q [$\AA^{-1}$]')
    ax4[1].set_xlabel('q [$\AA^{-1}$]')
    ax4[2].set_xlabel('q [$\AA^{-1}$]')
    ax4[0].set_ylabel('$\Delta S$')

    ax4[0].set_title('Fit components')
    ax4[1].set_title('Data and Fit')
    ax4[2].set_title('residual')



    abssum=[]
    ratio = []
    dT = []
    drho = []

    

    col_ord =plt.get_cmap('tab10')(range(10))
    for ii,num in enumerate(run_numbers):
        file = path + 'Run' + str(num) + '_Reduced.mat'
        scan =Scan(file)
        q = scan.q
        dt =scan.time
        S = np.nanmean(scan.atd_az[:,dt>0],axis=1)

        ax.plot(q,savgol_filter(S,31,5)/powers[ii],label = str(Laser_powers[ii]) + ' uJ')
        ax1.plot(q,savgol_filter(S,31,5),label = str(Laser_powers[ii]) + ' uJ')

        abssum.append(np.sum(np.abs(S)))

        model = Model(fit)
        params = model.make_params(dT = 1e-5,drho = 1e-5)
        result = model.fit(S,params, q=q)
        ax4[0].plot(q,temp_interp(q)*result.best_values['dT'] +ii*offset,'-',color = col_ord[-1,:-1])
        ax4[0].plot(q,dens_interp(q)*result.best_values['drho']*0+ii*offset,'-',color = col_ord[-2,:-1])
        ax4[1].plot(q,S+ii*offset,'.',color = col_ord[ii,:-1])
        ax4[1].plot(q,result.best_fit+ii*offset,'k-',linewidth = 2)
        ax4[2].plot(q,(S-result.best_fit)+ii*offset,'.',color= col_ord[ii,:-1])
        ax4[2].plot(q,(S-result.best_fit)+ii*offset,'.',color = col_ord[ii,:-1])
        string = str(Laser_powers[ii]) + ' uJ'
        #ax4[2].text(7, ii*1.2*offset,string,Fontsize = 22,color = col_ord[ii,:-1])
        dT.append(result.best_values['dT'])
        drho.append(result.best_values['drho'])

    fig4.tight_layout()
    
    fig2,ax2=plt.subplots()
    ax2.set_xlabel('Laser Power [uJ]')
    ax2.set_ylabel('$\Sigma |\Delta S|$')
    ax2.plot(Laser_powers,abssum,'o')


    fig2.tight_layout()


    ax1.legend()
    ax.legend()
    fig.tight_layout()
    fig1.tight_layout()



    fig5,ax5=plt.subplots()
    ax5.set_xlabel('Laser Power [uJ]')
    ax5.set_ylabel('fit param')
    ax5.plot(Laser_powers,dT,'.',label = 'dT')
    ax5.plot(Laser_powers,drho,'.',label = 'drho')
    ax5.legend()
    fig5.tight_layout()
    
