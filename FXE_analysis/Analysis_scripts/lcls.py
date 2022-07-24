"""
@author: Morten Haubro
@contributors: Morten Haubro, Asmus Ougaard Dohn
"""


import itertools, copy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.io import loadmat,savemat
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor, Ridge 
import scipy.stats as ss
from scipy.signal import medfilt
from tqdm.notebook import tqdm

class Scan(dict):
    ''' Scan object. 
    
        Can read reduced data, decompose the signal into S0, S2, AZ. 
        Can plot surf plots of timescans, do SVD, and scan polarization.

        Parameters:

        infile: str
            Full path to .mat (can also read .npz, but probably poorly)
        energy: float
            The X-ray energy of the scan, in keV. 
        linreg: str, "manual", "sklearn", or "elisa"
            which type of linreg to use.

    '''

    def __init__(self, infile, energy=18, linreg='manual'):
        if infile.endswith('.npz'):
            d = np.load(infile, allow_pickle=True)
        elif infile.endswith('.mat'):
            d = loadmat(infile)
        else:
            raise IOError('Can only read .npz or .mat')
        # This is a bit magic, but it just loads whatever vars in the .mat
        # into self.
        #super(Scan, self).__init__(d)
        super().__init__(d)
        self.__dict__ = self

        if 'energy' not in self.keys():  # if loading _reduced you already get this
            self.energy = energy

        self.time = self.scanvar[0]  # XXX assumes that time is what is being scanned. NOT GOOD
        if len(self.scanvar.shape) > 1:
            self.time = self.scanvar.reshape(-1)
        if len(self.q) == 1:
            self.q = self.q[0]
        if len(self.q.shape) > 1:
            self.q = self.q.reshape(-1)
        
        if 'offset' not in self.keys():
            self.offset = 67  # Angle offset from vertical polarization

        if 'Sazi' in self.keys():  # you have loaded a _reduced, not _binnedPhi
            self.atd_s0 = self.S0
            self.atd_s2 = self.S2
            self.atd_az = self.Sazi

        self.qmax = None  # For cutting q range 
        self.qmin = None

        self.linreg = linreg 

        self.tt = np.rad2deg(2 * np.arcsin(self.q * 12.3987 / energy / (4 * np.pi)) )
        self.rb_time = None
        self.sp_timesteps = None

    def make_qmask(self):
        qmask = np.zeros(len(self.q), bool)
        if all(x is not None for x in [self.qmin, self.qmax]):
            qmask[(self.q < self.qmax) & (self.q > self.qmin)] = True
        else:
            qmask = np.ones(len(self.q), bool) 
        return qmask

    def decompose_signal(self, debugplot=None):
        ''' Make S0, S2, and Az signals. 
            ts: (str)
                can be "sklearn" or "manual" 
                choose  between sklearn's and our own Theil Sen Regression 
            '''
        first = True
        tsr = {'sklearn':self.tsr_skl,
               'manual':self.tsr_man,
               'elisa':self.theil_sen_stats,
               'siegels':self.reg_siegelslopes,
               'ransac':self.reg_ransac,
               'huber':self.reg_huber,
               'ridge':self.reg_ridge}[self.linreg]

        qmask = self.make_qmask()

        # AllTTDelay. 
        atd_s0 = np.zeros((len(self.q), len(self.time)))
        atd_s2 = np.zeros((len(self.q), len(self.time))) 
        atd_az = np.zeros((len(self.q), len(self.time))) 

        if self.sp_timesteps is not None:
            do_times = np.zeros(len(self.time), bool)
            do_times[self.sp_timesteps] = True
        else:
            do_times = np.ones(len(self.time), bool) 

        for tbi in tqdm(range(len(self.time))): 
            if not do_times[tbi]:
                continue
            s = np.zeros((len(self.q), 2))
            for qbi, q in enumerate(self.q):
                if not qmask[qbi]:  # Skip everything outside of qmin & qmax
                    continue
                az_bins_with_an = np.isfinite(self.Sphi[tbi, :, qbi])  # indices of non-NaNs
                if len(az_bins_with_an) == 0:
                    continue
                image_ra2 = self.Sphi[tbi,:,:]
                if np.sum(az_bins_with_an.ravel()) > 2:
                    angle = np.linspace(0, 360, len(self.cake[:, 0]) + 1)
                    angle = angle[:-1]  # not the 360 plz
                    this_angle = angle[az_bins_with_an]  # phi
                    this_angle += self.offset + 90  # 90 from vertical pol. 
                    this_angle[this_angle > 180] -= 360 #+ this_angle[this_angle > 180] 

                    #this_r = image_ra2[az_bins_with_an, qbi]
                    this_r = self.tt[qbi]

                    this_i = image_ra2[az_bins_with_an, qbi]
                    x = - np.cos(np.deg2rad(this_r / 2)) * np.cos(np.deg2rad(this_angle))
                    p2 = 0.5 * ( (3 * x**2) -1 )
                    a, b = tsr(p2[:, None], this_i)
                    s[qbi, :] = [a, b]
                    if debugplot is not None:
                        ax = debugplot[1]
                        fig = debugplot[0]
                        cam = debugplot[2]
                        if qbi in [350, 400, 450]:
                            ax.cla()
                            ax.plot(p2, this_i, 'k+')
                            xfit = np.linspace(np.min(p2)-.1, np.max(p2)+.1, 100)
                            ax.plot(xfit, a*xfit + b)
                            ax.set_title(f'Q:{q:2.2f} Time: {self.time[tbi] :2.2f} ps')
                            fig.tight_layout
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            first = False
                            cam.snap()


            atd_s0[:, tbi] = s[:, 1]
            atd_s2[:, tbi] = -s[:, 0]
            atd_az[:, tbi] = np.nanmean(self.Sphi[tbi, :, :], axis=0)

        self.atd_s0 = atd_s0
        self.atd_s2 = atd_s2
        self.atd_az = atd_az

        if debugplot:
            return cam

    def rebin_time(self, tmin=None, tmax=None, dt=50e-3, equi_t=None):
        ''' Either put in tmin, tmax, dt, to make equitemporal bins 
            which wont be equistatistic, or put in equistatistc bins 
            as (t_edges, t_centres). 
        '''
        if tmin is not None:
            new_time = np.arange(tmin, tmax, dt)
        elif equi_t is not None:
            new_time = equi_t[0]

        rb_s0 = np.zeros((len(self.q), len(new_time) - 1))
        rb_s2 = np.zeros((len(self.q), len(new_time) - 1))
        rb_az = np.zeros((len(self.q), len(new_time) - 1))

        # Brainnfarttts
        for t in range(1, len(new_time)):
            mask = (self.time < new_time[t]) & (self.time >= new_time[t - 1])
            rb_s0[:, t - 1] = np.nanmean(self.atd_s0[:, mask], axis=1)
            rb_s2[:, t - 1] = np.nanmean(self.atd_s2[:, mask], axis=1)
            rb_az[:, t - 1] = np.nanmean(self.atd_az[:, mask], axis=1)
        self.rb_s0 = rb_s0
        self.rb_s2 = rb_s2
        self.rb_az = rb_az
        self.rb_time = equi_t[1]


    def scan_polarization(self, offset_range=[0, 360, 50], qrange=[0.5, 3], timesteps='all'):
        ''' Which offset gives the strongest S2? 
            Scan the polarization, plot offset vs absolute S2 in qrange '''
        offsets = np.linspace(offset_range[0], 
                                offset_range[1], 
                                offset_range[2])
        sum_abs_s2 = np.zeros(len(offsets))
        old_offset = np.copy(self.offset)
        self.qmin = qrange[0]
        self.qmax = qrange[1]
        self.sp_timesteps = timesteps
        if timesteps == 'All':
            self.sp_timesteps = None 

        for o, offset in enumerate(offsets):
            self.offset = offset
            self.decompose_signal()
            if timesteps == 'all':                
                sum_abs_s2[o] = np.sum(self.atd_s2)
            else:
                sum_abs_s2[o] = np.sum(self.atd_s2[:, timesteps])
        
        return offsets, sum_abs_s2
    
    
    def scan_polarization_pl(self, offset_range=[0, 360, 50], qrange=[0.5, 3], timesteps='allt0'):
        offsets = np.linspace(offset_range[0],offset_range[1],offset_range[2])
        sum_abs_s2 = np.zeros(len(offsets))
        qmin = qrange[0]
        qmax = qrange[1]
        if timesteps== 'allt0':
            tmin = 0
            tmax = 10
        else:
            print("Warning not programmed")
            return
        
        q_index_1 = np.where(self.q > qmin)[0][0]
        q_index_2 = np.where(self.q < qmax)[0][-1]
        
        t_index_1 = np.where(np.asarray(self.time) > tmin)[0][0]
        t_index_2 = np.where(np.asarray(self.time) < tmax )[0][-1]
        
        Data = self.Sphi[t_index_1:t_index_2,:, q_index_1:q_index_2]
        print(np.shape(Data), np.shape(self.q[q_index_1:q_index_2]), np.shape(self.time[t_index_1:t_index_2]))
        print(np.shape(self.Sphi), np.shape(self.q), np.shape(self.time))
        
        #Averaging time
        print(np.shape(np.nanmean(Data,0)))

    def tsr_skl(self, x, y):
        ''' Robust Slope Estimate for 1D data'''
        estimator = TheilSenRegressor()
        estimator.fit(x, y)
        return estimator.coef_, estimator.intercept_

    def tsr_man(self, x, y):
        ''' Manual Theil Sen, to give out the stat variable
            as in matlab - which then isn't used anywhere. 
            Does not give exactly the same as either the sklern version OR the 
            matlab version. The problem is in the np.diff vs matlabs diff. 
            
            Not sure how much it matters. '''
        [N, c] = x.shape
        comb = np.asarray(list(itertools.combinations(range(N), 2)))
        ml_comb = np.flip(comb)[:,[1, 0]]  # for ML consistency. Maybe can remove later
        delta_y = np.diff(y[ml_comb]).ravel()
        delta_x = np.diff(x[:, 0][ml_comb]).ravel()  # need to get rid of extra dim again first
        theil_m = delta_y / delta_x
        a = np.median(theil_m)
        bs =  y - a * x
        b = np.median(bs)
        #stat = [np.mean(np.abs(theil_m - np.mean(theil_m))), 
        #        np.mean(np.abs(b - np.mean(b)))]  # ML uses mean here, not median. but not used any further
        #self.stat = stat 
        return a, b

    def theil_sen_stats(self, x, y):
        ''' Theil-sen regression returning slope, lower and upper confidence of slope,
        intercept, and lower and upper confidence of the same.'''
        #handle things containing nan with a masked array:
        my = np.ma.masked_array(y, mask=np.isnan(y))
        if (len(my.mask)-sum(my.mask))<3: #if we have less than 3 points not nan, spit out nothing but nan
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        m, b,lowerm, upperm = ss.mstats.theilslopes(my, x)
        #intercept from median(y) - medslope*median(x)
        #still need error bar on intercepts.

        #new way: standard error of mean for points minus fitted line (spread around the line basically.)
        #95% bands via standard error (x+-1.96*SE)
        nanfilt = ~np.isnan(y)
        st_error = np.std(y[nanfilt] - (m * x[nanfilt] + b))/np.sqrt(len(x[nanfilt]))

        upperb = b + st_error * 1.96
        lowerb = b - st_error * 1.96
        return m,  b 

    def reg_siegelslopes(self, x, y):
        ''' Siegel slope robust regression '''
        res = ss.siegelslopes(y, x)
        return res[0], res[1] 

    def reg_ransac(self, x, y):
        ''' RANSAC robust regression '''
        ransac = RANSACRegressor()
        ransac.fit(x, y)
        return ransac.estimator_.coef_, ransac.estimator_.intercept_

    def reg_huber(self, x, y):
        ''' Huber robust regression '''
        reg = HuberRegressor()
        reg.fit(x, y)
        return reg.coef_, reg.intercept_

    def reg_ridge(self, x, y):
        ''' Ridge robust regression '''
        reg = Ridge(alpha=0.0, random_state=0, normalize=True)
        reg.fit(x, y)
        return reg.coef_, reg.intercept_

    def plot_lineseries(self, data, ax=None, ncol=2, smoof=None, shifted=False, sgolay=None):
        from scipy.signal import medfilt
        from scipy.ndimage import gaussian_filter1d as g1d
        from scipy.signal import savgol_filter as sg

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        colors = self.make_colors(data.shape[1]) 
        shift_step = 0

        if self.rb_time != None:
            time = self.rb_time
        if data.shape[1] == len(self.time):
            time = self.time

        if shifted:
            shift_step = np.nanmax(np.abs(data)) / 10
        for s, signal in enumerate(data.T):
            if shifted:
                ax.plot(self.q, np.zeros(self.q.shape) + shift_step * s, 'k--', alpha=0.25)
            if smoof is not None:
                ax.plot(self.q, g1d(medfilt(signal, smoof), smoof) + shift_step * s, color=colors[s], label=f'{time[s] : 2.2f} ps')
            elif sgolay != None:
                ax.plot(self.q, sg(signal, sgolay[0], sgolay[1]) + shift_step * s, color=colors[s], label=f'{time[s] : 2.2f} ps')
            else:
                ax.plot(self.q, signal + shift_step * s, color=colors[s], label=f'{time[s] * 1e12 : 2.2f} ps')
            ax.set_ylabel('S(Q)')
            ax.set_xlabel('Q (Å$^{-1}$)')
            if shifted:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax.legend(loc='best', ncol=ncol)
        return ax

    def plot_2d(self, t, x, ys, fig=None, sub='111', cb=True, logscan=False, title=None):
        '''Plot ys[N,M] on time [M] vs q [N] axes. 
           Stolen from K. Ledbetter. '''
        if fig is None:
            fig, ax = plt.subplots()               
        if logscan: 
            my_xticks = np.unique(t)
            t=np.arange(len(t))
            my_xticks=['%.4G'%n for n in my_xticks] #make them readable
            fig.xticks(t, my_xticks,rotation=45)
        ts,xs=np.meshgrid(t,x)
        try:
            plt.pcolormesh(ts,xs,ys, shading='auto')
        except TypeError:
            plt.pcolormesh(ts,xs,ys.T, shading='auto')
        #median above and below 0 * some number
        
        #plt.clim([np.nanmin(ys),np.nanmax(ys)])
        plt.clim([np.nanmin(ys)*0.05,np.nanmax(ys)*0.05])
        
        if cb:
            cbar=plt.colorbar()
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
        plt.xlabel('t (s)')
        plt.ylabel('Q ($\AA^{-1}$)')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if title:
            plt.title(title, fontweight='bold')
        return plt.gca(), cbar


    def phillip_crange(self, data, qidx=None, limit_m=5):
        qq = list(range(len(self.q)))
        if qidx is not None:
            qq = qidx
        mad = ss.median_absolute_deviation
        clims = [np.median(data[qq, :]) - limit_m * mad(data[qq,:], axis=None), 
                 np.median(data[qq, :]) + limit_m * mad(data[qq,:], axis=None)]
        return clims

    def plot_svd(self, qs, ts, data, n=5, shift_fac=0.01, smooth=None, showplot=True, 
                 fig=None, sub=None,logscan=False):
        '''Do an SVD of data and plot it against q and t axes. 
            n=num of SV to show 
            smooth=list [x,y] of # of values to median filter by.

            Returns n spectral components, SVs, and time traces.
            
            Also stolen from Elisa/Kathryn. They are the best! '''

        if smooth is not None:
            if len(data.shape) != len(smooth):
                raise ValueError('smooth must be a list with length = ' +
                                 'dimensions of data, use 1 for axes you' +
                                 ' do not wish to smooth.')
            data=medfilt(data,smooth)

        #SVD part
        U, S, V = np.linalg.svd(data, full_matrices=True)

        # These were reversed.. But would break the plotting later on
        # This seems consistent with the matlab results.
        spectraces = U[:, 0:n]
        timetraces = V[0:n, :]

        vals=S[0:n]
        if showplot:
            if fig is None:
                fig = plt.figure()
                fig.set_size_inches(9, 4)
                #set up for using subplots(a,b,c+i)
                a, b, c = 1, 3, 1
            else:
                plt.figure(fig)
                #assume sub is 1st of a line, make into (a,b,c)
                if isinstance(sub, str):
                    sub=tuple(sub) #format (a,b,c)
                (a, b, c)=sub
                a, b, c = int(a), int(b), int(c)
            #plot S in pretty color points
            Sx = np.arange(1, n + 1)
            plt.subplot(a, b, c)
            for i in range(n):
                plt.plot(Sx[i], S[i], 'o')
            plt.xlabel('SV')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            ''' Plot time traces'''
            plt.subplot(a, b, c + 1)
            if logscan:
                my_xticks = np.unique(ts)
                ts=np.arange(len(ts))
                my_xticks=['%.4g'%n for n in my_xticks] #make them readable
                plt.xticks(ts, my_xticks,rotation=45)

            shift = np.ones((timetraces.T.shape))
            for x in range(n):
                shift[:, x] *= (x + 1) * shift_fac

            plt.plot(ts, timetraces.T * vals + shift)
            plt.xlabel('t (s)')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.title('      S$_{i,i} \cdot $U$_i$ (Typogram)')

            ''' Plot Q traces'''
            shift = np.ones((spectraces.shape))
            for x in range(n):
                shift[:, x] *= (x + 1) * shift_fac

            plt.subplot(a, b, c + 2)
            plt.plot(qs,spectraces * vals + shift)
            plt.xlabel('Q (Å$^{-1}$)')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.title('        S$_{i,i} \cdot $V$_i$ (Chronogram)')

        return spectraces, vals, timetraces

    def beamcenter_check(self, qrange=[0.5, 2.4], ax=None):
        self.qmin = qrange[0]
        self.qmax = qrange[1]
        qmask = self.make_qmask()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        colors = self.make_colors(self.cake.shape[0])

        for c, slice in enumerate(self.cake):
            ax.plot(self.q[qmask], slice[qmask] / max(slice[qmask]), 
                    color=colors[c])

        ax.set_xlim( [qrange[0], qrange[1]] )
        ax.set_ylim([0, 1.2])
        ax.set_ylabel('S(Q)')
        ax.set_xlabel('Q (Å$^{-1}$)')    

        return ax

    def make_colors(self, n):
        return [cm.viridis(x/n) for x in range(n)]
    
    def copy(self):
        return copy.deepcopy(self)






 # Philipp Code

def stack_scans(runlist, experiment ="xcslv9618", data_path = '/gpfs/exfel/u/scratch/FXE/202201/p002787/Diffs/', output_path ='/gpfs/exfel/u/scratch/FXE/202201/p002787/Diffs/', save=True):
    scans = []
    t_full = 0
    for i in range(len(runlist)):
        full_file_name = data_path +  'Run' + str(runlist[i]) + '_Reduced.mat'
        scans.append(Scan(full_file_name, energy=18))
        t_full += len(scans[-1].time)    
    q = scans[0].q
    
    AzMatrixFull = np.zeros((len(q), t_full))
    S0MatrixFull = np.zeros((len(q), t_full))
    S2MatrixFull = np.zeros((len(q), t_full))
    
    t_index = 0
    t_axis = np.asarray([])
    for entry in scans:
        AzMatrixFull[:,t_index:t_index+len(entry.time)] = entry.Sazi
        S0MatrixFull[:,t_index:t_index+len(entry.time)] = entry.S0
        S2MatrixFull[:,t_index:t_index+len(entry.time)] = entry.S2
        t_index  += len(entry.time)
        t_axis = np.append(t_axis, entry.time)
        #print(entry.keys())
    
    # sorting
    t_axis_sorted = np.sort(t_axis)
    t_axis_stacked = []
    
    desired_t_axis_normalizer = min(len(runlist), 5) #how many bins are to be stacked together
    
    az_matrix_stacked= np.zeros((len(q), int(np.ceil(t_full/desired_t_axis_normalizer))  ))
    s0_matrix_stacked = np.zeros((len(q), int(np.ceil(t_full/desired_t_axis_normalizer))  ))
    s2_matrix_stacked = np.zeros((len(q), int(np.ceil(t_full/desired_t_axis_normalizer))  ))
    for i in range(int(t_full/desired_t_axis_normalizer)):
        for a in t_axis_sorted[i*desired_t_axis_normalizer:(i+1)*desired_t_axis_normalizer]:
            index = np.where(t_axis == a)[0][0]
            az_matrix_stacked[:,i]+= AzMatrixFull[:,index]/desired_t_axis_normalizer
            s0_matrix_stacked[:,i]+= S0MatrixFull[:,index]/desired_t_axis_normalizer
            s2_matrix_stacked[:,i]+= S2MatrixFull[:,index]/desired_t_axis_normalizer        
        
          
        t_axis_stacked.append(np.mean(t_axis_sorted[i*desired_t_axis_normalizer:(i+1)*desired_t_axis_normalizer]))
        
        #
        if i == int(t_full/desired_t_axis_normalizer)-1 and len(t_axis_sorted[(i+1)*desired_t_axis_normalizer:] > 0):
            for a in t_axis_sorted[(i+1)*desired_t_axis_normalizer:]:
                index = np.where(t_axis == a)[0][0]
                az_matrix_stacked[:,i+1] +=AzMatrixFull[:,index]/len(t_axis_sorted[(i+1)*desired_t_axis_normalizer:])
                s0_matrix_stacked[:,i+1]+= S0MatrixFull[:,index]/len(t_axis_sorted[(i+1)*desired_t_axis_normalizer:])
                s2_matrix_stacked[:,i+1]+= S2MatrixFull[:,index]/len(t_axis_sorted[(i+1)*desired_t_axis_normalizer:])
            t_axis_stacked.append(np.mean(t_axis_sorted[(i+1)*desired_t_axis_normalizer:]))

            
    name_tag = ""
    for entry in runlist:
        name_tag+= '_'
        name_tag+= '{:04d}'.format(entry)          
            
    save_dict = {
        'q': q,
        'scanvar' : t_axis_stacked,
        'S0' : s0_matrix_stacked,
        'S2' : s2_matrix_stacked,
        'Sazi' : az_matrix_stacked,
        'RunNumber': name_tag[1:]
            }
    if save:
        name_tag += '.mat'
        print('../DTUReduction/Stacked_Runs' + name_tag)
        savemat(output_path + 'Stacked_Runs' + name_tag ,save_dict)
    
    return az_matrix_stacked,s0_matrix_stacked,s2_matrix_stacked, t_axis_stacked, q
