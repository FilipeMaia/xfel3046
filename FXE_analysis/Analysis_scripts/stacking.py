"""
@author: Philipp Lenzen
"""

import numpy as np
import h5py
from calculating_diff_2D import decompose_S0_S2_Az

def create_bins(range_set, bin_limit = 50, t_0_offset = 2250, minimal_bin_time = 0.05):
    '''
        creates time-bins to bin the stacked data into

        Input:
            range_set : list of scans that should be processed
                e.g. [101,103,104]
            bin_limit : int, how many trains have to be measured at a given position to not discard the time bin
                e.g. 50

    '''
    total_delay_list = np.asarray([])
    for i in range_set :
        try:
            #print("/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/ChiRunRAW_Corr_full_"+ str(i)+ "_testing.h5")
            with h5py.File("/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/ChiRunRAW_Corr_full_"+ str(i)+ "_testing.h5","r") as f:
                total_delay_list = [*total_delay_list, *f['DelayRun']]
                #total_delay_list.append(np.asarray(f['DelayRun']))
        except:
            print("error: could not read time from: ", i)
            pass
    print("total delays:", len(total_delay_list))
    reduced_bins_1 = create_time_bins(np.unique(-np.asarray(total_delay_list)+t_0_offset), short_delay_min=minimal_bin_time)
    counter_list = get_counter(reduced_bins_1, range_set,t_0_offset)
    
    poor_bins = np.where(np.asarray(counter_list) < bin_limit )[0]
    final_bins = []
    for i in range(len(reduced_bins_1)):
        if i not in poor_bins:
            final_bins.append(reduced_bins_1[i])
            
    final_counter =  get_counter(final_bins, range_set,t_0_offset)
    print("bins created: ", len(reduced_bins_1), ", bins rejected: ", len(poor_bins), ", final bins: ", len(final_bins))
    return final_bins, final_counter
    
def get_counter(bins_to_pass, range_set, t_0_offset):
    '''


    '''
    counter_list = []
    for a in range(len(bins_to_pass)):
        binned_delay = bins_to_pass[a]
        #print("Starting bin ", a, "binned value: ", binned_delay)
        counter = 0
        #counter += 1
        for i in range_set :
            with h5py.File("/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/ChiRunRAW_Corr_full_"+ str(i)+ "_testing.h5","r") as f:
                indexes_smaller = np.where(-np.asarray(f['DelayRun'])+t_0_offset >= bins_to_pass[a])[0]
                try:
                    indexes_larger = np.where(-np.asarray(f['DelayRun'])+t_0_offset < bins_to_pass[a+1])[0]
                except:
                    indexes_larger = [len(np.asarray(f['DelayRun']))-1]
                list1_as_set = set(indexes_smaller)
                intersection = list1_as_set.intersection(indexes_larger)
                intersection_as_list = list(intersection)
                counter += len(intersection_as_list)
        counter_list.append(counter)
    return counter_list 


def create_time_bins(list_of_vars, short_delay_min = 0.05, medium_delay_min = 1, long_delay_min= 10):
    ''' create minimum 30 fs size bins '''
    unique_vars = np.unique(list_of_vars)
    new_time_bin = [list_of_vars[0]]    
    for i in range(len(unique_vars)):
        if unique_vars[i] < 10 and unique_vars[i] > -1:
            if unique_vars[i] > new_time_bin[-1]+short_delay_min:
                new_time_bin.append(unique_vars[i])
        elif abs(unique_vars[i]) < 40:
            if unique_vars[i] > new_time_bin[-1]+medium_delay_min:
                new_time_bin.append(unique_vars[i])                
        else:
            if unique_vars[i] > new_time_bin[-1]+long_delay_min:
                new_time_bin.append(unique_vars[i])
    return new_time_bin


def generate_meta_parameters(scan_no, norm_range, mode = "1D"):
    '''
        reads out q axis, N_pulses and the q, norm_range
    '''
    with h5py.File("/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/ChiRunRAW_Corr_full_"+ str(scan_no)+ "_testing.h5","r") as f:
        if mode == "1D":
            N_pulses = int(np.shape(f['Sq_sa0'])[1]/len(np.asarray(f['DelayRun'])))
        elif mode =="2D":
            swapped = np.swapaxes(data['Sq_sa0'], 1,2)
            N_pulses = int(np.shape(f['Sq_sa0'])[2]/len(np.asarray(f['DelayRun'])))
        q = np.asarray(f['q']).T[0]
        q1 = np.where(q >norm_range[0])[0][0]
        q2 = np.where(q <norm_range[1])[0][-1]
    return q, q1, q2,N_pulses

def bin_data_into_stack(bins_to_use, range_set, final_counter, t_0_offset = 2250,norm_range = [0.5,3.5], mode = "pulse"):
    '''
        Does the stacking
    '''
    
    q, q1, q2,N_pulses = generate_meta_parameters(range_set[0],norm_range)

    final_data_matrix = np.empty((len(bins_to_use),len(q)))
    for i in range(len(bins_to_use)): 
        print("starting bin ", i ," out of ", len(bins_to_use))
        Az_prep = np.empty((max(final_counter)*N_pulses//2*2,len(q)))
        counter = 0
        for scan_no in range_set:
            with h5py.File("/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/ChiRunRAW_Corr_full_"+ str(scan_no)+ "_testing.h5","r") as f:
                #print("opening scan ", scan_no)
                swapped = np.asarray(f['Sq_sa0'])
                indexes_smaller = np.where(-np.asarray(f['DelayRun'])+t_0_offset >= bins_to_use[i])[0]
                try:
                    indexes_larger = np.where(-np.asarray(f['DelayRun'])+t_0_offset < bins_to_use[i+1])[0]
                except:
                    indexes_larger = [len(np.asarray(f['DelayRun']))-1]
                list1_as_set = set(indexes_smaller)
                intersection = list1_as_set.intersection(indexes_larger)
                intersection_as_list = list(intersection)
                #print("indexes here: ", len(intersection_as_list))
                small_counter = 0
                if mode == "pulse":
                    for index in intersection_as_list:
                        for ii in range(N_pulses):
                        #for ii in [2]:
                            if ii % 2 == 0:
                                S_on = swapped[:,index*N_pulses + ii]
                                S_on = S_on / np.nanmean(S_on[q1:q2])
                                if ii == 0:
                                    S_off = swapped[:,index*N_pulses + ii+1]
                                    S_off = S_off/ np.nanmean(S_off [q1:q2])
                                else:
                                    S_off_1 = swapped[:,index*N_pulses + ii+1]
                                    S_off_2 = swapped[:,index*N_pulses + ii-1]                    
                                    S_off = S_off_1/ np.nanmean(S_off_1 [q1:q2]) * 0.5 + S_off_2/ np.nanmean(S_off_2[q1:q2]) * 0.5                
                                Az_prep[counter,:] = S_on - S_off
                                counter +=1
                elif mode == "train":
                    print("len(intersection_as_list): ", len(intersection_as_list))
                    indexes_even = np.asarray(intersection_as_list)[np.asarray(intersection_as_list)%2==0]
                    for index in indexes_even:
                        S_on = swapped[:,index*N_pulses:(index+1)*N_pulses]
                        if index == 0:
                            S_off = swapped[:,(index+1)*N_pulses:(index+2)*N_pulses]
                        else:
                            S_off = swapped[:,(index-1)*N_pulses:(index)*N_pulses] * 0.5
                            try:
                                S_off += swapped[:,(index+1)*N_pulses:(index+2)*N_pulses]* 0.5
                            except:
                                S_off = S_off*2
                        S_on = np.nanmean(S_on,axis=1)
                        S_on = S_on / np.nanmean(S_on[q1:q2])
                        S_off = np.nanmean(S_off,axis=1)
                        S_off = S_off / np.nanmean(S_off[q1:q2])

                        #print(np.shape(S_off))
                        Az_prep[counter,:] = S_on - S_off
                        counter +=1

                    print("len iseven: ", len(indexes_even))

        Az_prep[Az_prep==0] = np.nan
        final_data_matrix[i,:] = np.nanmedian(Az_prep,axis=0)
    return final_data_matrix,q

def bin_data_into_stack_2D(bins_to_use, range_set,final_counter, t_0_offset = 2250,norm_range = [0.5,3.5],offset_angle=67,returnImage = False):
    '''
        Does the stacking
    '''
    
    q, q1, q2,N_pulses = generate_meta_parameters(range_set[0],norm_range)

    final_data_matrix = np.empty((len(bins_to_use),len(q), 17))
    for i in range(len(bins_to_use)-1): 
        print("starting bin ", i ," out of ", len(bins_to_use))
        #Az_prep = np.empty((max(final_counter)*N_pulses//2*2,len(q),17))
        Az_prep = np.empty((2500*N_pulses//2*2,len(q),17))
        counter = 0
        for scan_no in range_set:
            with h5py.File("/gpfs/exfel/u/scratch/FXE/202201/p002787/Reduced/ChiRunRAW_Corr_full_"+ str(scan_no)+ "_testing_2D.h5","r") as f:
                #print("opening scan ", scan_no)
                swapped = np.swapaxes(f['Sq_sa0'], 1,2)
                indexes_smaller = np.where(-np.asarray(f['DelayRun'])+t_0_offset >= bins_to_use[i])[0]
                #print("current time bin: ", bins_to_use[i],", next time bin: ", bins_to_use[i+1])

                #print("delays available: ", -np.asarray(f['DelayRun'])+t_0_offset)
                try:
                    indexes_larger = np.where(-np.asarray(f['DelayRun'])+t_0_offset < bins_to_use[i+1])[0]
                except:
                    indexes_larger = [len(np.asarray(f['DelayRun']))-1]

                #print("indexes small: ", indexes_smaller, " indexes_larger: ", indexes_larger)
                list1_as_set = set(indexes_smaller)
                intersection = list1_as_set.intersection(indexes_larger)
                intersection_as_list = list(intersection)
                print("indexes here: ", len(intersection_as_list))
                small_counter = 0
                for index in intersection_as_list:
                    for ii in range(N_pulses):
                        if ii % 2 == 0:
                            S_on = swapped[:,:,index*N_pulses + ii]
                            S_on = S_on / np.nanmean(S_on[q1:q2,:])
                            #print(S_on[:,q1:q2],q1,q2,np.shape(S_on))
                            if ii == 0:
                                S_off = swapped[:,:,index*N_pulses + ii+1]
                                S_off = S_off/ np.nanmean(S_off [q1:q2,:])
                            else:
                                S_off_1 = swapped[:,:,index*N_pulses + ii+1]
                                S_off_2 = swapped[:,:,index*N_pulses + ii-1]                    
                                S_off = S_off_1/ np.nanmean(S_off_1 [q1:q2,:]) * 0.5 + S_off_2/ np.nanmean(S_off_2[q1:q2,:]) * 0.5                
                            if counter < 2500:
                                Az_prep[counter,:,:] = S_on - S_off
                            #print(np.nanmean(swapped[:,:,index*N_pulses + ii]), np.nanmean(S_on))
                            counter +=1
        Az_prep[Az_prep==0] = np.nan
        #print(np.shape(np.nanmedian(Az_prep,axis=0)), np.shape(final_data_matrix[i,:,:]))
        final_data_matrix[i,:,:] = np.nanmedian(Az_prep,axis=0)

    if returnImage:
        return final_data_matrix
    decomposed= np.empty((len(bins_to_use),3,len(q)))
    phi_angles = np.deg2rad(np.linspace(0,360,18)[:-1]+90 +offset_angle)
    for i in range(len(bins_to_use)):
        decomposed[i,:] = decompose_S0_S2_Az(final_data_matrix[i,:], phi_angles,q).T
    return decomposed,q



