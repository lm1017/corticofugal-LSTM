import numpy as np
import scipy
import scipy.stats as stats
import all_models
import matplotlib.pyplot as plt
import statsmodels.api as sm
from util_funcs import get_stim_response_grid_concat, calc_CC_norm, extents
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor
from scipy.signal import butter, filtfilt
from scipy import stats
from plot_funcs import fake_parula
import random
import torch
from torch.autograd import grad

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def corr_with_stims(cc_type, averaging_type, n_neurons, all_stim_dataset, resp_dataset, stimuli, zero_padding, padding_bins, 
                    n_F, folds):
    '''
    Parameters
    ----------
    cc_type : type of correlation coefficient -> Pearson or Spearman
    averaging_type : type of averaging -> either take CC of activity with each stimulus frequency channel and average all 
    CCs, or sum stimulus over frequency channels and take CC of activity with this
    n_neurons : number of neurons in dataset
    all_stim_dataset : whole set of stimuli (dimensions -> (f, t, stim))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    stimuli : numbers of stimuli in stim_dataset, in order
    padding_bins : number of bins used for zero-padding, if true
    zero_padding : boolean variable, indicates if zero-padding has been used
    n_F : number of frequency channels in stimuli
    folds : number of cross-validation folds

    Returns
    -------
    ccs_stim_resp : negative CCs between stimuli and neural activity (selected stimuli and response sets), for each neuron
    ccs_stim_resp_p_value : significance (True/False) of CCs between stimuli and neural activity
    '''
    
    if not zero_padding: # if zero-padding was not used, set number of padding bins to 0
        padding_bins = 0
    
    ccs_stim_resp = np.zeros((n_neurons)) # array to hold CCs between stimuli and neural activity
    ccs_stim_resp_p_value = np.zeros((n_neurons)) # array to hold p-value of CCs between stimuli and neural activity
    
    resp_size = int(resp_dataset.size(-1)/len(stimuli[0])) # length (in time bins) of response to one stimulus
    
    stim_size = np.size(all_stim_dataset, 1)-padding_bins # length (in time bins) of one stimulus
    # array to hold stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    stim_grid = np.zeros((n_F, len(stimuli)*len(stimuli[0])*stim_size))
    
    for fID in range(folds): # loop over folds
        for sID in range(len(stimuli[fID])): # loop over stimuli in each fold
            # insert stimulus in grid
            stim = all_stim_dataset[:, padding_bins:, stimuli[fID][sID]]
            stim_grid[:, (2*fID + sID)*stim_size:(2*fID + sID + 1)*stim_size] = stim
     
    for nID in range(n_neurons): # loop over neurons - the correlation is measured for each neuron
        # array to hold responses from all folds, concatenated in time (dimensions -> (concatenated t))
        resp_grid = np.zeros((len(stimuli)*len(stimuli[0])*resp_size))
        
        for fID in range(folds): # loop over folds
            for sID in range(len(stimuli[fID])): # loop over stimuli in fold
                # insert response in grid
                resp = resp_dataset[nID, fID, sID*resp_size:(sID + 1)*resp_size].detach().cpu().numpy()
                resp_grid[(2*fID + sID)*resp_size:(2*fID + sID + 1)*resp_size] = resp
        
        # Pearson product-moment correlation coefficient
        if cc_type == 'pearson':
            if averaging_type == 'average_stim': # correlation of responses with frequency-averaged stimulus
                # significance is computed for 'less' alternative as we are looking at negative correlations
                resp_pearson_r = stats.pearsonr(np.mean(stim_grid, 0), resp_grid, alternative='less')
                
                ccs_stim_resp[nID] = -resp_pearson_r[0] # negative CC
                ccs_stim_resp_p_value[nID] = resp_pearson_r[1]
            
            elif averaging_type == 'average_CC': # average correlation of responses with each frequency channel of stimulus
                ccs_stim_resp_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    # significance is computed for 'less' alternative as we are looking at negative correlations
                    ccs_stim_resp_sum += stats.pearsonr(stim_grid[f_chan], resp_grid, alternative='less')[0]
                
                # divide by number of frequency channels
                ccs_stim_resp[nID] = -ccs_stim_resp_sum/n_F # negative CC
                ccs_stim_resp_p_value[nID] = np.nan # p-value of CC is set to NaN if average_CC method is used
        
        # Spearman rank-order correlation coefficient
        elif cc_type == 'spearman':
            if averaging_type == 'average_stim': # correlation of responses with frequency-averaged stimulus
                # significance is computed for 'less' alternative as we are looking at negative correlations
                resp_spearman_r = stats.spearmanr(np.mean(stim_grid, 0), resp_grid, alternative='less')
                
                ccs_stim_resp[nID] = -resp_spearman_r[0] # negative CC
                ccs_stim_resp_p_value[nID] = resp_spearman_r[1]
            
            elif averaging_type == 'average_CC': # average correlation of responses with each frequency channel of stimulus
                ccs_stim_resp_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    # significance is computed for 'less' alternative as we are looking at negative correlations
                    ccs_stim_resp_sum += stats.spearmanr(stim_grid[f_chan], resp_grid, alternative='less')[0]
                
                # divide by number of frequency channels
                ccs_stim_resp[nID] = -ccs_stim_resp_sum/n_F # negative CC
                ccs_stim_resp_p_value[nID] = np.nan # p-value of CC is set to NaN if average_CC method is used
    
    # p-values transformed into Boolean variables, where True indicates significance
    ccs_stim_resp_p_value = ccs_stim_resp_p_value < 0.01
    
    return [ccs_stim_resp, ccs_stim_resp_p_value]

#############################################################################################################################

def corr_with_stims_silence(cc_type, averaging_type, n_neurons, all_stim_dataset, resp_dataset, stimuli, models_fits_dict, 
                            zero_padding, padding_bins, n_F, folds):
    '''
    Parameters
    ----------
    cc_type : type of correlation coefficient -> Pearson or Spearman
    averaging_type : type of averaging -> either take CC of activity with each stimulus frequency channel and average all 
    CCs, or sum stimulus over frequency channels and take CC of activity with this
    n_neurons : number of neurons in dataset
    all_stim_dataset : whole set of stimuli (dimensions -> (f, t, stim))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    stimuli : numbers of stimuli in stim_dataset, in order
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    padding_bins : number of bins used for zero-padding, if true
    zero_padding : boolean variable, indicates if zero-padding has been used
    n_F : number of frequency channels in stimuli
    folds : number of cross-validation folds

    Returns
    -------
    ccs_stim_resp : CCs between silence-triggered stimuli and neural activity (selected stimuli and response sets), for each 
    neuron
    ccs_stim_resp_p_value : significance (True/False) of CCs between silence-triggered stimuli and neural activity
    '''
    
    if not zero_padding: # if zero-padding was not used, set number of padding bins to 0
        padding_bins = 0
    
    ccs_stim_resp = np.zeros((n_neurons)) # array to hold CCs between silence-triggered stimuli and neural activity
    # array to hold p-values of CCs between silence-triggered stimuli and neural activity
    ccs_stim_resp_p_value = np.zeros((n_neurons))
    
    resp_size = int(resp_dataset.size(-1)/len(stimuli[0])) # length (in time bins) of response to one stimulus
    stim_size = np.size(all_stim_dataset, 1)-padding_bins # length (in time bins) of one stimulus
    
    # array to hold silence-triggered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    silence_tr_stim_grid = np.zeros((n_F, len(stimuli)*len(stimuli[0])*stim_size))

    for fID in range(folds): # loop over folds
        for sID in range(len(stimuli[fID])): # loop over stimuli in each fold
            stim = all_stim_dataset[:, padding_bins:, stimuli[fID][sID]]
            
            silence_tr_stim = np.zeros_like(stim) # array to hold silence-triggered stimulus
            
            for f_chan in range(n_F): # loop over frequency channels
                # z-score each stimulus frequency channel - negative log-power does not necessarily mean silence
                stim_chan_zscored = (stim[f_chan] - np.mean(stim[f_chan]))/np.std(stim[f_chan], ddof=1)
            
                negative_counter = 0 # set counter since last positive time bin to 0
                for t_bin in range(np.size(stim, 1)): # loop over time bins
                    if stim_chan_zscored[t_bin] >= 0: # if stimulus is >= 0, silence-triggered stimulus is set to 0
                        silence_tr_stim[f_chan, t_bin] = 0
                        negative_counter = 0 # counter remains at/is reset to 0 
                    else:
                        # if stimulus is < 0, counter is updated and silence-triggered stimulus is set to counter
                        negative_counter += 1
                        silence_tr_stim[f_chan, t_bin] = negative_counter
            
            # uncomment to plot average silence-triggered stimulus, and some example silence-triggered channels
            '''bin_size = models_fits_dict[1]['bin_size']
            # time axis, wrong axis, replace with next line if available
            t_axis = np.linspace(0, bin_size*(stim_size+1), stim_size)
            t_axis = models_fits_dict[1]['time_axis']
            
            plt.figure()
            plt.plot(t_axis, np.mean(silence_tr_stim, 0))
            
            plt.figure()
            plt.plot(t_axis, silence_tr_stim[0])
            plt.figure()
            plt.plot(t_axis, silence_tr_stim[1])
            plt.figure()
            plt.plot(t_axis, silence_tr_stim[14])
            plt.figure()
            plt.plot(t_axis, silence_tr_stim[15])'''
            
            # insert silence-triggered stimulus in grid
            silence_tr_stim_grid[:, (2*fID + sID)*stim_size:(2*fID + sID + 1)*stim_size] = silence_tr_stim
     
    for nID in range(n_neurons): # loop over neurons - the correlation is measured for each neuron
        # array to hold responses from all folds, concatenated in time (dimensions -> (concatenated t))
        resp_grid = np.zeros((len(stimuli)*len(stimuli[0])*resp_size))
        
        for fID in range(folds): # loop over folds
            for sID in range(len(stimuli[fID])): # loop over stimuli in fold
                # insert response in grid
                resp = resp_dataset[nID, fID, sID*resp_size:(sID + 1)*resp_size].detach().cpu().numpy()
                resp_grid[(2*fID + sID)*resp_size:(2*fID + sID + 1)*resp_size] = resp
        
        # Pearson product-moment correlation coefficient
        if cc_type == 'pearson':
            # correlation of responses with frequency-averaged silence-triggered stimulus
            if averaging_type == 'average_stim':
                resp_pearson_r = stats.pearsonr(np.mean(silence_tr_stim_grid, 0), resp_grid, alternative='greater')
                
                ccs_stim_resp[nID] = resp_pearson_r[0]
                ccs_stim_resp_p_value[nID] = resp_pearson_r[1]
                
                # uncomment to plot scatter plots of PSTH firing rate against frequency-averaged silence-triggered stimulus
                # level
                '''if resp_pearson_r[0] > 0.3:
                    plt.figure()
                    plt.scatter(np.mean(silence_tr_stim_grid, 0), resp_grid)
                    plt.xlabel('time points since sound')
                    plt.ylabel('firing rate')
                    plt.title('Neuron ' + str(nID))'''
            
            # average correlation of responses with each frequency channel of silence-triggered stimulus
            elif averaging_type == 'average_CC':
                # uncomment to use maximum absolute correlation over frequency channels
                '''ccs_stim_resp_by_chan = np.zeros((n_F))
                ccs_stim_resp_by_chan_p_values = np.zeros((n_F))
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_stim_resp_by_chan[f_chan] = stats.pearsonr(silence_tr_stim_grid[f_chan], resp_grid, 
                                                                   alternative='greater')[0]
                    ccs_stim_resp_by_chan_p_values[f_chan] = stats.pearsonr(silence_tr_stim_grid[f_chan], resp_grid, 
                                                                            alternative='greater')[1]
                
                ccs_stim_resp[nID] = np.nanmax(np.abs(ccs_stim_resp_by_chan))
                ccs_stim_resp_p_value[nID] = ccs_stim_resp_sum_p_values[np.nanargmax(np.abs(ccs_stim_resp_by_chan))]'''
                
                # uncomment to use correlation averaged over frequency channels
                ccs_stim_resp_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_stim_resp_sum += stats.pearsonr(silence_tr_stim_grid[f_chan], resp_grid, alternative='greater')[0]
                
                # divide by frequency channels
                ccs_stim_resp[nID] = ccs_stim_resp_sum/n_F
                ccs_stim_resp_p_value[nID] = np.nan # p-value of CC is set to NaN if average_CC method is used
            
        # Spearman rank-order correlation coefficient
        elif cc_type == 'spearman':
            # correlation of responses with frequency-averaged silence-triggered stimulus
            if averaging_type == 'average_stim':
                resp_spearman_r = stats.spearmanr(np.mean(silence_tr_stim_grid, 0), resp_grid, alternative='greater')
                
                ccs_stim_resp[nID] = resp_spearman_r[0]
                ccs_stim_resp_p_value[nID] = resp_spearman_r[1]
            
            # average correlation of responses with each frequency channel of silence-triggered stimulus
            elif averaging_type == 'average_CC':
                ccs_stim_resp_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_stim_resp_sum += stats.spearmanr(silence_tr_stim_grid[f_chan], resp_grid, alternative='greater')[0]
                
                # divide by frequency channels
                ccs_stim_resp[nID] = ccs_stim_resp_sum/n_F
                ccs_stim_resp_p_value[nID] = np.nan # p-value of CC is set to NaN if average_CC method is used
    
    # p-values transformed into Boolean variables, where True indicates significance
    ccs_stim_resp_p_value = ccs_stim_resp_p_value < 0.01
    
    return [ccs_stim_resp, ccs_stim_resp_p_value]

#############################################################################################################################

def corr_with_stims_silence_by_stim(cc_type, averaging_type, n, n_neurons, all_stim_dataset, resp_dataset, stimuli, 
                                    zero_padding, padding_bins, n_F, folds):
    '''
    Parameters
    ----------
    cc_type : type of correlation coefficient -> Pearson or Spearman
    averaging_type : type of averaging -> either take CC of activity with each stimulus frequency channel and average all 
    CCs, or sum stimulus over frequency channels and take CC of activity with this
    n : number of sounds to use for SDRM calculation (best n sounds based on SDRM measured separately)
    n_neurons : number of neurons in dataset
    all_stim_dataset : whole set of stimuli (dimensions -> (f, t, stim))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    stimuli : numbers of stimuli in stim_dataset, in order
    padding_bins : number of bins used for zero-padding, if true
    zero_padding : boolean variable, indicates if zero-padding has been used
    n_F : number of frequency channels in stimuli
    folds : number of cross-validation folds

    Returns
    -------
    ccs_stim_resp : CCs between silence-triggered stimuli and neural activity (selected stimuli and response sets), for each 
    neuron
    ccs_stim_resp_p_value : significance (True/False) of CCs between silence-triggered stimuli and neural activity
    '''
    
    if not zero_padding: # if zero-padding was not used, set number of padding bins to 0
        padding_bins = 0
    
    # array to hold CCs between silence-triggered stimuli and neural activity
    ccs_stim_resp_by_stim = np.zeros((n_neurons, np.size(all_stim_dataset, 2)))
    
    resp_size = int(resp_dataset.size(-1)/len(stimuli[0])) # length (in time bins) of response to one stimulus
    stim_size = np.size(all_stim_dataset, 1)-padding_bins # length (in time bins) of one stimulus
    
    # array to hold silence-triggered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    silence_tr_stim_grid = np.zeros_like(all_stim_dataset)
    silence_tr_stim_grid = silence_tr_stim_grid[:, padding_bins:]

    for fID in range(folds): # loop over folds
        for sID in range(len(stimuli[fID])): # loop over stimuli in each fold
            stim = all_stim_dataset[:, padding_bins:, stimuli[fID][sID]]
            
            silence_tr_stim = np.zeros_like(stim) # array to hold silence-triggered stimulus
            
            for f_chan in range(n_F): # loop over frequency channels
                # z-score each stimulus frequency channel - negative log-power does not necessarily mean silence
                stim_chan_zscored = (stim[f_chan] - np.mean(stim[f_chan]))/np.std(stim[f_chan], ddof=1)
            
                negative_counter = 0 # set counter since last positive time bin to 0
                for t_bin in range(np.size(stim, 1)): # loop over time bins
                    if stim_chan_zscored[t_bin] >= 0: # if stimulus is >= 0, silence-triggered stimulus is set to 0
                        silence_tr_stim[f_chan, t_bin] = 0
                        negative_counter = 0 # counter remains at/is reset to 0 
                    else:
                        # if stimulus is < 0, counter is updated and silence-triggered stimulus is set to counter
                        negative_counter += 1
                        silence_tr_stim[f_chan, t_bin] = negative_counter
            
            # insert silence-triggered stimulus in grid
            silence_tr_stim_grid[:, :, sID] = silence_tr_stim
     
    for nID in range(n_neurons): # loop over neurons - the correlation is measured for each neuron
        # array to hold responses from all folds, concatenated in time (dimensions -> (concatenated t))
        resp_grid = np.zeros((resp_size, np.size(all_stim_dataset, 2)))
        
        for fID in range(folds): # loop over folds
            for sID in range(len(stimuli[fID])): # loop over stimuli in fold
                # insert response in grid
                resp = resp_dataset[nID, fID, sID*resp_size:(sID + 1)*resp_size].detach().cpu().numpy()
                resp_grid[:, sID] = resp
        
        # Pearson product-moment correlation coefficient
        if cc_type == 'pearson':
            # correlation of responses with frequency-averaged silence-triggered stimulus
            if averaging_type == 'average_stim':
                for sID in range(len(stimuli[fID])):
                    resp_pearson_r = stats.pearsonr(np.mean(silence_tr_stim_grid[:, :, sID], 0), resp_grid[:, sID], 
                                                    alternative='greater')
                    ccs_stim_resp_by_stim[nID, sID] = resp_pearson_r[0]
            
            # average correlation of responses with each frequency channel of silence-triggered stimulus
            elif averaging_type == 'average_CC':
                for sID in range(len(stimuli[fID])):
                    ccs_stim_resp_sum = 0
                    
                    for f_chan in range(n_F): # loop over frequency channels
                        # sum over frequency channels
                        ccs_stim_resp_sum += stats.pearsonr(silence_tr_stim_grid[f_chan, :, sID], resp_grid[:, sID], 
                                                            alternative='greater')[0]
                    
                    # divide by frequency channels
                    ccs_stim_resp_by_stim[nID, sID] = ccs_stim_resp_sum/n_F
            
        # Spearman rank-order correlation coefficient
        elif cc_type == 'spearman':
            # correlation of responses with frequency-averaged silence-triggered stimulus
            if averaging_type == 'average_stim':
                for sID in range(len(stimuli[fID])):
                    resp_spearman_r = stats.spearmanr(np.mean(silence_tr_stim_grid[:, :, sID], 0), resp_grid[:, sID], 
                                                      alternative='greater')
                    
                    ccs_stim_resp_by_stim[nID, sID] = resp_spearman_r[0]
            
            # average correlation of responses with each frequency channel of silence-triggered stimulus
            elif averaging_type == 'average_CC':
                for sID in range(len(stimuli[fID])):
                    ccs_stim_resp_sum = 0
                    
                    for f_chan in range(n_F): # loop over frequency channels
                        # sum over frequency channels
                        ccs_stim_resp_sum += stats.spearmanr(silence_tr_stim_grid[f_chan, :, sID], resp_grid[:, sID], 
                                                             alternative='greater')[0]
                    
                    # divide by frequency channels
                    ccs_stim_resp_by_stim[nID, sID] = ccs_stim_resp_sum/n_F
    
    ########## PICK TOP N SOUNDS BASED ON THEIR RESPONSES TO SILENCE AND REPEAT ANALYSIS
    # array to hold CCs between silence-triggered stimuli and neural activity, with concatenated stimuli
    ccs_stim_resp = np.zeros((n_neurons))
    # array to hold p-values of CCs between silence-triggered stimuli and neural activity
    ccs_stim_resp_p_value = np.zeros((n_neurons))

    # 4 stimuli which produce the highest SDRM averaged across neurons
    top_SDRM = np.argpartition(np.nanmean(ccs_stim_resp_by_stim, 0), -n)[-n:]
    
    # array to hold silence-triggered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    silence_tr_stim_grid = np.zeros((n_F, len(top_SDRM)*stim_size))

    for sID in range(len(top_SDRM)): # loop over stimuli in each fold
        stim = all_stim_dataset[:, padding_bins:, stimuli[fID][top_SDRM[sID]]]
        
        silence_tr_stim = np.zeros_like(stim) # array to hold silence-triggered stimulus
        
        for f_chan in range(n_F): # loop over frequency channels
            # z-score each stimulus frequency channel - negative log-power does not necessarily mean silence
            stim_chan_zscored = (stim[f_chan] - np.mean(stim[f_chan]))/np.std(stim[f_chan], ddof=1)
        
            negative_counter = 0 # set counter since last positive time bin to 0
            for t_bin in range(np.size(stim, 1)): # loop over time bins
                if stim_chan_zscored[t_bin] >= 0: # if stimulus is >= 0, silence-triggered stimulus is set to 0
                    silence_tr_stim[f_chan, t_bin] = 0
                    negative_counter = 0 # counter remains at/is reset to 0 
                else:
                    # if stimulus is < 0, counter is updated and silence-triggered stimulus is set to counter
                    negative_counter += 1
                    silence_tr_stim[f_chan, t_bin] = negative_counter
        
        # insert silence-triggered stimulus in grid
        silence_tr_stim_grid[:, sID*stim_size:(sID + 1)*stim_size] = silence_tr_stim
     
    for nID in range(n_neurons): # loop over neurons - the correlation is measured for each neuron
        # array to hold responses from all folds, concatenated in time (dimensions -> (concatenated t))
        resp_grid = np.zeros((len(top_SDRM)*resp_size))
        
        for sID in range(len(top_SDRM)): # loop over stimuli in fold
            # insert response in grid
            resp = resp_dataset[nID, fID, top_SDRM[sID]*resp_size:(top_SDRM[sID] + 1)*resp_size].detach().cpu().numpy()
            resp_grid[sID*resp_size:(sID + 1)*resp_size] = resp
    
        # Pearson product-moment correlation coefficient
        if cc_type == 'pearson':
            # correlation of responses with frequency-averaged silence-triggered stimulus
            if averaging_type == 'average_stim':
                resp_pearson_r = stats.pearsonr(np.mean(silence_tr_stim_grid, 0), resp_grid, alternative='greater')
                
                ccs_stim_resp[nID] = resp_pearson_r[0]
                ccs_stim_resp_p_value[nID] = resp_pearson_r[1]
            
            # average correlation of responses with each frequency channel of silence-triggered stimulus
            elif averaging_type == 'average_CC':
                ccs_stim_resp_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_stim_resp_sum += stats.pearsonr(silence_tr_stim_grid[f_chan], resp_grid, alternative='greater')[0]
                
                # divide by frequency channels
                ccs_stim_resp[nID] = ccs_stim_resp_sum/n_F
                ccs_stim_resp_p_value[nID] = np.nan # p-value of CC is set to NaN if average_CC method is used
            
        # Spearman rank-order correlation coefficient
        elif cc_type == 'spearman':
            # correlation of responses with frequency-averaged silence-triggered stimulus
            if averaging_type == 'average_stim':
                resp_spearman_r = stats.spearmanr(np.mean(silence_tr_stim_grid, 0), resp_grid, alternative='greater')
                
                ccs_stim_resp[nID] = resp_spearman_r[0]
                ccs_stim_resp_p_value[nID] = resp_spearman_r[1]
            
            # average correlation of responses with each frequency channel of silence-triggered stimulus
            elif averaging_type == 'average_CC':
                ccs_stim_resp_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_stim_resp_sum += stats.spearmanr(silence_tr_stim_grid[f_chan], resp_grid, alternative='greater')[0]
                
                # divide by frequency channels
                ccs_stim_resp[nID] = ccs_stim_resp_sum/n_F
                ccs_stim_resp_p_value[nID] = np.nan # p-value of CC is set to NaN if average_CC method is used
    
    # p-values transformed into Boolean variables, where True indicates significance
    ccs_stim_resp_p_value = ccs_stim_resp_p_value < 0.01
    
    return [ccs_stim_resp, ccs_stim_resp_p_value], top_SDRM

#############################################################################################################################

def corr_with_stims_sound(cc_type, averaging_type, n_neurons, all_stim_dataset, resp_dataset, stimuli, zero_padding, 
                          padding_bins, n_F, folds):
    '''
    Parameters
    ----------
    cc_type : type of correlation coefficient -> Pearson or Spearman
    averaging_type : type of averaging -> either take CC of activity with each stimulus frequency channel and average all 
    CCs, or sum stimulus over frequency channels and take CC of activity with this
    n_neurons : number of neurons in dataset
    all_stim_dataset : whole set of stimuli (dimensions -> (f, t, stim))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    stimuli : numbers of stimuli in stim_dataset, in order (dimensions -> (fold, stimuli))
    padding_bins : number of bins used for zero-padding, if true
    zero_padding : boolean variable, indicates if zero-padding has been used
    n_F : number of frequency channels in stimuli

    Returns
    -------
    ccs_stim_resp : CCs between sound-triggered stimuli and neural activity (selected stimuli and response sets), for each 
    neuron
    ccs_stim_resp_p_value : significance (True/False) of CCs between sound-triggered stimuli and neural activity
    '''
    
    if not zero_padding: # if zero-padding was not used, set number of padding bins to 0
        padding_bins = 0
    
    ccs_stim_resp = np.zeros((n_neurons)) # array to hold CCs between sound-triggered stimuli and neural activity
    # array to hold p-values of CCs between sound-triggered stimuli and neural activity
    ccs_stim_resp_p_value = np.zeros((n_neurons))
    
    resp_size = int(resp_dataset.size(-1)/len(stimuli[0])) # length (in time bins) of response to one stimulus
    stim_size = np.size(all_stim_dataset, 1)-padding_bins # length (in time bins) of one stimulus
    
    # array to hold sound-triggered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    sound_tr_stim_grid = np.zeros((n_F, len(stimuli)*len(stimuli[0])*stim_size))
    
    for fID in range(folds): # loop over folds
        for sID in range(len(stimuli[fID])): # loop over stimuli in each fold
            stim = all_stim_dataset[:, padding_bins:, stimuli[fID][sID]]
            
            sound_tr_stim = np.zeros_like(stim) # array to hold sound-triggered stimulus
            
            for f_chan in range(n_F): # loop over frequency channels
                stim_chan_zscored = (stim[f_chan] - np.mean(stim[f_chan]))/np.std(stim[f_chan], ddof=1)
            
                positive_counter = 0 # set counter since last positive time bin to 0
                for t_bin in range(np.size(stim, 1)): # loop over time bins
                    if stim_chan_zscored[t_bin] <= 0: # if stimulus is <= 0, sound-triggered stimulus is set to 0
                        sound_tr_stim[f_chan, t_bin] = 0
                        positive_counter = 0 # counter remains at/is reset to 0 
                    else:
                        # if stimulus is > 0, counter is updated and sound-triggered stimulus is set to counter
                        positive_counter += 1
                        sound_tr_stim[f_chan, t_bin] = positive_counter
            
            # insert sound-triggered stimulus in grid
            sound_tr_stim_grid[:, (2*fID + sID)*stim_size:(2*fID + sID + 1)*stim_size] = sound_tr_stim
     
    for nID in range(n_neurons): # loop over neurons - the correlation is measured for each neuron
        # array to hold responses from all folds, concatenated in time (dimensions -> (concatenated t))
        resp_grid = np.zeros((len(stimuli)*len(stimuli[0])*resp_size))
        
        for fID in range(folds): # loop over folds
            for sID in range(len(stimuli[fID])): # loop over stimuli in fold
                # insert response in grid
                resp = resp_dataset[nID, fID, sID*resp_size:(sID + 1)*resp_size].detach().cpu().numpy()
                resp_grid[(2*fID + sID)*resp_size:(2*fID + sID + 1)*resp_size] = resp
        
        # Pearson product-moment correlation coefficient
        if cc_type == 'pearson':
            if averaging_type == 'average_stim': # correlation of responses with frequency-averaged sound-triggered stimulus
                resp_pearson_r = stats.pearsonr(np.mean(sound_tr_stim_grid, 0), resp_grid, alternative='greater')
                
                ccs_stim_resp[nID] = resp_pearson_r[0]
                ccs_stim_resp_p_value[nID] = resp_pearson_r[1]
            
            # average correlation of responses with each frequency channel of sound-triggered stimulus
            elif averaging_type == 'average_CC':
                ccs_stim_resp_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_stim_resp_sum += stats.pearsonr(sound_tr_stim_grid[f_chan], resp_grid, alternative='greater')[0]
                
                # divide by frequency channels
                ccs_stim_resp[nID] = ccs_stim_resp_sum/n_F
            
        # Spearman rank-order correlation coefficient
        elif cc_type == 'spearman':
            if averaging_type == 'average_stim': # correlation of responses with frequency-averaged sound-triggered stimulus
                resp_spearman_r = stats.spearmanr(np.mean(sound_tr_stim_grid, 0), resp_grid, alternative='greater')
                
                ccs_stim_resp[nID] = resp_spearman_r[0]
                ccs_stim_resp_p_value[nID] = resp_spearman_r[1]
           
            # average correlation of responses with each frequency channel of sound-triggered stimulus
            elif averaging_type == 'average_CC':
                ccs_stim_resp_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_stim_resp_sum += stats.spearmanr(sound_tr_stim_grid[f_chan], resp_grid, alternative='greater')[0]
                
                # divide by frequency channels
                ccs_stim_resp[nID] = ccs_stim_resp_sum/n_F
    
    # p-values transformed into Boolean variables, where True indicates significance
    ccs_stim_resp_p_value = ccs_stim_resp_p_value < 0.01

    return [ccs_stim_resp, ccs_stim_resp_p_value]

#############################################################################################################################

def stims_revcorr(n_neurons, stim_dataset, resp_dataset, stimuli, models_fits_dict, model_ID, folds):
    '''
    Parameters
    ----------
    n_neurons : number of neurons in dataset
    stim_dataset : selected set of stimuli (dimensions -> (fold, f, history, concatenated t))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    stimuli : numbers of stimuli in stim_dataset, in order
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    model_ID : name of model being analysed
    folds : number of cross-validation folds

    Returns
    -------
    fold_gate_kernels : list of reverse correlation kernels of increases and decreases in gating variables on stimuli in 
    stim_dataset (dimensions -> (folds, lambdas, gates, hidden units)); if model is not gated, the list is empty
    '''
    
    fold_gate_kernels = [] # list to hold reverse correlation kernels for each fold, by lambda
    
    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params']

    for fID in range(folds): # loop over folds
        lamb_gate_kernels = [] # list to hold reverse correlation for each lambda, by hidden unit
        
        for lamb in range(len(lambda_sequence)): # loop over lambdas
            model = getattr(all_models, model_ID)(model_params) # initialise model
            model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
            
            # NumPy traces of model internal variables during forward pass of stimuli
            traces = model.np_forward_loop(stim_dataset[fID])

            # append kernels by gate and hidden unit
            lamb_gate_kernels.append(traces_revcorr(stim_dataset[fID], traces))
        
        fold_gate_kernels.append(lamb_gate_kernels)
        
    return fold_gate_kernels

def traces_revcorr(inputs, traces):
    '''
    Parameters
    ----------
    inputs : set of stimuli for one fold
    traces : model internal variables over time
    
    Returns
    -------
    kernels : dictionary of reverse correlation kernels of gating variables on stimuli in inputs (dimensions -> (gates, 
    hidden units)); each entry contains two kernels: one for gating variable increases and one for gating variable decreases
    '''
    
    kernels = {} # dictionary to hold reverse correlation kernels for each gate, by hidden unit
    
    inputs = inputs.detach().cpu().numpy() # convert torch inputs to NumPy array
    
    gates_list = ['r', 'z', 'i', 'f', 'out'] # list of gates: r and z are from GRU, i, f, and out are from LSTM
    
    for key in traces: # loop over traces
        if key in gates_list: # check whether trace is that of a gating variable
            gate_trace = traces[key]
            gate_trace_diff = np.diff(gate_trace, axis=-1) # derivative of gating variable trace
            
            # clip first element from input so that dimensions agree with gating variable derivative
            input_trace = inputs[:, :, 1:]
            
            kernels[key] = [] # list to hold reverse correlation kernels for each gate
            
            for unit in range(np.size(gate_trace, 0)): # loop over hidden units
                gate_increase = np.nonzero(gate_trace_diff[unit] > 0)[0] # time points at which gating variable increases
                gate_decrease = np.nonzero(gate_trace_diff[unit] < 0)[0] # time points at which gating variable decreases
                
                # gating increase- and gating decrease-triggered average of stimulus cochleagram
                gate_i_kernel = np.sum(input_trace[:, :, gate_increase], axis=-1)/len(gate_increase)
                gate_d_kernel = np.sum(input_trace[:, :, gate_decrease], axis=-1)/len(gate_decrease)
                
                kernel_dict = {'gate_i': gate_i_kernel, 'gate_d': gate_d_kernel}
                kernels[key].append(kernel_dict)
        
    return kernels

#############################################################################################################################

def effective_HUs(n_neurons, stim_dataset, models_fits_dict, model_ID, folds):
    '''
    Parameters
    ----------
    n_neurons : number of neurons in dataset
    stim_dataset : selected set of stimuli (dimensions -> (fold, f, history, concatenated t))
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    model_ID : name of model being analysed
    folds : number of cross-validation folds
    model_type : convolutional or nonconvolutional

    Returns
    -------
    eff_HU_list : list of effective hidden units for a given model architecture (dimensions -> (folds, lambdas, neurons))
    '''
    
    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    model_type = models_fits_dict[1]['model_type']
    if model_type == 'not_conv':
        hidden_size = model_params['hidden_size'] # network hidden size
    elif model_type == 'conv':
        hidden_size = np.size(models_fits_dict[0][0][0][0]['w_l1'], 0)
    output_size = model_params['output_size'] # network output size
    
    # array to hold variance of hidden units (over time) weighted by hidden-to-output weight for each neuron
    HU_var = np.zeros((n_neurons, folds, len(lambda_sequence), hidden_size))
    
    eff_HU_list = [] # list to hold all effective HUs for a model type, by fold, lambda, and neuron
        
    for fID in range(folds): # loop over folds
        eff_HU_list_fold = [] # list to hold all effective HUs for a fold, by lambda and neuron
        
        for lamb in range(len(lambda_sequence)): # loop over lambdas
            model = getattr(all_models, model_ID)(model_params) # initialise model
            model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
            
            # NumPy traces of model internal variables during forward pass of stimuli
            traces = model.np_forward_loop(stim_dataset[fID])
            
            eff_HU_list_neuron = [] # list to hold all effective HUs for a lambda, by neuron
            
            for nID in range(n_neurons): # loop over neurons
                w_model = models_fits_dict[0][fID][lamb][0] # fitted model weights
                
                for key in w_model: # loop over sets of fitted weights and biases
                    # since hidden-to-output weights have different keys depending on the architecure, identify them by:
                    # choosing parameter set with two dimensions (weights as opposed to biases) and choosing parameter set
                    # whose first dimension is equal to the output size
                    if np.size(w_model[key], 0) == output_size and len(np.shape(w_model[key])) == 2:
                        w_to_neuron = w_model[key][nID]
                
                # weigh HU output by its weight to output neuron, at each time point
                if len(np.shape(traces['h'])) == 2:
                    weighted_HU_act = np.multiply(np.transpose(traces['h']), w_to_neuron)
                elif len(np.shape(traces['h'])) == 3:
                    traces_h_transpose = np.transpose(traces['h'], (0, 2, 1))
                    traces_h_reshaped = np.reshape(traces_h_transpose, (-1, np.size(traces_h_transpose, 2)))
                    weighted_HU_act = np.multiply(traces_h_reshaped, w_to_neuron)
                weighted_HU_var = np.var(weighted_HU_act, 0) # variance of HU weighted output over time
                HU_var[nID, fID, lamb] = weighted_HU_var

                # effective HUs chosen as those whose variance is higher than the average HU variance
                effective_HUs = np.where(weighted_HU_var >= np.sum(weighted_HU_var)/hidden_size)[0]
                
                # effective HUs chosen as the top 5 with highest variance
                #effective_HUs = np.argsort(weighted_HU_var)[-5:]
                
                eff_HU_list_neuron.append(effective_HUs)
            
            eff_HU_list_fold.append(eff_HU_list_neuron)
        
        eff_HU_list.append(eff_HU_list_fold)
        
    return eff_HU_list

#############################################################################################################################

def corr_with_HUs(h_or_c, n_neurons, stim_dataset, resp_dataset, models_fits_dict, model_ID, folds):
    '''
    Parameters
    ----------
    n_neurons : number of neurons in dataset
    stim_dataset : selected set of stimuli (dimensions -> (fold, f, history, concatenated t))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    model_ID : name of model being analysed
    folds : number of cross-validation folds

    Returns
    -------
    corr_HUs_dict : dictionary of correlations between model hidden units and a) pre-activation model output, b) model
    output, c) recorded PSTH; for each neuron and each fold and lambda (dimensions -> (neurons, folds, lambdas, HUs))
    '''
    
    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    model_type = models_fits_dict[1]['model_type']
    if model_type == 'not_conv':
        hidden_size = model_params['hidden_size'] # network hidden size
    elif model_type == 'conv':
        hidden_size = np.size(models_fits_dict[0][0][0][0]['w_l1'], 0)
    
    # array to hold correlations between model HUs and pre-activation model output
    corr_out_pre_act_HU = np.zeros((n_neurons, folds, len(lambda_sequence), hidden_size))
    # array to hold correlations between model HUs and model output
    corr_out_HU = np.zeros((n_neurons, folds, len(lambda_sequence), hidden_size))
    # array to hold correlations between model HUs and recorded PSTH
    corr_resp_HU = np.zeros((n_neurons, folds, len(lambda_sequence), hidden_size))
        
    for fID in range(folds): # loop over folds
        for lamb in range(len(lambda_sequence)): # loop over lambdas
            model = getattr(all_models, model_ID)(model_params) # initialise model
            model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
            
            # NumPy traces of model internal variables during forward pass of stimuli
            traces = model.np_forward_loop(stim_dataset[fID])
            
            if h_or_c == 'h':
                h_traces = traces['h'] # HU traces
            elif h_or_c == 'c':
                h_traces = traces['c']
            
            if model_type == 'conv':
                h_traces = np.transpose(h_traces, (0, 2, 1))
                h_traces = np.reshape(h_traces, (-1, np.size(h_traces, 2)))
                h_traces = np.transpose(h_traces)
            
            for nID in range(n_neurons): # loop over neurons
                if model_type == 'conv':
                    n_output_pre_act_trace = traces['o_pre_act'][:, nID].flatten() # pre-activation output traces
                    n_output_trace = traces['o'][:, nID].flatten() # output traces
                    
                elif model_type == 'not_conv':
                    n_output_pre_act_trace = traces['o_pre_act'][nID] # pre-activation output traces
                    n_output_trace = traces['o'][nID] # output traces
                
                resp = resp_dataset[nID, fID].detach().cpu().numpy() # recorded PSTH

                # correlations: first row of correlation matrix is the one of interest
                corr_out_pre_act_HU[nID, fID, lamb] = np.corrcoef(n_output_pre_act_trace, h_traces)[0, 1:]
                corr_out_HU[nID, fID, lamb] = np.corrcoef(n_output_trace, h_traces)[0, 1:]
                corr_resp_HU[nID, fID, lamb] = np.corrcoef(resp, h_traces)[0, 1:]
    
    corr_HUs_dict = {'out_pre_act': corr_out_pre_act_HU, 'out': corr_out_HU, 'actual': corr_resp_HU}

    return corr_HUs_dict

#############################################################################################################################

def corr_with_gates(CC_type, n_neurons, stim_dataset, resp_dataset, all_stim_dataset, spike_times, stimuli, models_fits_dict, 
                    models_data_dict, model_ID, folds, fID):
    '''
    Parameters
    ----------
    CC_type : type of correlation coefficient - either 'CCraw' or 'CCnorm'
    n_neurons : number of neurons in dataset
    stim_dataset : selected set of stimuli (dimensions -> (fold, f, history, concatenated t))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    all_stim_dataset : cochleagrams of all multi-repeat stimuli (dimensions -> (frequency, t, stimulus))
    spike_times : spike times of each neuron, in response to each repeat of each mulit-repeat stimulus
    stimuli : indices of stimuli contained in stim_dataset
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    models_built_data : dictionary containing built datasets for each model
    model_ID : name of model being analysed
    folds : number of cross-validation folds
    fID : active fold, not useful for NS3 and NS3_PEG, and not useful for test set analyses for any dataset

    Returns
    -------
    corr_resp_gates : list of correlations between model gates and recorded PSTH; for each neuron and each fold and lambda 
    (dimensions -> (folds, lambdas, gate types, neurons, HUs))
    '''

    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    hidden_size = model_params['hidden_size'] # network hidden size
    
    meta_info = models_fits_dict[-1]
    
    t_lims = meta_info['time_axis']
    clip_bins = meta_info['clip_bins']
    shift_bins = meta_info['shift_bins']
    n_h = meta_info['n_h']
    padding_bins = meta_info['padding_bins']
    norm = meta_info['norm']
    zero_padding = True
    
    mean_cv = models_data_dict[model_ID][3][0]
    std_cv = models_data_dict[model_ID][3][0]
    
    multi_rep_resps = []
    for nID in range(n_neurons):
        data = [spike_times[nID][stim] for stim in stimuli[fID]]
        _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, meta_info['bin_size'], 
                                                       'val', norm, mean_cv, std_cv, clip_bins, shift_bins, zero_padding,
                                                       padding_bins, t_lims)
        multi_rep_resps.append(actual_reps)

    corr_resp_gates = [] # list to hold all correlations
    
    gates_list = ['r', 'z', 'i', 'f', 'out']
    gated_models_list = ['pop_MGU', 'pop_GRU', 'pop_LSTM', 'pop_subLSTM', 'pop_f_LSTM']
    
    if model_ID in gated_models_list:
        for fID in range(folds): # loop over folds
            fold_corr_resp_gates = [] # list to hold correlations for fold
        
            for lamb in range(len(lambda_sequence)): # loop over lambdas
                print('lamb = ' + str(lamb))
                lamb_corr_resp_gates = {} # dictionary to hold correlations for lambda
            
                model = getattr(all_models, model_ID)(model_params) # initialise model
                model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
                
                # NumPy traces of model internal variables during forward pass of stimuli
                traces = model.np_forward_loop(stim_dataset[fID])
                
                for key in traces: # loop over traces
                    if key in gates_list: # check whether trace is that of a gating variable
                        print(key)
                        gate_traces = traces[key]
                        
                        resps = np.transpose(resp_dataset[:, fID, :].detach().cpu().numpy())
                        gate_traces = np.transpose(gate_traces)

                        if CC_type == 'CCraw':
                            combined = np.hstack((resps, gate_traces))
                            corr_matrix = np.corrcoef(combined, rowvar=False)
                            
                            lamb_corr_resp_gates[key] = corr_matrix[:n_neurons, n_neurons:]

                        elif CC_type == 'CCnorm':
                            corr_matrix = np.zeros((n_neurons, hidden_size))
                            
                            for nID in range(n_neurons):
                                n_resps = multi_rep_resps[nID]
                                
                                for HU_ID in range(hidden_size):
                                    gate_trace = gate_traces[:, HU_ID]
                                    ccnorm, _, _ = calc_CC_norm(n_resps, gate_trace)
                                    corr_matrix[nID, HU_ID] = ccnorm
                            
                            lamb_corr_resp_gates[key] = corr_matrix
                
                fold_corr_resp_gates.append(lamb_corr_resp_gates)
                
            corr_resp_gates.append(fold_corr_resp_gates)

    return corr_resp_gates

#############################################################################################################################

def corr_with_t_gate(CC_type, n_neurons, stim_dataset, resp_dataset, all_stim_dataset, spike_times, stimuli, 
                     models_fits_dict, models_data_dict, model_ID, folds, fID):
    '''
    Parameters
    ----------
    CC_type : type of correlation coefficient - either 'CCraw' or 'CCnorm'
    n_neurons : number of neurons in dataset
    stim_dataset : selected set of stimuli (dimensions -> (fold, f, history, concatenated t))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    all_stim_dataset : cochleagrams of all multi-repeat stimuli (dimensions -> (frequency, t, stimulus))
    spike_times : spike times of each neuron, in response to each repeat of each mulit-repeat stimulus
    stimuli : indices of stimuli contained in stim_dataset
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    models_built_data : dictionary containing built datasets for each model
    model_ID : name of model being analysed
    folds : number of cross-validation folds
    fID : active fold, not useful for NS3 and NS3_PEG, and not useful for test set analyses for any dataset

    Returns
    -------
    corr_resp_gates : list of correlations between model gates and recorded PSTH; for each neuron and each fold and lambda 
    (dimensions -> (folds, lambdas, gate types, neurons, HUs))
    '''

    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    hidden_size = model_params['hidden_size'] # network hidden size
    
    meta_info = models_fits_dict[-1]
    
    t_lims = meta_info['time_axis']
    clip_bins = meta_info['clip_bins']
    shift_bins = meta_info['shift_bins']
    n_h = meta_info['n_h']
    padding_bins = meta_info['padding_bins']
    norm = meta_info['norm']
    zero_padding = True
    
    mean_cv = models_data_dict[model_ID][3][0]
    std_cv = models_data_dict[model_ID][3][0]
    
    multi_rep_resps = []
    for nID in range(n_neurons):
        data = [spike_times[nID][stim] for stim in stimuli[fID]]
        _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, meta_info['bin_size'], 
                                                       'val', norm, mean_cv, std_cv, clip_bins, shift_bins, zero_padding,
                                                       padding_bins, t_lims)
        multi_rep_resps.append(actual_reps)

    corr_resp_gates = [] # list to hold all correlations
    
    gates_list = ['t']
    t_gated_models_list = ['pop_f_LSTM']
    
    if model_ID in t_gated_models_list:
        for fID in range(folds): # loop over folds
            fold_corr_resp_gates = [] # list to hold correlations for fold
        
            for lamb in range(len(lambda_sequence)): # loop over lambdas
                print('lamb = ' + str(lamb))
                lamb_corr_resp_gates = {} # dictionary to hold correlations for lambda
            
                model = getattr(all_models, model_ID)(model_params) # initialise model
                model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
                
                # NumPy traces of model internal variables during forward pass of stimuli
                traces = model.np_forward_loop(stim_dataset[fID])
                for key in traces: # loop over traces
                    if key in gates_list: # check whether trace is that of a gating variable
                        gate_traces = traces[key]
                        
                        resps = np.transpose(resp_dataset[:, fID, :].detach().cpu().numpy())
                        gate_traces = np.transpose(gate_traces)

                        if CC_type == 'CCraw':
                            combined = np.hstack((resps, gate_traces))
                            corr_matrix = np.corrcoef(combined, rowvar=False)
                            
                            lamb_corr_resp_gates[key] = corr_matrix[:n_neurons, n_neurons:]

                        elif CC_type == 'CCnorm':
                            corr_matrix = np.zeros((n_neurons, hidden_size))
                            
                            for nID in range(n_neurons):
                                n_resps = multi_rep_resps[nID]
                                
                                for HU_ID in range(hidden_size):
                                    gate_trace = gate_traces[:, HU_ID]
                                    ccnorm, _, _ = calc_CC_norm(n_resps, gate_trace)
                                    corr_matrix[nID, HU_ID] = ccnorm
                            
                            lamb_corr_resp_gates[key] = corr_matrix
                
                fold_corr_resp_gates.append(lamb_corr_resp_gates)
                
            corr_resp_gates.append(fold_corr_resp_gates)

    return corr_resp_gates

#############################################################################################################################

def weight_sparsity(n_neurons, models_fits_dict, model_ID, folds):
    '''
    Parameters
    ----------
    n_neurons : number of neurons in dataset
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    model_ID : name of model being analysed
    folds : number of cross-validation folds

    Returns
    -------
    sparsity : dictionary of output weight sparsity measures for each neuron/model output unit, as well as their p-values 
    (dimensions -> (neurons, folds, lambdas))
    '''
    
    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    output_size = model_params['output_size'] # network output size
    
    # array to hold sparsity of output weights for each neuron
    sparsity = np.zeros((n_neurons, folds, len(lambda_sequence)))
    # array to hold p-values of sparsity measure
    sparsity_pvalue = np.zeros((n_neurons, folds, len(lambda_sequence)))
        
    for fID in range(folds): # loop over folds
        for lamb in range(len(lambda_sequence)): # loop over lambdas
            model = getattr(all_models, model_ID)(model_params) # initialise model
            model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
            
            w_model = models_fits_dict[0][fID][lamb][0] # fitted model weights
            
            for key in w_model: # loop over sets of fitted weights and biases
                # since hidden-to-output weights have different keys depending on the architecure, identify them by:
                # choosing parameter set with two dimensions (weights as opposed to biases) and choosing parameter set
                # whose first dimension is equal to the output size
                if np.size(w_model[key], 0) == output_size and len(np.shape(w_model[key])) == 2:
                    w_to_neurons = w_model[key]
                    
                    # calculate sparsity as kurtosis
                    sparsity[:, fID, lamb] = stats.kurtosis(w_to_neurons, axis=1)
                    # calculate p-value of kurtosis
                    sparsity_pvalue[:, fID, lamb] = stats.kurtosistest(w_to_neurons, axis=1)[1]
    
    sparsity_dict = {'sparsity': sparsity, 'p_values': sparsity_pvalue}
    
    return sparsity_dict

#############################################################################################################################

def gate_weights(n_neurons, stim_dataset, resp_dataset, models_fits_dict, model_ID, folds, best_lambdas):
    '''
    Parameters
    ----------
    n_neurons : number of neurons in dataset
    stim_dataset : selected set of stimuli (dimensions -> (fold, f, history, concatenated t))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    model_ID : name of model being analysed
    folds : number of cross-validation folds
    best_lambdas: dictionary of best lambda for each neuron, for each model

    Returns
    -------
    '''
    
    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    hidden_size = model_params['hidden_size'] # network hidden size
    output_size = model_params['output_size'] # network output size
    
    if model_ID == 'pop_gatesGRU':
        gates_names = ['r', 'z']
    elif model_ID == 'pop_gatesLSTM' or model_ID == 'pop_gatessubLSTM':
        gates_names = ['i', 'f', 'o']
    elif model_ID == 'pop_gatesfLSTM':
        gates_names = ['i', 'f', 'o', 'c']
    else:
        return
    
    n_gates = len(gates_names)
    
    # array to hold gate weights contribution for each neuron and gate
    gate_weights_percent = np.zeros((n_neurons, folds, len(lambda_sequence), n_gates))
    # array to hold total gate weights contribution for each neuron
    all_gate_weights_percent = np.zeros((n_neurons, folds, len(lambda_sequence)))
    
    for fID in range(folds): # loop over folds
        for lamb in range(len(lambda_sequence)): # loop over lambdas
            model = getattr(all_models, model_ID)(model_params) # initialise model
            model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
            
            w_model = models_fits_dict[0][fID][lamb][0] # fitted model weights
            
            for key in w_model: # loop over sets of fitted weights and biases
                # since hidden-to-output weights have different keys depending on the architecure, identify them by:
                # choosing parameter set with two dimensions (weights as opposed to biases) and choosing parameter set
                # whose first dimension is equal to the output size
                if np.size(w_model[key], 0) == output_size and len(np.shape(w_model[key])) == 2:
                    w_to_neurons = w_model[key]
                    
                    for nID in range(np.size(w_to_neurons, 0)):
                        w_sum = np.sum(np.abs(w_to_neurons[nID]))

                        for gID in range(n_gates):
                            gate_w_sum = np.sum(np.abs(w_to_neurons[nID, (gID+1)*hidden_size:(gID+2)*hidden_size]))
                            gate_weights_percent[nID, fID, lamb, gID] = gate_w_sum/w_sum
                        
                        all_gate_w_sum = np.sum(np.abs(w_to_neurons[nID, hidden_size:(n_gates+1)*hidden_size]))
                        all_gate_weights_percent[nID, fID, lamb] = all_gate_w_sum/w_sum
    
    gate_contribution = np.zeros((n_neurons, n_gates))
    for gID in range(n_gates):
        for nID in range(n_neurons):
            gate_contribution[nID, gID] = gate_weights_percent[nID, 0, best_lambdas[model_ID][nID], gID]
        
        '''plt.figure()
        plt.hist(gate_contribution[:, gID], 50)
        plt.title(gates_names[gID] + ' gate contribution distribution')'''
        
    all_gate_contribution = np.zeros((n_neurons))
    
    for nID in range(n_neurons):
        all_gate_contribution[nID] = all_gate_weights_percent[nID, 0, best_lambdas[model_ID][nID]]
        
    '''plt.figure()
    plt.hist(all_gate_contribution, 50)
    plt.title(model_ID + ' all gates contribution distribution')'''
    
    return gate_contribution, all_gate_contribution

#############################################################################################################################

def any_CC_norms(calc_mode, cc_mode, stim_data, resp_data, all_stim_dataset, spike_times, stimuli_to_use, models_fits_dict, 
                 models_data_dict, stimuli, models_best_lambdas, n_neurons, model_ID, fID):

    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    
    out_lamb_traces = []
    
    for lamb in range(len(lambda_sequence)): # loop over lambdas
        print('lamb ' + str(lamb))
        model = getattr(all_models, model_ID)(model_params) # initialise model
        model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda

        out, _ = model.forward_loop(stim_data[fID], resp_data[:, fID], 'val')
        
        if len(np.shape(out)) == 2:
            out_lamb_traces.append(np.transpose(out.detach().cpu().numpy()))
        elif len(np.shape(out)) == 3:
            out_lamb_traces.append(np.transpose(out.detach().cpu().numpy(), (0, 2, 1)))
    
    meta_info = models_fits_dict[-1]
    
    t_lims = meta_info['time_axis']
    clip_bins = meta_info['clip_bins']
    shift_bins = meta_info['shift_bins']
    n_h = meta_info['n_h']
    padding_bins = meta_info['padding_bins']
    norm = meta_info['norm']
    zero_padding = True
    stim_size = np.size(all_stim_dataset, 1) - padding_bins
    
    mean_cv = models_data_dict[model_ID][3][0]
    std_cv = models_data_dict[model_ID][3][0]
    
    if calc_mode == 'together':
        CCs = np.zeros((n_neurons))
        
        for nID in range(n_neurons):
            data = [spike_times[nID][stim] for stim in stimuli_to_use]
            if cc_mode == 'CC_norm':
                _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, 
                                                               meta_info['bin_size'], 'val', norm, mean_cv, std_cv, 
                                                               clip_bins, shift_bins, zero_padding, padding_bins, t_lims)
            elif cc_mode == 'CC_raw':
                _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, 
                                                               meta_info['bin_size'], 'train', norm, mean_cv, std_cv, 
                                                               clip_bins, shift_bins, zero_padding, padding_bins, t_lims)
    
            nID_bl = models_best_lambdas[model_ID][nID]
    
            # indexing works differently for convolutional and nonconvolutional models
            if len(out_lamb_traces[nID_bl].shape) == 3:
                stims = []
                for sID in range(len(stimuli_to_use)):
                    stims.append(np.where(np.array(stimuli[fID]) == stimuli_to_use[sID])[0][0])
                nID_pred = out_lamb_traces[nID_bl][stims, nID]
                nID_pred = nID_pred.flatten()
                
            elif len(out_lamb_traces[nID_bl].shape) == 2:
                nID_all_preds = out_lamb_traces[nID_bl][nID]
                nID_pred = np.zeros((np.size(actual_reps, 0)))
                for sID in range(len(stimuli_to_use)):
                    stim = np.where(np.array(stimuli[fID]) == stimuli_to_use[sID])[0][0]
                    nID_pred[sID*stim_size:(sID + 1)*stim_size] = \
                        nID_all_preds[stim*stim_size:(stim+1)*stim_size]
            
            if cc_mode == 'CC_norm':
                ccnorm, ccraw, _ = calc_CC_norm(actual_reps, nID_pred)
                CCs[nID] = ccnorm
            elif cc_mode == 'CC_raw':
                ccraw = np.corrcoef(actual_reps, nID_pred)[0, 1]
                CCs[nID] = ccraw
            
    elif calc_mode == 'separate':
        CCs = np.zeros((n_neurons, len(stimuli_to_use)))
        
        for nID in range(n_neurons):
            for stimID in range(len(stimuli_to_use)):
                data = [spike_times[nID][stimuli_to_use[stimID]]]
                if cc_mode == 'CC_norm':
                    _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, 
                                                                   meta_info['bin_size'], 'val', norm, mean_cv, std_cv, 
                                                                   clip_bins, shift_bins, zero_padding, padding_bins, t_lims)
                elif cc_mode == 'CC_raw':
                    _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, 
                                                                   meta_info['bin_size'], 'train', norm, mean_cv, std_cv, 
                                                                   clip_bins, shift_bins, zero_padding, padding_bins, t_lims)
        
                nID_bl = models_best_lambdas[model_ID][nID]

                # indexing works differently for convolutional and nonconvolutional models
                if len(out_lamb_traces[nID_bl].shape) == 3:
                    stimID_idx = np.where(np.array(stimuli[fID]) == stimuli_to_use[stimID])[0][0]
                    nID_pred = out_lamb_traces[nID_bl][stimID_idx, nID]
                    
                elif len(out_lamb_traces[nID_bl].shape) == 2:
                    stimID_idx = np.where(np.array(stimuli[fID]) == stimuli_to_use[stimID])[0][0]
                    
                    nID_pred = out_lamb_traces[nID_bl][nID][stimID_idx*stim_size:(stimID_idx + 1)*stim_size]
                
                if cc_mode == 'CC_norm':
                    ccnorm, ccraw, _ = calc_CC_norm(actual_reps, nID_pred)
                    CCs[nID, stimID] = ccnorm
                elif cc_mode == 'CC_raw':
                    ccraw = np.corrcoef(actual_reps, nID_pred)[0, 1]
                    CCs[nID, stimID] = ccraw
        
    return CCs

#############################################################################################################################

def any_CC_norms_bl_only(calc_mode, cc_mode, stim_data, resp_data, all_stim_dataset, spike_times, stimuli_to_use, 
                         models_fits_dict, models_data_dict, stimuli, models_best_lambdas, n_neurons, model_ID, fID, 
                         shuffle_time, shuffle_stims):

    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    
    unique_best_lambdas = np.unique(models_best_lambdas[model_ID])
    
    out_lamb_traces = []
    
    for lamb in range(len(lambda_sequence)): # loop over lambdas
        if lamb in unique_best_lambdas:
            print('lamb ' + str(lamb))
            
            model = getattr(all_models, model_ID)(model_params) # initialise model
            model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
            
            out, _ = model.forward_loop(stim_data[fID], resp_data[:, fID], 'val')
            
            if len(np.shape(out)) == 2:
                out_lamb_traces.append(np.transpose(out.detach().cpu().numpy()))
            elif len(np.shape(out)) == 3:
                out_lamb_traces.append(np.transpose(out.detach().cpu().numpy(), (0, 2, 1)))
    
    meta_info = models_fits_dict[-1]
    
    t_lims = meta_info['time_axis']
    clip_bins = meta_info['clip_bins']
    shift_bins = meta_info['shift_bins']
    n_h = meta_info['n_h']
    padding_bins = meta_info['padding_bins']
    norm = meta_info['norm']
    zero_padding = True
    stim_size = np.size(all_stim_dataset, 1) - padding_bins
    
    mean_cv = models_data_dict[model_ID][3][0]
    std_cv = models_data_dict[model_ID][3][0]
    
    if calc_mode == 'together':
        CCs = np.zeros((n_neurons))
        
        for nID in range(n_neurons):
            data = [spike_times[nID][stim] for stim in stimuli_to_use]
            if cc_mode == 'CC_norm':
                _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, 
                                                               meta_info['bin_size'], 'val', norm, mean_cv, std_cv, 
                                                               clip_bins, shift_bins, zero_padding, padding_bins, t_lims)
            elif cc_mode == 'CC_raw':
                _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, 
                                                               meta_info['bin_size'], 'train', norm, mean_cv, std_cv, 
                                                               clip_bins, shift_bins, zero_padding, padding_bins, t_lims)
            nID_bl = models_best_lambdas[model_ID][nID]
    
            # indexing works differently for convolutional and nonconvolutional models
            if len(out_lamb_traces[nID_bl].shape) == 3:
                stims = []
                for sID in range(len(stimuli_to_use)):
                    stims.append(np.where(np.array(stimuli[fID]) == stimuli_to_use[sID])[0][0])
                nID_pred = out_lamb_traces[nID_bl][stims, nID]
                nID_pred = nID_pred.flatten()
                
            elif len(out_lamb_traces[nID_bl].shape) == 2:
                nID_all_preds = out_lamb_traces[nID_bl][nID]
                nID_pred = np.zeros((np.size(actual_reps, 0)))
                for sID in range(len(stimuli_to_use)):
                    stim = np.where(np.array(stimuli[fID]) == stimuli_to_use[sID])[0][0]
                    nID_pred[sID*stim_size:(sID + 1)*stim_size] = \
                        nID_all_preds[stim*stim_size:(stim+1)*stim_size]
            
            if cc_mode == 'CC_norm':
                ccnorm, ccraw, _ = calc_CC_norm(actual_reps, nID_pred)
                CCs[nID] = ccnorm
            elif cc_mode == 'CC_raw':
                ccraw = np.corrcoef(actual_reps, nID_pred)[0, 1]
                CCs[nID] = ccraw
            
    elif calc_mode == 'separate':
        CCs = np.zeros((n_neurons, len(stimuli_to_use)))
        
        model_preds = {}
        real_psths = {}
        
        for nID in range(n_neurons):
            nID_model_preds = {}
            nID_real_psths = {}
            
            for stimID in range(len(stimuli_to_use)):
                data = [spike_times[nID][stimuli_to_use[stimID]]]
                if cc_mode == 'CC_norm':
                    _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, 
                                                                   meta_info['bin_size'], 'val', norm, mean_cv, std_cv, 
                                                                   clip_bins, shift_bins, zero_padding, padding_bins, t_lims)
                elif cc_mode == 'CC_raw':
                    _, actual_reps = get_stim_response_grid_concat(data, all_stim_dataset, stimuli, fID, n_h, 
                                                                   meta_info['bin_size'], 'train', norm, mean_cv, std_cv, 
                                                                   clip_bins, shift_bins, zero_padding, padding_bins, t_lims)
                nID_bl = models_best_lambdas[model_ID][nID]
                nID_bl_idx = np.where(unique_best_lambdas == nID_bl)[0][0]

                # indexing works differently for convolutional and nonconvolutional models
                if len(out_lamb_traces[nID_bl_idx].shape) == 3:
                    stimID_idx = np.where(np.array(stimuli[fID]) == stimuli_to_use[stimID])[0][0]
                    nID_pred = out_lamb_traces[nID_bl_idx][stimID_idx, nID]
                    
                elif len(out_lamb_traces[nID_bl_idx].shape) == 2:
                    stimID_idx = np.where(np.array(stimuli[fID]) == stimuli_to_use[stimID])[0][0]
                    
                    nID_pred = out_lamb_traces[nID_bl_idx][nID][stimID_idx*stim_size:(stimID_idx + 1)*stim_size]

                if shuffle_time:
                    np.random.shuffle(actual_reps)
                
                if cc_mode == 'CC_norm':
                    ccnorm, ccraw, _ = calc_CC_norm(actual_reps, nID_pred)
                    CCs[nID, stimID] = ccnorm
                elif cc_mode == 'CC_raw':
                    ccraw = np.corrcoef(actual_reps, nID_pred)[0, 1]
                    CCs[nID, stimID] = ccraw
                    
                nID_model_preds[str(stimID)] = nID_pred
                nID_real_psths[str(stimID)] = actual_reps
                    
            model_preds[str(nID)] = nID_model_preds
            real_psths[str(nID)] = nID_real_psths
        
    return CCs, model_preds, real_psths

#############################################################################################################################

def compute_gradients(calc_mode, stim_data, resp_data, stimuli_to_use, models_fits_dict, models_data_dict, 
                      models_best_lambdas, n_neurons, model_ID, fID, method):

    lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
    model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    
    unique_best_lambdas = np.unique(models_best_lambdas[model_ID])
    
    lamb_gradients = []
    lamb_se = []
    
    for lamb in range(len(lambda_sequence)): # loop over lambdas
        if lamb in unique_best_lambdas:
            print('lamb ' + str(lamb))

            model = getattr(all_models, model_ID)(model_params) # initialise model
            model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
            
            gradients = np.zeros((n_neurons, np.size(stim_data, 1), np.size(stim_data, 4)))
            se = np.zeros((n_neurons, np.size(stim_data, 1), np.size(stim_data, 4)))
            #gradients = []
            
            for sID in range(np.size(stim_data, 1)):
                print('stimulus ' + str(sID))
                
                if method == 1:
                    inputs = torch.transpose(stim_data[fID, sID].reshape(model.input_size, -1), 1, 0)
                    
                    h_n = torch.zeros((1, model.hidden_size), device=device)
                    c_n = torch.zeros((1, model.hidden_size), device=device)
    
                    out, h_n, _ = model.forward(inputs, h_n, c_n, 'val')
                
                    h_n.retain_grad()
                    for t in range(h_n.size(0)):
                        h_n[t, :].retain_grad()
                    
                    for nID in range(n_neurons):
                        for t in range(h_n.size(0)):
                            n_pred = out[t, nID]
                            n_pred.backward(retain_graph=True)
                            if t == 5:
                                import pdb
                                pdb.set_trace()
                            if t == h_n.size(0) - 1:
                                h_grad = h_n.grad
                                import pdb
                                pdb.set_trace()
                                np_h_grad = h_grad.detach().cpu().numpy()
                                mean_h_grad = np.mean(np.square(np_h_grad), 1)
                        
                        for tao in range(h_n.size(0)):
                            tao_grads = []
                            
                            for t in range(tao, h_n.size(0)):
                                tao_grads.append(mean_h_grad[t - tao])
    
                            gradients[nID, sID, tao] = sum(tao_grads)/len(tao_grads)
                            se[nID, sID, tao] = np.nanstd(tao_grads, ddof=1)/np.sqrt(len(tao_grads))
                            
                elif method == 2:
                    # build inputs with explicit batch dim and make a true leaf
                    raw = stim_data[fID, sID].reshape(model.input_size, -1)   # (input_size, seq_len)
                    inputs = raw.T.unsqueeze(1).contiguous()                  # -> (seq_len, 1, input_size)
                    inputs = inputs.clone().detach().requires_grad_(True)     # make a true leaf requiring grad
                    
                    # proper initial states (adjust num_layers if >1)
                    batch = 1
                    num_layers = 1   # change if your model uses >1 layer
                    h_0 = torch.zeros(num_layers, batch, model.hidden_size, device=device)
                    c_0 = torch.zeros(num_layers, batch, model.hidden_size, device=device)
                    
                    out, h_n, c_n = model.forward(inputs, h_0, c_0, 'val')
                    import pdb
                    pdb.set_trace()
                    t_out = 5
                    b = 0
                    h0_test = h_0.clone().detach().requires_grad_(True)
                    out_test, _, _ = model.forward(inputs, h0_test, c_0, 'val')
                    y_test = out_test[t_out, b].view(-1)[0]
                    g_h0 = torch.autograd.grad(y_test, h0_test, retain_graph=False)[0]
                    print("g_h0.shape:", g_h0.shape)
                    print("||∂y/∂h0|| (norm):", g_h0.norm().item())
                    
                    
                    with torch.no_grad():
                        t_gate = torch.sigmoid(model.t_gate_w_x(inputs) + model.t_gate_w_h(h_0))
                    # t_gate shape might be (seq_len, batch, input_size) or (seq_len, batch, 1) depending on implementation
                    print("t_gate shape:", t_gate.shape)
                    print("t_gate (per-timestep mean):", [t_gate[t].abs().mean().item() for t in range(t_gate.shape[0])])
                    print("t_gate (per-timestep min,max):", [(t_gate[t].min().item(), t_gate[t].max().item()) for t in range(t_gate.shape[0])])
                    import pdb
                    pdb.set_trace()
                    
            lamb_gradients.append(gradients)
            lamb_se.append(se)
            
    return lamb_gradients, lamb_se
                            

#############################################################################################################################

def corr_with_ON_OFF_stims(cc_type, averaging_type, n_neurons, all_stim_dataset, resp_dataset, stimuli, models_fits_dict, 
                           zero_padding, padding_bins, n_F, folds):
    '''
    Parameters
    ----------
    cc_type : type of correlation coefficient -> Pearson or Spearman
    averaging_type : type of averaging -> either take CC of activity with each stimulus frequency channel and average all 
    CCs, or sum stimulus over frequency channels and take CC of activity with this
    n_neurons : number of neurons in dataset
    all_stim_dataset : whole set of stimuli (dimensions -> (f, t, stim))
    resp_dataset : selected set of responses (dimensions -> (neuron, fold, concatenated t))
    stimuli : numbers of stimuli in stim_dataset, in order
    models_fits_dict : dictionary containing fitted parameters, meta-info, and loss traces for one model, for each fold and
    lambda
    padding_bins : number of bins used for zero-padding, if true
    zero_padding : boolean variable, indicates if zero-padding has been used
    n_F : number of frequency channels in stimuli
    folds : number of cross-validation folds

    Returns
    -------
    ccs_ON_OFF : CCs between ON- and OFF-filtered stimuli and neural activity (selected stimuli and response sets), for each 
    neuron
    ccs_ON_OFF_p_value : significance (True/False) of CCs between ON- and OFF-filtered stimuli and neural activity
    '''
    
    if not zero_padding: # if zero-padding was not used, set number of padding bins to 0
        padding_bins = 0
    
    f_range = np.logspace(np.log10(500), np.log10(22000), n_F)
    
    ccs_ON_OFF = np.zeros((n_neurons)) # array to hold CCs between ON- and OFF-filtered stimuli and neural activity
    # array to hold p-values of CCs between ON- and OFF-filtered stimuli and neural activity
    ccs_ON_OFF_p_value = np.zeros((n_neurons))
    
    resp_size = int(resp_dataset.size(-1)/len(stimuli[0])) # length (in time bins) of response to one stimulus
    stim_size = np.size(all_stim_dataset, 1)-padding_bins # length (in time bins) of one stimulus
    
    # arrays to hold ON- and OFF-filtered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    ON_OFF_stim_grid = np.zeros((n_F, len(stimuli)*len(stimuli[0])*stim_size))

    for fID in range(folds): # loop over folds
        for sID in range(len(stimuli[fID])): # loop over stimuli in each fold
            stim = all_stim_dataset[:, padding_bins:, stimuli[fID][sID]]
            
            # arrays to hold ON- and OFF-filtered stimuli
            ON_OFF_tr_stim = np.zeros_like(stim)
            
            for f_chan in range(n_F): # loop over frequency channels
                tao_f = 500 - 105*np.log10(f_range[f_chan])
                
                h_values = np.arange(0, 100, 1)
                exp_filter = np.exp(-h_values/tao_f)
                exp_filter = np.flip(exp_filter)
                exp_filter = exp_filter/np.sum(exp_filter)
                filtered_f_band = np.convolve(stim[f_chan], exp_filter, mode='same')
                hp_filtered_band = stim[f_chan] - filtered_f_band
                hwr_ON_OFF_stim = np.zeros_like(hp_filtered_band)
                hwr_ON_OFF_stim[np.where(hp_filtered_band >= 0)] = hp_filtered_band[np.where(hp_filtered_band >= 0)]
                ON_OFF_tr_stim[f_chan] = hwr_ON_OFF_stim
            
            ON_OFF_stim_grid[:, (2*fID + sID)*stim_size:(2*fID + sID + 1)*stim_size] = ON_OFF_tr_stim

    for nID in range(n_neurons): # loop over neurons - the correlation is measured for each neuron
        # array to hold responses from all folds, concatenated in time (dimensions -> (concatenated t))
        resp_grid = np.zeros((len(stimuli)*len(stimuli[0])*resp_size))
        
        for fID in range(folds): # loop over folds
            for sID in range(len(stimuli[fID])): # loop over stimuli in fold
                # insert response in grid
                resp = resp_dataset[nID, fID, sID*resp_size:(sID + 1)*resp_size].detach().cpu().numpy()
                resp_grid[(2*fID + sID)*resp_size:(2*fID + sID + 1)*resp_size] = resp
        
        # Pearson product-moment correlation coefficient
        if cc_type == 'pearson':
            # correlation of responses with frequency-averaged silence-triggered stimulus
            if averaging_type == 'average_stim':
                resp_pearson_r = stats.pearsonr(np.mean(ON_OFF_stim_grid, 0), resp_grid, alternative='greater')
                
                ccs_ON_OFF[nID] = resp_pearson_r[0]
                ccs_ON_OFF_p_value[nID] = resp_pearson_r[1]
            
            # average correlation of responses with each frequency channel of silence-triggered stimulus
            elif averaging_type == 'average_CC':
                ccs_ON_OFF_sum = 0
                
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_ON_OFF_sum += np.abs(stats.pearsonr(ON_OFF_stim_grid[f_chan], resp_grid, alternative='greater')[0])
                
                # divide by frequency channels
                ccs_ON_OFF[nID] = ccs_ON_OFF_sum/n_F
                ccs_ON_OFF_p_value[nID] = np.nan # p-value of CC is set to NaN if average_CC method is used
            
        # Spearman rank-order correlation coefficient
        elif cc_type == 'spearman':
            # correlation of responses with frequency-averaged silence-triggered stimulus
            if averaging_type == 'average_stim':
                resp_spearman_r = stats.spearmanr(np.mean(ON_OFF_stim_grid, 0), resp_grid, alternative='greater')
                
                ccs_ON_OFF[nID] = resp_spearman_r[0]
                ccs_ON_OFF_p_value[nID] = resp_spearman_r[1]
            
            # average correlation of responses with each frequency channel of silence-triggered stimulus
            elif averaging_type == 'average_CC':
                ccs_ON_OFF_sum = 0
                
                for f_chan in range(n_F): # loop over frequency channels
                    # sum over frequency channels
                    ccs_ON_OFF_sum += stats.spearmanr(ON_OFF_stim_grid[f_chan], resp_grid, alternative='greater')[0]
                
                # divide by frequency channels
                ccs_ON_OFF[nID] = ccs_ON_OFF_sum/n_F
                ccs_ON_OFF_p_value[nID] = np.nan # p-value of CC is set to NaN if average_CC method is used
    
    # p-values transformed into Boolean variables, where True indicates significance
    ccs_ON_OFF_p_value = ccs_ON_OFF_p_value < 0.01
    
    return [ccs_ON_OFF, ccs_ON_OFF_p_value]

#############################################################################################################################

def multi_analysis_scatter(ccs_change_measure, models_ccs, stims_used, stim_dataset, padding_bins, zero_padding, n_neurons, 
                           dataset_ID, models):
    
    log_measures = True
    decile_av_measures = True
    decile_pc_cc_change = True
    plot_coef_multi_model = False
    
    n_stims = np.size(stim_dataset, 2)
    n_top_stims = int(0.1*n_stims)
    
    ############################### Repetitiveness analysis ##########################
    autocorr_max = np.zeros((np.size(stim_dataset, 2)))
    autocorr_max_lag = np.zeros((np.size(stim_dataset, 2)))
    
    for sID in range(np.size(stim_dataset, 2)):
        stim = stim_dataset[:, padding_bins:, sID]
        stim_autocorr = scipy.signal.correlate2d(stim, stim, mode='same', boundary='wrap')
        autocorr_trace = stim_autocorr[int((np.size(stim_autocorr, 0)-1)/2)]
        autocorr_trace_norm = autocorr_trace/np.nanmax(autocorr_trace)
        autocorr_1sided = autocorr_trace_norm[np.where(autocorr_trace_norm == 1)[0][0]+1:]
        
        if len(np.where(np.diff(autocorr_1sided)>=0)[0]) > 0:
            autocorr_max[stims_used[sID]] = np.nanmax(autocorr_1sided[np.where(np.diff(autocorr_1sided)>=0)[0] + 1])
            autocorr_max_lag[stims_used[sID]] = np.where(autocorr_1sided == \
                                             np.nanmax(autocorr_1sided[np.where(np.diff(autocorr_1sided)>=0)[0] + 1]))[0][0]
        else:
            autocorr_max[stims_used[sID]] = 0
            autocorr_max_lag[stims_used[sID]] = 0
            
        autocorr_z = (autocorr_trace_norm - np.mean(autocorr_trace_norm)) / np.std(autocorr_trace_norm)
        autocorr_z_1sided = autocorr_z[np.where(autocorr_z == np.nanmax(autocorr_z))[0][0]+1:]
        
        if len(np.where(np.diff(autocorr_z_1sided)>=0)[0]) > 0:
            autocorr_z_max = np.nanmax(autocorr_z_1sided[np.where(np.diff(autocorr_z_1sided)>=0)[0] + 1])
        else:
            autocorr_z_max = 0
        
        autocorr_max[stims_used[sID]] = autocorr_z_max

    ############################### Silence duration analysis #########################
    if not zero_padding: # if zero-padding was not used, set number of padding bins to 0
        padding_bins = 0
    
    stim_dataset = stim_dataset[:, padding_bins:]
    n_F = np.size(stim_dataset, 0)
    
    # array to hold silence-triggered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    silence_tr_stim_grid = np.zeros_like(stim_dataset)


    for sID in range(len(stims_used)): # loop over stimuli in each fold
        stim = stim_dataset[:, :, stims_used[sID]]
        
        silence_tr_stim = np.zeros_like(stim) # array to hold silence-triggered stimulus
        
        for f_chan in range(n_F): # loop over frequency channels
            # z-score each stimulus frequency channel - negative log-power does not necessarily mean silence
            stim_chan_zscored = (stim[f_chan] - np.mean(stim[f_chan]))/np.std(stim[f_chan], ddof=1)
        
            negative_counter = 0 # set counter since last positive time bin to 0
            for t_bin in range(np.size(stim, 1)): # loop over time bins
                if stim_chan_zscored[t_bin] >= 0: # if stimulus is >= 0, silence-triggered stimulus is set to 0
                    silence_tr_stim[f_chan, t_bin] = 0
                    negative_counter = 0 # counter remains at/is reset to 0 
                else:
                    # if stimulus is < 0, counter is updated and silence-triggered stimulus is set to counter
                    negative_counter += 1
                    silence_tr_stim[f_chan, t_bin] = negative_counter
        
        # insert silence-triggered stimulus in grid
        silence_tr_stim_grid[:, :, sID] = silence_tr_stim
        
    av_silence_tr_stim_grid = np.nanmean(4*silence_tr_stim_grid, 0) # multiply by bin size
    av_silence_tr_stims = np.nanmean(av_silence_tr_stim_grid, 0)
    
    ############################### ON-OFF responses analysis ########################
    f_range = np.logspace(np.log10(500), np.log10(22000), n_F)
    
    # arrays to hold ON- and OFF-filtered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    ON_OFF_stim_grid = np.zeros_like(stim_dataset)

    for sID in range(len(stims_used)): # loop over stimuli in each fold
        stim = stim_dataset[:, :, stims_used[sID]]
        
        av_amp = np.nanmean(stim)
        std_amp = np.nanstd(stim, ddof=1)
        stim = (stim - av_amp)/std_amp
        
        # arrays to hold ON- and OFF-filtered stimuli
        ON_OFF_tr_stim = np.zeros_like(stim)
        
        for f_chan in range(n_F): # loop over frequency channels
            tao_f = 500 - 105*np.log10(f_range[f_chan])
            tao_f = tao_f/4
            
            hp_filtered_band = high_pass_filter(stim[f_chan], 1/tao_f, 250)
            
            # half-wave rectification
            hwr_ON_OFF_stim = np.zeros_like(hp_filtered_band)
            hwr_ON_OFF_stim[np.where(hp_filtered_band >= 0)] = hp_filtered_band[np.where(hp_filtered_band >= 0)]
            ON_OFF_tr_stim[f_chan] = hwr_ON_OFF_stim
        
        ON_OFF_stim_grid[:, :, sID] = ON_OFF_tr_stim
    
    var_ON_OFF_stims = np.var(ON_OFF_stim_grid, 1)
    av_var_ON_OFF_stims = np.nanmean(var_ON_OFF_stims, 0)
    
    ############################### Intensity analysis ###############################
    av_stim_amp = np.nanmean(stim_dataset, (0, 1))
    
    ###############################
    if log_measures:
        av_silence_tr_stims = np.log(av_silence_tr_stims)
        av_var_ON_OFF_stims = np.log(av_var_ON_OFF_stims)
        av_stim_amp = np.log(-av_stim_amp)
    
    ###############################
    blocks = int(n_stims/n_top_stims)
    
    autocorr_stims_idx = []
    silence_stims_idx = []
    ON_OFF_stims_idx = []
    amp_stims_idx = []
    
    for block in range(blocks):
        if block == 0:
            autocorr_stims_idx.append(np.argsort(autocorr_max)[-n_top_stims:])
            silence_stims_idx.append(np.argsort(av_silence_tr_stims)[-n_top_stims:])
            ON_OFF_stims_idx.append(np.argsort(av_var_ON_OFF_stims)[-n_top_stims:])
            amp_stims_idx.append(np.argsort(av_stim_amp)[-n_top_stims:])
            
        elif block > 0 and block < blocks-1:
            autocorr_stims_idx.append(np.argsort(autocorr_max)[-(block+1)*n_top_stims:-block*n_top_stims])
            silence_stims_idx.append(np.argsort(av_silence_tr_stims)[-(block+1)*n_top_stims:-block*n_top_stims])
            ON_OFF_stims_idx.append(np.argsort(av_var_ON_OFF_stims)[-(block+1)*n_top_stims:-block*n_top_stims])
            amp_stims_idx.append(np.argsort(av_stim_amp)[-(block+1)*n_top_stims:-block*n_top_stims])
            
        elif block == blocks-1:
            autocorr_stims_idx.append(np.argsort(autocorr_max)[:-block*n_top_stims])
            silence_stims_idx.append(np.argsort(av_silence_tr_stims)[:-block*n_top_stims])
            ON_OFF_stims_idx.append(np.argsort(av_var_ON_OFF_stims)[:-block*n_top_stims])
            amp_stims_idx.append(np.argsort(av_stim_amp)[:-block*n_top_stims])
    
    # Relate top stimuli for each analysis to CCnorm
    models_av_ccs = {}
    for model_ID in models:
        models_av_ccs[model_ID] = np.nanmean(models_ccs[model_ID][2], 0)
    
    cc_diff = {}
    cc_diff_pvalues = {}
    reg_model = {}
    
    for i_model_ID in models: # loop over models
        for j_model_ID in models: # loop over models
            if i_model_ID != j_model_ID and i_model_ID == 'pop_f_LSTM':
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID] = {}

                # difference in CCnorm between two models, for each neuron
                ccs_diff = models_av_ccs[i_model_ID] - models_av_ccs[j_model_ID]
                
                # percentage change in CCnorm between two models, for each neuron
                ccs_percent_change = (models_av_ccs[i_model_ID] - models_av_ccs[j_model_ID])/models_av_ccs[j_model_ID]*100
                
                # choose CCnorm change measure
                if ccs_change_measure == 'CC_diff':
                    ccs_change = ccs_diff
                elif ccs_change_measure == 'CC_percent_change':
                    ccs_change = ccs_percent_change
                
                autocorr_stims_ccs, silence_stims_ccs, ON_OFF_stims_ccs, amp_stims_ccs = [], [], [], []
                autocorr_blocks, silence_blocks, ON_OFF_blocks, amp_blocks = [], [], [], []
                
                for block in range(blocks):
                    autocorr_stims_ccs.append(ccs_change[autocorr_stims_idx[block]])
                    silence_stims_ccs.append(ccs_change[silence_stims_idx[block]])
                    ON_OFF_stims_ccs.append(ccs_change[ON_OFF_stims_idx[block]])
                    amp_stims_ccs.append(ccs_change[amp_stims_idx[block]])
                    
                    autocorr_blocks.append(autocorr_max[autocorr_stims_idx[block]])
                    silence_blocks.append(av_silence_tr_stims[silence_stims_idx[block]])
                    ON_OFF_blocks.append(av_var_ON_OFF_stims[ON_OFF_stims_idx[block]])
                    amp_blocks.append(av_stim_amp[amp_stims_idx[block]])
                
                autocorr_stims_ccs = list(reversed(autocorr_stims_ccs))
                silence_stims_ccs = list(reversed(silence_stims_ccs))
                ON_OFF_stims_ccs = list(reversed(ON_OFF_stims_ccs))
                amp_stims_ccs = list(reversed(amp_stims_ccs))
                
                autocorr_blocks = list(reversed(autocorr_blocks))
                silence_blocks = list(reversed(silence_blocks))
                ON_OFF_blocks = list(reversed(ON_OFF_blocks))
                amp_blocks = list(reversed(amp_blocks))
                
                autocorr_stims_ccs_blocks = [np.nanmean(block_stims) for block_stims in autocorr_stims_ccs]
                silence_stims_ccs_blocks = [np.nanmean(block_stims) for block_stims in silence_stims_ccs]
                ON_OFF_stims_ccs_blocks = [np.nanmean(block_stims) for block_stims in ON_OFF_stims_ccs]
                amp_stims_ccs_blocks = [np.nanmean(block_stims) for block_stims in amp_stims_ccs]
                
                av_autocorr_blocks = [np.nanmean(block) for block in autocorr_blocks]
                av_silence_blocks = [np.nanmean(block) for block in silence_blocks]
                av_ON_OFF_blocks = [np.nanmean(block) for block in ON_OFF_blocks]
                av_amp_blocks = [np.nanmean(block) for block in amp_blocks]
                
                cc_diff[i_model_ID + ' vs ' + j_model_ID] = \
                    {'autocorr': [autocorr_stims_ccs, autocorr_stims_ccs_blocks], 
                     'silence': [silence_stims_ccs, silence_stims_ccs_blocks], 
                     'ON_OFF': [ON_OFF_stims_ccs, ON_OFF_stims_ccs_blocks],
                     'amp': [amp_stims_ccs, amp_stims_ccs_blocks]}
                
                ######### Blocked (decile) plotting
                # Autocorrelation
                if decile_av_measures:
                    blocks_axis = av_autocorr_blocks
                else:
                    blocks_axis = np.arange(blocks)
                
                X2 = sm.add_constant(blocks_axis) # add bias to independent variable
                xseq = np.linspace(np.min(blocks_axis)-0.05, np.max(blocks_axis)+0.05, num=100)
                
                if decile_pc_cc_change:
                    autocorr_ccs_pc_change = autocorr_stims_ccs_blocks/np.mean(autocorr_stims_ccs_blocks)*100
                else:
                    autocorr_ccs_pc_change = autocorr_stims_ccs_blocks
                est = sm.OLS(autocorr_ccs_pc_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                
                # Spearman r
                spearman_test = stats.spearmanr(blocks_axis, autocorr_ccs_pc_change)
                
                f1, ax1 = plt.subplots(1, 1)
                ax1.scatter(blocks_axis, autocorr_ccs_pc_change)
                ax1.plot(xseq, a+b*xseq, color='r')
                ax1.set_xlabel('block - autocorrelation')
                if decile_pc_cc_change:
                    ax1.set_ylabel('CC diff/Mean CC diff (%)')
                else:
                    ax1.set_ylabel(ccs_change_measure)
                ax1.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]) + ' - ' + dataset_ID)
                ax1.spines[['right', 'top']].set_visible(False)
                
                #f1.savefig('/home/lorenzom/Downloads/Temp/Multi_analysis/' + dataset_ID + '_autocorr_1.png')
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID]['autocorr'] = est2.pvalues[1]
                
                # Silence
                if decile_av_measures:
                    blocks_axis = av_silence_blocks
                
                X2 = sm.add_constant(blocks_axis) # add bias to independent variable
                xseq = np.linspace(np.min(blocks_axis)-0.05, np.max(blocks_axis)+0.05, num=100)
                
                if decile_pc_cc_change:
                    silence_ccs_pc_change = silence_stims_ccs_blocks/np.mean(silence_stims_ccs_blocks)*100
                else:
                    silence_ccs_pc_change = silence_stims_ccs_blocks
                est = sm.OLS(silence_ccs_pc_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope

                # Spearman r
                spearman_test = stats.spearmanr(blocks_axis, silence_ccs_pc_change)
                
                f2, ax2 = plt.subplots(1, 1)
                ax2.scatter(blocks_axis, silence_ccs_pc_change)
                ax2.plot(xseq, a+b*xseq, color='r')
                ax2.set_xlabel('block - silence')
                if decile_pc_cc_change:
                    ax2.set_ylabel('CC diff/Mean CC diff (%)')
                else:
                    ax2.set_ylabel(ccs_change_measure)
                ax2.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]) + ' - ' + dataset_ID)
                ax2.spines[['right', 'top']].set_visible(False)
                #ax2.set_xticks([2, 3, 4, 5])
                #ax2.set_yticks([0.006, 0.008, 0.01, 0.012])
                #ax2.set_yticks([80, 100, 120])
                #ax2.xaxis.set_tick_params(labelcolor='none')
                #ax2.yaxis.set_tick_params(labelcolor='none')
                
                #f2.savefig('/home/lorenzom/Downloads/Temp/Multi_analysis/' + dataset_ID + '_silence_1.png')
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID]['silence'] = est2.pvalues[1]
                
                # Sudden changes
                if decile_av_measures:  
                    blocks_axis = av_ON_OFF_blocks
                
                X2 = sm.add_constant(blocks_axis) # add bias to independent variable
                xseq = np.linspace(np.min(blocks_axis)-0.05, np.max(blocks_axis)+0.05, num=100)
                
                if decile_pc_cc_change:
                    ON_OFF_ccs_pc_change = ON_OFF_stims_ccs_blocks/np.mean(ON_OFF_stims_ccs_blocks)*100
                else:
                    ON_OFF_ccs_pc_change = ON_OFF_stims_ccs_blocks
                est = sm.OLS(ON_OFF_ccs_pc_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope

                # Spearman r
                spearman_test = stats.spearmanr(blocks_axis, ON_OFF_ccs_pc_change)
                
                f3, ax3 = plt.subplots(1, 1)
                ax3.scatter(blocks_axis, ON_OFF_ccs_pc_change)
                ax3.plot(xseq, a+b*xseq, color='r')
                #ax3.set_xlabel('block - ON-OFF')
                '''if decile_pc_cc_change:
                    ax3.set_ylabel('CC diff/Mean CC diff (%)')
                else:
                    ax3.set_ylabel(ccs_change_measure)'''
                #ax3.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]) + ' - ' + dataset_ID)
                ax3.spines[['right', 'top']].set_visible(False)
                ax3.set_xticks([-3, -2, -1, 0])
                #ax3.set_yticks([0.006, 0.008, 0.010, 0.012])
                ax3.set_yticks([70, 90, 110, 130])
                ax3.xaxis.set_tick_params(labelcolor='none')
                ax3.yaxis.set_tick_params(labelcolor='none')
                
                #f3.savefig('/home/lorenzom/Downloads/Temp/Multi_analysis/' + dataset_ID + '_ON_OFF_1.png')
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID]['ON_OFF'] = est2.pvalues[1]
                
                # Amplitude
                if decile_av_measures:
                    blocks_axis = av_amp_blocks
                
                X2 = sm.add_constant(blocks_axis) # add bias to independent variable
                xseq = np.linspace(np.min(blocks_axis)-0.05, np.max(blocks_axis)+0.05, num=100)
                
                if decile_pc_cc_change:
                    amp_ccs_pc_change = amp_stims_ccs_blocks/np.mean(amp_stims_ccs_blocks)*100
                else:
                    amp_ccs_pc_change = amp_stims_ccs_blocks
                est = sm.OLS(amp_ccs_pc_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                b = est2.params[1] # slope
                
                # Spearman r
                spearman_test = stats.spearmanr(blocks_axis, amp_ccs_pc_change)
                
                f4, ax4 = plt.subplots(1, 1)
                ax4.scatter(blocks_axis, amp_ccs_pc_change)
                ax4.plot(xseq, a+b*xseq, color='r')
                ax4.set_xlabel('block - amp')
                if decile_pc_cc_change:
                    ax4.set_ylabel('CC diff/Mean CC diff (%)')
                else:
                    ax4.set_ylabel(ccs_change_measure)
                ax4.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]) + ' - ' + dataset_ID)
                ax4.spines[['right', 'top']].set_visible(False)
                
                #f4.savefig('/home/lorenzom/Downloads/Temp/Multi_analysis/' + dataset_ID + '_amp_1.png')
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID]['amp'] = est2.pvalues[1]
                
                ######### Full scatter plotting
                if log_measures:
                    X = np.stack((av_silence_tr_stims, av_var_ON_OFF_stims, av_stim_amp), axis=1)
                else:
                    X = np.stack((autocorr_max, av_silence_tr_stims, av_var_ON_OFF_stims, av_stim_amp), axis=1)
                #poly = PolynomialFeatures(interaction_only=True, include_bias=False)
                #X = poly.fit_transform(X)
                y = ccs_change
                
                ### OLS model
                X2 = sm.add_constant(X)
                est = sm.OLS(y, X2)
                est2 = est.fit()
                print(est2.summary())
                
                reg_model[i_model_ID + ' vs ' + j_model_ID] = est2
                
                ### Robust regression models
                model = HuberRegressor()
                model.fit(X2, y)
                
                params = np.append(model.intercept_,model.coef_[1:])
                predictions = model.predict(X2)
                
                newX = X2
                MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
                
                var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
                sd_b = np.sqrt(var_b)
                ts_b = params/ sd_b
                
                p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
                
                sd_b = np.round(sd_b,3)
                ts_b = np.round(ts_b,3)
                p_values = np.round(p_values,3)
                params = np.round(params,4)
                
                if log_measures:
                    silence_coef = est2.params[1]
                    ON_OFF_coef = est2.params[2]
                    amp_coef = est2.params[3]
                    
                    silence_p = est2.pvalues[1]
                    ON_OFF_p = est2.pvalues[2]
                    amp_p = est2.pvalues[3]
                
                else:
                    autocorr_coef = est2.params[1]
                    silence_coef = est2.params[2]
                    ON_OFF_coef = est2.params[3]
                    amp_coef = est2.params[4]
                    
                    autocorr_p = est2.pvalues[1]
                    silence_p = est2.pvalues[2]
                    ON_OFF_p = est2.pvalues[3]
                    amp_p = est2.pvalues[4]
                
                # Autocorrelation
                X2 = sm.add_constant(autocorr_max) # add bias to independent variable
                xseq = np.linspace(np.min(autocorr_max)-0.05, np.max(autocorr_max)+0.05, num=100)
            
                est = sm.OLS(ccs_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                if plot_coef_multi_model:
                    b = autocorr_coef
                else:
                    b = est2.params[1] # slope
                
                f1, ax1 = plt.subplots(1, 1)
                ax1.scatter(autocorr_max, ccs_change)
                ax1.plot(xseq, a+b*xseq, color='b')
                ax1.set_xlabel('autocorrelation')
                ax1.set_ylabel('CC diff')
                if plot_coef_multi_model:
                    ax1.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(autocorr_p) + ' - ' + dataset_ID)
                else:
                    ax1.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]) + ' - ' + dataset_ID)
                ax1.spines[['right', 'top']].set_visible(False)
                
                #f1.savefig('/home/lorenzom/Downloads/Temp/Multi_analysis/Scatter/' + dataset_ID + '_autocorr_1.png') 
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID]['autocorr'] = est2.pvalues[1]
                
                # Silence
                X2 = sm.add_constant(av_silence_tr_stims) # add bias to independent variable
                xseq = np.linspace(np.min(av_silence_tr_stims)-0.05, np.max(av_silence_tr_stims)+0.05, num=100)
                
                est = sm.OLS(ccs_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                if plot_coef_multi_model:
                    b = silence_coef
                else:
                    b = est2.params[1] # slope
                
                f2, ax2 = plt.subplots(1, 1)
                ax2.scatter(av_silence_tr_stims, ccs_change, label='regular - '+str(est2.pvalues[1]))
                ax2.plot(xseq, a+b*xseq, color='k', linewidth=3)
                ax2.set_xlabel('silence duration')
                ax2.set_ylabel('CC diff')
                if plot_coef_multi_model:
                    ax2.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(silence_p) + ' - ' + dataset_ID)
                else:
                    ax2.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]) + ' - ' + dataset_ID)
                
                #ax2.set_xticks([1, 2, 3, 4, 5])
                #ax2.set_yticks([0, 15, 30, 45])
                #ax2.xaxis.set_tick_params(labelcolor='none')
                #ax2.yaxis.set_tick_params(labelcolor='none')
                ax2.spines[['right', 'top']].set_visible(False)
                
                #f2.savefig('/home/lorenzom/Downloads/Temp/Multi_analysis/Scatter/' + dataset_ID + '_silence_1.png')
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID]['silence'] = est2.pvalues[1]
                    
                # Sudden changes
                X2 = sm.add_constant(av_var_ON_OFF_stims) # add bias to independent variable
                xseq = np.linspace(np.min(av_var_ON_OFF_stims)-0.05, np.max(av_var_ON_OFF_stims)+0.05, num=100)
                
                est = sm.OLS(ccs_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                if plot_coef_multi_model:
                    b = ON_OFF_coef
                else:
                    b = est2.params[1] # slope
                
                f3, ax3 = plt.subplots(1, 1)
                ax3.scatter(av_var_ON_OFF_stims, ccs_change, label='regular - '+str(est2.pvalues[1]))
                ax3.plot(xseq, a+b*xseq, color='k', linewidth=3)
                ax3.set_xlabel('ON-OFF')
                ax3.set_ylabel('CC diff')
                if plot_coef_multi_model:
                    ax3.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(ON_OFF_p) + ' - ' + dataset_ID)
                else:
                    ax3.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]) + ' - ' + dataset_ID)
                
                #ax3.set_xticks([-6, -4, -2, 0])
                #ax3.set_yticks([0, 15, 30, 45])
                #ax3.xaxis.set_tick_params(labelcolor='none')
                #ax3.yaxis.set_tick_params(labelcolor='none')
                ax3.spines[['right', 'top']].set_visible(False)
                
                #f3.savefig('/home/lorenzom/Downloads/Temp/Multi_analysis/Scatter/' + dataset_ID + '_ON_OFF_1.png')
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID]['ON_OFF'] = est2.pvalues[1]
                
                # Amplitude
                X2 = sm.add_constant(av_stim_amp) # add bias to independent variable
                xseq = np.linspace(np.min(av_stim_amp)-0.05, np.max(av_stim_amp)+0.05, num=100)
                
                est = sm.OLS(ccs_change, X2) # Ordinary Least Squares fit
                est2 = est.fit()
                a = est2.params[0] # bias
                if plot_coef_multi_model:
                    b = amp_coef
                else:
                    b = est2.params[1] # slope
                
                f4, ax4 = plt.subplots(1, 1)
                ax4.scatter(av_stim_amp, ccs_change, label='regular - '+str(est2.pvalues[1]))
                ax4.plot(xseq, a+b*xseq, color='b')
                ax4.set_xlabel('amp')
                ax4.set_ylabel('CC diff')
                if plot_coef_multi_model:
                    ax4.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(amp_p) + ' - ' + dataset_ID)
                else:
                    ax4.set_title(i_model_ID + ' vs ' + j_model_ID + ' - p = ' + str(est2.pvalues[1]) + ' - ' + dataset_ID)
                ax4.spines[['right', 'top']].set_visible(False)
                
                #f4.savefig('/home/lorenzom/Downloads/Temp/Multi_analysis/Scatter/' + dataset_ID + '_amp_1.png')
                cc_diff_pvalues[i_model_ID + ' vs ' + j_model_ID]['amp'] = est2.pvalues[1]
    
    cc_diffs = {}
    cc_diffs['regular'] = cc_diff
    
    cc_diffs_pvalues = {}
    cc_diffs_pvalues['regular'] = cc_diff_pvalues
    
    reg_models = {}
    reg_models['regular'] = reg_model
    
    return cc_diffs, cc_diffs_pvalues, reg_models

#############################################################################################################################

def multi_analysis_stim_only(stim_dataset, stims_used, padding_bins, zero_padding, time_axis):
    parula_cmap = fake_parula()
    
    if not zero_padding: # if zero-padding was not used, set number of padding bins to 0
        padding_bins = 0
    
    stim_dataset = stim_dataset[:, padding_bins:]
    n_F = np.size(stim_dataset, 0)
    
    f_range = np.logspace(np.log10(500), np.log10(22000), n_F)
    t_i = np.mean([time_axis[:-1], time_axis[1:]], axis=0)
    
    ############################### Repetitiveness analysis ##########################
    autocorr_max = np.zeros((np.size(stim_dataset, 2)))
    autocorr_max_lag = np.zeros((np.size(stim_dataset, 2)))
    
    rep_scores = np.zeros((np.size(stim_dataset, 2)))
    
    for sID in range(len(stims_used)): # loop over stimuli in each fold
        stim = stim_dataset[:, :, stims_used[sID]]
        
        stim_autocorr = scipy.signal.correlate2d(stim, stim, mode='same', boundary='wrap')
        autocorr_trace = stim_autocorr[int((np.size(stim_autocorr, 0)-1)/2)]
        #autocorr_trace_norm = autocorr_trace/np.nanmax(autocorr_trace)
        #autocorr_1sided = autocorr_trace_norm[np.where(autocorr_trace_norm == 1)[0][0]+1:]
        autocorr_trace_norm = autocorr_trace
        autocorr_1sided = autocorr_trace_norm[np.where(autocorr_trace_norm == np.nanmax(autocorr_trace))[0][0]+1:]
        
        if len(np.where(np.diff(autocorr_1sided)>=0)[0]) > 0:
            autocorr_max[stims_used[sID]] = np.nanmax(autocorr_1sided[np.where(np.diff(autocorr_1sided)>=0)[0] + 1])
            autocorr_max_lag[stims_used[sID]] = np.where(autocorr_1sided == \
                                             np.nanmax(autocorr_1sided[np.where(np.diff(autocorr_1sided)>=0)[0] + 1]))[0][0]
        else:
            autocorr_max[stims_used[sID]] = 0
            autocorr_max_lag[stims_used[sID]] = 0
        
        autocorr_z = (autocorr_trace_norm - np.mean(autocorr_trace_norm)) / np.std(autocorr_trace_norm)
        autocorr_z_1sided = autocorr_z[np.where(autocorr_z == np.nanmax(autocorr_z))[0][0]+1:]
        #autocorr_z_max = np.nanmax(autocorr_z_1sided[np.where(np.diff(autocorr_z_1sided)>=0)[0] + 1])
    
    ############################### Silence duration analysis #############################
    # array to hold silence-triggered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    silence_tr_stim_grid = np.zeros_like(stim_dataset)

    for sID in range(len(stims_used)): # loop over stimuli in each fold
        stim = stim_dataset[:, :, stims_used[sID]]
        
        silence_tr_stim = np.zeros_like(stim) # array to hold silence-triggered stimulus
        
        for f_chan in range(n_F): # loop over frequency channels
            # z-score each stimulus frequency channel - negative log-power does not necessarily mean silence
            stim_chan_zscored = (stim[f_chan] - np.mean(stim[f_chan]))/np.std(stim[f_chan], ddof=1)

            negative_counter = 0 # set counter since last positive time bin to 0
            for t_bin in range(np.size(stim, 1)): # loop over time bins
                if stim_chan_zscored[t_bin] >= 0: # if stimulus is >= 0, silence-triggered stimulus is set to 0
                #if stim_chan_zscored[t_bin] >= -1: # if stimulus is >= 0, silence-triggered stimulus is set to 0
                    silence_tr_stim[f_chan, t_bin] = 0
                    negative_counter = 0 # counter remains at/is reset to 0 
                else:
                    # if stimulus is < 0, counter is updated and silence-triggered stimulus is set to counter
                    negative_counter += 1
                    silence_tr_stim[f_chan, t_bin] = negative_counter
        
        # insert silence-triggered stimulus in grid
        silence_tr_stim_grid[:, :, sID] = 4*silence_tr_stim # multiply by bin size
        
    av_silence_tr_stim_grid = np.nanmean(silence_tr_stim_grid, 0)
    av_silence_tr_stims = np.nanmean(av_silence_tr_stim_grid, 0)
    
    #av_silence_tr_stims = np.log(av_silence_tr_stims)
    
    ############################### ON-OFF responses analysis ################################
    f_range = np.logspace(np.log10(500), np.log10(22000), n_F)
    
    # arrays to hold ON- and OFF-filtered stimuli from all folds, concatenated in time (dimensions -> (f, concatenated t))
    ON_OFF_stim_grid = np.zeros_like(stim_dataset)

    for sID in range(len(stims_used)):
        stim = stim_dataset[:, :, stims_used[sID]]
        
        av_amp = np.nanmean(stim)
        std_amp = np.nanstd(stim, ddof=1)
        stim = (stim - av_amp)/std_amp
        
        # arrays to hold ON- and OFF-filtered stimuli
        ON_OFF_tr_stim = np.zeros_like(stim)
        
        for f_chan in range(n_F): # loop over frequency channels
            tao_f = 500 - 105*np.log10(f_range[f_chan])
            tao_f = tao_f/4

            hp_filtered_band = high_pass_filter(stim[f_chan], 1/tao_f, 250)
            #hp_filtered_band = stim[f_chan] - high_pass_filter(stim[f_chan], 1/av_tao, 250)
            
            # half-wave rectification
            hwr_ON_OFF_stim = np.zeros_like(hp_filtered_band)
            hwr_ON_OFF_stim[np.where(hp_filtered_band >= 0)] = hp_filtered_band[np.where(hp_filtered_band >= 0)]
            ON_OFF_tr_stim[f_chan] = hwr_ON_OFF_stim

        ON_OFF_stim_grid[:, :, sID] = ON_OFF_tr_stim

    var_ON_OFF_stims = np.var(ON_OFF_stim_grid, 1)
    av_var_ON_OFF_stims = np.nanmean(var_ON_OFF_stims, 0)
    
    #av_var_ON_OFF_stims = np.log(av_var_ON_OFF_stims)
    
    ############################# Overall stimulus intensity #############################
    av_stim_amp = np.nanmean(stim_dataset, (0, 1))
    
    #av_stim_amp = np.log(av_stim_amp)
    
    
    
    X2 = sm.add_constant(av_stim_amp) # add bias to independent variable
    est = sm.OLS(av_var_ON_OFF_stims, X2) # Ordinary Least Squares fit
    est2 = est.fit()
    a = est2.params[0] # bias
    b = est2.params[1] # slope
    
    xseq = np.linspace(np.min(av_stim_amp)-0.05, np.max(av_stim_amp)+0.05, num=100)
    
    f, ax = plt.subplots()
    ax.scatter(av_stim_amp, av_var_ON_OFF_stims)
    ax.plot(xseq, a+b*xseq, color='r')
    ax.set_xlabel('stim intensity')
    ax.set_ylabel('sudden changes measure')
    #ax.set_ylim([-10, 130])
    ax.set_title('p = ' + str(est2.pvalues[1]))
    
    est = sm.OLS(av_silence_tr_stims, X2) # Ordinary Least Squares fit
    est2 = est.fit()
    a = est2.params[0] # bias
    b = est2.params[1] # slope
    
    f, ax = plt.subplots()
    ax.scatter(av_stim_amp, av_silence_tr_stims)
    ax.plot(xseq, a+b*xseq, color='r')
    ax.set_xlabel('stim intensity')
    ax.set_ylabel('silence duration measure')
    #ax.set_ylim([-10, 130])
    ax.set_title('p = ' + str(est2.pvalues[1]))
    
    return av_silence_tr_stims, av_var_ON_OFF_stims, av_stim_amp

def high_pass_filter(signal: np.ndarray, cutoff: float, fs: float, order: int = 4, axis: int = -1) -> np.ndarray:
    """
    Applies a Butterworth high-pass filter to a NumPy array (1D or 2D).

    Parameters:
    - signal: NumPy array of the signal (1D or 2D).
    - cutoff: Cutoff frequency in Hz.
    - fs: Sampling frequency in Hz.
    - order: Order of the filter (higher means sharper cutoff).
    - axis: Axis to filter along (default is -1, last axis).

    Returns:
    - Filtered signal as a NumPy array.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False)  # Design filter
    
    # Apply filter along the specified axis
    filtered_signal = filtfilt(b, a, signal, axis=axis)
    
    return filtered_signal


#############################################################################################################################

def rcLSTM_weight_share(n_neurons, stim_dataset, resp_dataset, models_fits_dict, model_ID, folds, best_lambdas):
    if model_ID == 'pop_directrcLSTM':
        lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
        model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
        hidden_size = model_params['hidden_size'] # network hidden size
        fc_layer_size = np.size(models_fits_dict[0][0][0][0]['b_added1'])
        
        # array to hold direct weights sum for each neuron
        fc_weights_all_lamb = np.zeros((n_neurons, folds, len(lambda_sequence)))
        # array to hold gated recurrent weights sum for each neuron
        subLSTM_weights_all_lamb = np.zeros((n_neurons, folds, len(lambda_sequence)))
        # array to hold direct weights percentage for each neuron
        fc_weights_percent_all_lamb = np.zeros((n_neurons, folds, len(lambda_sequence)))
        # array to hold gated recurrent weights percentage for each neuron
        subLSTM_weights_percent_all_lamb = np.zeros((n_neurons, folds, len(lambda_sequence)))
        
        for fID in range(folds): # loop over folds
            for lamb in range(len(lambda_sequence)): # loop over lambdas
                model = getattr(all_models, model_ID)(model_params) # initialise model
                model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
                
                w_model = models_fits_dict[0][fID][lamb][0] # fitted model weights
                
                fc_w = w_model['w_added_to_out']
                subLSTM_w = w_model['w_l1']
                
                fc_w_weighted = w_model['w_added_to_out']/fc_layer_size
                subLSTM_w_weighted = w_model['w_l1']/hidden_size
                
                for nID in range(np.size(fc_w, 0)):
                    fc_weights_all_lamb[nID, fID, lamb] = np.sum(np.abs(fc_w[nID]))
                    subLSTM_weights_all_lamb[nID, fID, lamb] = np.sum(np.abs(subLSTM_w[nID]))
                    
                    w_sum = np.sum(np.abs(fc_w_weighted[nID])) + np.sum(np.abs(subLSTM_w_weighted[nID]))
                    
                    fc_weights_percent_all_lamb[nID, fID, lamb] = np.sum(np.abs(fc_w_weighted[nID]))/w_sum
                    subLSTM_weights_percent_all_lamb[nID, fID, lamb] = np.sum(np.abs(subLSTM_w_weighted[nID]))/w_sum
        
        fc_weights = np.zeros((n_neurons))
        subLSTM_weights = np.zeros((n_neurons))
        fc_weights_percent = np.zeros((n_neurons))
        subLSTM_weights_percent = np.zeros((n_neurons))
        
        for nID in range(n_neurons):
            fc_weights[nID] = fc_weights_all_lamb[nID, 0, best_lambdas[model_ID][nID]]
            subLSTM_weights[nID] = subLSTM_weights_all_lamb[nID, 0, best_lambdas[model_ID][nID]]
            fc_weights_percent[nID] = fc_weights_percent_all_lamb[nID, 0, best_lambdas[model_ID][nID]]
            subLSTM_weights_percent[nID] = subLSTM_weights_percent_all_lamb[nID, 0, best_lambdas[model_ID][nID]]
            
        return {'fc': fc_weights, 'subLSTM': subLSTM_weights, 'fc_percent': fc_weights_percent, 
                'subLSTM_percent': subLSTM_weights_percent}

#############################################################################################################################

def rcLSTM_weighted_output(n_neurons, stim_dataset, resp_dataset, models_fits_dict, model_ID, folds, best_lambdas):
    if model_ID == 'pop_directrcLSTM':
        lambda_sequence = models_fits_dict[1]['lambdas'] # list of lambdas trialled in cross-validation
        model_params = models_fits_dict[1]['all_model_params'] # hyperparameters used to implement model
    
        # array to hold variance of hidden units (over time) weighted by hidden-to-output weight for each neuron
        fc_var_all_lamb = np.zeros((n_neurons, folds, len(lambda_sequence)))
        subLSTM_var_all_lamb = np.zeros((n_neurons, folds, len(lambda_sequence)))
        
        # array to hold percentage variance of hidden units (over time) weighted by hidden-to-output weight for each neuron
        fc_var_all_lamb_percent = np.zeros((n_neurons, folds, len(lambda_sequence)))
        subLSTM_var_all_lamb_percent = np.zeros((n_neurons, folds, len(lambda_sequence)))
        
        for fID in range(folds): # loop over folds
            for lamb in range(len(lambda_sequence)): # loop over lambdas
                model = getattr(all_models, model_ID)(model_params) # initialise model
                model.reload(models_fits_dict[0][fID][lamb][0]) # reload fitted parameters from fold and lambda
                
                # NumPy traces of model internal variables during forward pass of stimuli
                traces = model.np_forward_loop(stim_dataset[fID])
                
                w_model = models_fits_dict[0][fID][lamb][0] # fitted model weights
                
                w_fc = w_model['w_added_to_out']
                w_subLSTM = w_model['w_l1']
                
                weighted_fc_act = np.dot(np.transpose(traces['x_t']), np.transpose(w_fc))
                weighted_subLSTM_act = np.dot(np.transpose(traces['h']), np.transpose(w_subLSTM))

                fc_var = np.var(weighted_fc_act, 0)
                subLSTM_var = np.var(weighted_subLSTM_act, 0)

                    
                fc_var_all_lamb[:, fID, lamb] = fc_var
                subLSTM_var_all_lamb[:, fID, lamb] = subLSTM_var
                
                fc_var_all_lamb_percent[:, fID, lamb] = fc_var/(fc_var + subLSTM_var)
                subLSTM_var_all_lamb_percent[:, fID, lamb] = subLSTM_var/(fc_var + subLSTM_var)
        
        fc_var = np.zeros((n_neurons))
        subLSTM_var = np.zeros((n_neurons))
        
        fc_var_percent = np.zeros((n_neurons))
        subLSTM_var_percent = np.zeros((n_neurons))
        
        for nID in range(n_neurons):
            fc_var[nID] = fc_var_all_lamb[nID, 0, best_lambdas[model_ID][nID]]
            subLSTM_var[nID] = subLSTM_var_all_lamb[nID, 0, best_lambdas[model_ID][nID]]
            
            fc_var_percent[nID] = fc_var_all_lamb_percent[nID, 0, best_lambdas[model_ID][nID]]
            subLSTM_var_percent[nID] = subLSTM_var_all_lamb_percent[nID, 0, best_lambdas[model_ID][nID]]
            
        return {'fc': fc_var, 'subLSTM': subLSTM_var, 'fc_percent': fc_var_percent, 'subLSTM_percent' : subLSTM_var_percent}
    
#############################################################################################################################

def count_params(models_fits_dict):
    params = models_fits_dict[0][0][0][0]
    
    num_params = 0    
    for param in params:
        num_params += np.size(params[param])
        
    return num_params

#############################################################################################################################

def calc_se_median(models_results_dict):
    results = models_results_dict[2]
    n_neurons = len(results)
    
    median_se = {}
    
    median_se['median'] = np.nanmedian(results)
    
    reps = 10000
    sample_median = np.zeros((reps))
    
    for rep in range(reps):
        sample = np.random.randint(0, n_neurons, size=n_neurons, dtype=int)
        sample_median[rep] = np.nanmedian(results[sample])
        
    median_se['se'] = np.nanstd(sample_median, ddof=1)
    
    return median_se