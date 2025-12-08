import os
import scipy.io.wavfile as wav
import numpy as np
from util_funcs import get_cochleagrams_NS2, get_noise_ratio_NS2, get_psth
import pickle

#############################################################################################################################

def build_dataset_NS2_include_single(params, zero_padding):
    cochleagram_type = params['cochleagram_type']
    bin_size = params['bin_size']
    n_F = params['n_F']
    min_F = params['min_F']
    max_F = params['max_F']
    padding_bins = params['padding_bins']
    
    dataset = 'NS2_include_single'
    
    filename = 'Data/NS2_raw/NS2_prebuild.pckl' # pre-built dataset, used to save time
    f = open(filename, 'rb')
    NS2_prebuild = pickle.load(f)
    f.close()
    
    unique_stim_list = NS2_prebuild['unique_stim_list'] # list of IDs of stimuli which will be used
    unique_stim_occ = NS2_prebuild['unique_stim_occ'] # number of repeats of each stimulus in unique_stim_list
    sessions = NS2_prebuild['sessions'] # list of session IDs
    unique_stim_num = NS2_prebuild['unique_stim_num'] # list of number of repeats of each stimulus, by session
    neuron_IDs = NS2_prebuild['neuron_IDs'] # list of neuron IDs, by session
    # list of spike times by session, for each neuron, for each stimulus in chronological order of presentation
    # only contains data from sessions where the stimuli in unique_stim_list were presented
    all_sess_resps = NS2_prebuild['all_sess_resps']
    # list of stimulus IDs by session, in chronological order of presentation with repetitions
    stim_IDs = NS2_prebuild['stim_IDs']
    

    # list of stimuli, within unique_stim_list, presented more than once
    multi_repeat_stims = [unique_stim_list[x] for x in range(len(unique_stim_occ)) if unique_stim_occ[x] > 1]
    # list of stimuli, within unique_stim_list, presented once
    single_repeat_stims = [unique_stim_list[x] for x in range(len(unique_stim_occ)) if unique_stim_occ[x] == 1]

    multi_rep_resps = []
    single_rep_resps = []
    
    subset_neuron_array = []
    subset_session_array = []
    
    sess_count = 0
    for session in range(len(sessions)): # loop over sessions
        if len(unique_stim_num[session]) == 306: # only consider session containing stimuli from unique_stim_list
            for nID in range(len(neuron_IDs[session])): # loop over neurons in each session
                # dictionary to hold responses of neuron to multi-repeat stimuli
                multi_rep_dict = dict.fromkeys(multi_repeat_stims, [])
                # dictionary to hold responses of neuron to single-repeat stimuli
                single_rep_dict = dict.fromkeys(single_repeat_stims, [])
                
                # list of spike times of neuron in response to each stimulus, in chronological order of presentation
                n_resps = all_sess_resps[sess_count][nID]
                
                for stim in range(len(n_resps)): # loop over stimuli in chronological order of presentation
                    sID = stim_IDs[session][stim] # stimulus ID
                    n_resp_stim = n_resps[stim] # list of spike times of neuron in response to stimulus

                    if sID in multi_rep_dict: # check if stimulus is a multi-repeat or a single-repeat
                        if not multi_rep_dict[sID]: # if stimulus field in dict is empty, update field with spike times
                            multi_rep_dict.update({sID: [n_resp_stim]})
                        else:
                            multi_rep_dict[sID].append(n_resp_stim) # if field is not empty, simply append spike times
                    elif sID in single_rep_dict:
                        if not single_rep_dict[sID]:
                            single_rep_dict.update({sID: [n_resp_stim]})
                        else:
                            single_rep_dict[sID].append(n_resp_stim)

                multi_rep_resps.append(multi_rep_dict) # append to list containing data from all sessions
                single_rep_resps.append(single_rep_dict) # append to list containing data from all sessions
                
            subset_neuron_array.append(neuron_IDs[session]) # neurons in sessions of interest
            subset_session_array.append([sess_count+1]*len(neuron_IDs[session])) # assign number to each session
            
            sess_count += 1
            
    subset_neuron_array_flat = [n_ID for exp_session in subset_neuron_array for n_ID in exp_session] # flatten neuron ID list
    # flattened array of session numbers for each neuron; neurons with the same number were recorded in the same session
    subset_session_array_flat = [session_ID for exp_session in subset_session_array for session_ID in exp_session]

    ###################
    sounds_dir = '' # stimulus files directory
    soundFiles_lst = os.listdir(sounds_dir) # list content of directory
    
    soundFiles = [stim.lstrip('STIM_') for stim in multi_repeat_stims] # remove initial "STIM_" string from stimulus IDs
    # compute cochleagrams and relative time and frequency axes for multi-repeat stimuli
    multi_rep_cochleagrams, t, f = get_cochleagrams_NS2(sounds_dir, soundFiles, bin_size, n_F, min_F, max_F, 
                                                        cochleagram_type, dataset, padding_bins, zero_padding)  
    
    soundFiles = [stim.lstrip('STIM_') for stim in single_repeat_stims] # remove initial "STIM_" string from stimulus IDs

    sounds_list = []
    fs_list = []
    for sID in soundFiles_lst: # loop over all stimuli in directory
        # read in sounds and their sampling frequencies
        fs, y = wav.read(sounds_dir + sID)
        sounds_list.append(y)
        fs_list.append(fs)
    
    # remove responses to stimuli shorter than 4 seconds
    for stim in range(len(sounds_list)):
        if int(len(sounds_list[stim])/fs_list[stim]) != 4: # signal length/sampling frequency must be equal to 4
            mismatch_stim_ID = soundFiles_lst[stim]
            soundFiles.remove(mismatch_stim_ID) # remove stimuli from list of sound files

            for nID in range(len(single_rep_resps)):
                # remove responses to the stimuli for each neuron - only consider single repeats as multi repeats are not affected
                single_rep_resps[nID].pop('STIM_' + mismatch_stim_ID)
    
    # compute cochleagrams and relative time and frequency axes for single-repeat stimuli
    single_rep_cochleagrams, t, f = get_cochleagrams_NS2(sounds_dir, soundFiles, bin_size, n_F, min_F, max_F, 
                                                         cochleagram_type, dataset, padding_bins, zero_padding) 
    
    all_resps = []
    for nID in range(len(multi_rep_resps)):
        # responses to all stimuli for each neuron, as dicts with multi-repeat and single-repeat responses together
        all_resps.append({**multi_rep_resps[nID], **single_rep_resps[nID]})
    
    n_neurons = len(all_resps)
    n_multi_rep_stimuli = len(multi_rep_resps[0])
    n_single_rep_stimuli = len(single_rep_resps[0])
    
    neuron_response_psths = []
    neuron_firing_rates = np.zeros((n_neurons))
    
    # only calculate PSTHs of multi-repeat responses for noise ratio calculation
    for nID in range(n_neurons):
        n_spiketimes = multi_rep_resps[nID] # list of responses (spike times) of neuron to repeats of each stimulus
        # array to hold PSTHs, dimensions -> (stimuli, repeats, t)
        n_psths = np.zeros((n_multi_rep_stimuli, len(n_spiketimes[list(n_spiketimes.keys())[0]]), len(t)))
        
        stim_count = 0
        for sID in n_spiketimes:
            n_s_spiketimes = n_spiketimes[sID] # list of responses of neuron to each repeat of stimulus
            n_reps = len(n_s_spiketimes)
            
            for rID in range(n_reps):
                n_s_r_spiketimes = n_s_spiketimes[rID] # list of responses of neuron to repeat of stimulus
                # get PSTHs of response binned in time according to t
                n_s_r_psth, spikes_after_end = get_psth(n_s_r_spiketimes, t)
                n_psths[stim_count, rID, :] = np.append(n_s_r_psth[:], spikes_after_end)
            stim_count += 1
            
        # average firing rate of neuron (sum spikes and divide by stimulus length (4 s)) over all stimuli and repeats
        neuron_firing_rates[nID] = np.sum(n_psths)/(4*stim_count*n_reps)
        neuron_response_psths.append(n_psths)
    
    noise_ratios = []
    all_noisy_neurons = []
    
    non_noisy_neuron_data = []
    non_noisy_ratios = []
    non_noisy = 0
    
    subset_neur_array = []
    subset_sess_array = []
    
    for nID in range(n_neurons):
        # get noise ratio of PSTHs of responses to all stimuli and their repeats, for neuron nID
        # nrs is an array of NRs, one for each stimulus; nr is the NR over all stimuli
        nrs, nr = get_noise_ratio_NS2(neuron_response_psths[nID])
        #nrs, nr = get_noise_ratio_NS2(neuron_response_spiketrains[nID])
        noise_ratios.append(nr)
    
        if nr >= 40: # chosen noise threshold
            all_noisy_neurons.append(nID)
        else:
            if neuron_firing_rates[nID] > 0.5: # chosen average firing rate threshold to remove neurons that fire too rarely
                non_noisy += 1
                # keep neurons that survive both criteria
                non_noisy_neuron_data.append(all_resps[nID])
                non_noisy_ratios.append(nr)
                
                # keep neuron IDs and their corresponding session IDs
                subset_neur_array.append(subset_neuron_array_flat[nID])
                subset_sess_array.append(subset_session_array_flat[nID])
    
    # inconsistency criteria: remove neurons whose firing rate variance over repeats is over a certain threshold for more 
    # than half of the stimuli; threshold was determined by looking at average variances over repeats
    inconsistent = 0
    inconsistent_nID = []

    for nID in range(len(non_noisy_neuron_data)): # loop over neurons
        # array to hold firing rate variances for each stimulus, for one neuron
        n_av_fr_variance = np.zeros((len(list(multi_rep_resps[nID].keys()))))
        # number of stimuli - number of keys in dictionary of responses for each neuron
        n_stimuli = len(list(multi_rep_resps[nID].keys()))
        
        for sID in range(n_stimuli): # loop over stimuli
            # sum spike count over time bins for each repeat and divide by stimulus length to get average firing rate for 
            # each repeat (spikes/ms)
            av_fr_repeats = np.sum(neuron_response_psths[nID][sID], 1)/t[-1]

            # variance of average firing rate over repets, for one stimulus
            n_av_fr_variance[sID] = np.var(av_fr_repeats, ddof=1)
        
        # count number of stimuli for which firing rate variance is above a certain threshold
        n_var_inconsistent = np.count_nonzero(n_av_fr_variance > 1e-4)
        
        if n_var_inconsistent > n_stimuli/2: # if the count is higher than half the stimuli, neuron is inconsistent
            inconsistent = inconsistent + 1
            inconsistent_nID.append(nID)
    
    # reverse order to avoid changing later indices when removing elements
    inconsistent_nID = sorted(inconsistent_nID, reverse=True)
    
    # remove data from inconsistent neurons across all relevant variables
    for i_nID in range(len(inconsistent_nID)):
        non_noisy_neuron_data[inconsistent_nID[i_nID]] = []
        non_noisy_neuron_data.remove(non_noisy_neuron_data[inconsistent_nID[i_nID]])
        non_noisy_ratios.remove(non_noisy_ratios[inconsistent_nID[i_nID]])
        subset_neur_array.remove(subset_neur_array[inconsistent_nID[i_nID]])
        subset_sess_array.remove(subset_sess_array[inconsistent_nID[i_nID]])

    n_neurons = len(non_noisy_neuron_data)
    n_repeats = [10, 20]
    
    subset_neur_array = list(zip(subset_neur_array, subset_sess_array)) # neuron and session IDs
    
    params = {'bin_size': bin_size, 'n_F': n_F, 'min_F': min_F, 'max_F': max_F, 'n_neurons' : n_neurons, 
              'n_stimuli' : [n_multi_rep_stimuli, n_single_rep_stimuli], 'n_repeats' : n_repeats, 
              'noise_ratios' : non_noisy_ratios, 'stimuli_order' : list(all_resps[0].keys()),
              'neurons_ID_sess' : subset_neur_array, 't_axis' : t}
    
    return [multi_rep_cochleagrams, single_rep_cochleagrams], non_noisy_neuron_data, params

#############################################################################################################################

def build_dataset_NS3(params, zero_padding):
    cochleagram_type = params['cochleagram_type']
    bin_size = params['bin_size']
    n_F = params['n_F']
    min_F = params['min_F']
    max_F = params['max_F']
    padding_bins = params['padding_bins']
    
    dataset = 'NS3'

    filename = 'Data/NS3/NS3_prebuild.pckl' # pre-built dataset, used to save time
    f = open(filename, 'rb')
    NS3_prebuild = pickle.load(f)
    f.close()

    unique_stim_list = NS3_prebuild['unique_stim_list'] # list of IDs of stimuli which will be used
    unique_stim_occ = NS3_prebuild['unique_stim_occ'] # number of repeats of each stimulus in unique_stim_list
    sessions = NS3_prebuild['sessions'] # list of session IDs
    neuron_IDs = NS3_prebuild['neuron_IDs'] # list of neuron IDs, by session
    # list of spike times by session, for each neuron, for each stimulus in chronological order of presentation
    all_sess_resps = NS3_prebuild['all_sess_resps']
    # list of stimulus IDs by session, in chronological order of presentation with repetitions
    stim_IDs = NS3_prebuild['stim_IDs']
    
    # list of stimuli, within unique_stim_list, presented more than once
    # occurrence == 20 is used as a criteria instead of occurrence > 1, because there are two sounds presented twice;
    # these two sounds and their responses, in all neurons, are excluded from the modelling
    multi_repeat_stims = [unique_stim_list[x] for x in range(len(unique_stim_occ)) if unique_stim_occ[x] == 20]
    # list of stimuli, within unique_stim_list, presented once
    single_repeat_stims = [unique_stim_list[x] for x in range(len(unique_stim_occ)) if unique_stim_occ[x] == 1]

    multi_rep_resps = []
    single_rep_resps = []

    subset_session_array = []

    sess_count = 0
    for session in range(len(sessions)): # loop over sessions
        for nID in range(len(neuron_IDs[session])): # loop over neurons in each session
            # dictionary to hold responses of neuron to multi-repeat stimuli
            multi_rep_dict = dict.fromkeys(multi_repeat_stims, [])
            # dictionary to hold responses of neuron to single-repeat stimuli
            single_rep_dict = dict.fromkeys(single_repeat_stims, [])
            
            # list of spike times of neuron in response to each stimulus, in chronological order of presentation
            n_resps = all_sess_resps[sess_count][nID]

            for stim in range(len(n_resps)): # loop over stimuli in chronological order of presentation
                sID = stim_IDs[session][stim] # stimulus ID
                n_resp_stim = n_resps[stim] # list of spike times of neuron in response to stimulus
                
                if sID in multi_rep_dict: # check if stimulus is a multi-repeat or a single-repeat
                    if not multi_rep_dict[sID]: # if stimulus field in dict is empty, update field with spike times
                        multi_rep_dict.update({sID: [n_resp_stim]})
                    else:
                        multi_rep_dict[sID].append(n_resp_stim) # if field is not empty, simply append spike times
                elif sID in single_rep_dict:
                    if not single_rep_dict[sID]:
                        single_rep_dict.update({sID: [n_resp_stim]})
                    else:
                        single_rep_dict[sID].append(n_resp_stim)
        
            multi_rep_resps.append(multi_rep_dict) # append to list containing data from all sessions
            single_rep_resps.append(single_rep_dict) # append to list containing data from all sessions
        
        subset_session_array.append([sess_count+1]*len(neuron_IDs[session])) # assign number to each session
        
        sess_count += 1
    
    subset_neuron_array_flat = [n_ID for exp_session in neuron_IDs for n_ID in exp_session] # flatten neuron ID list
    # flattened array of session numbers for each neuron; neurons with the same number were recorded in the same session
    subset_session_array_flat = [session_ID for exp_session in subset_session_array for session_ID in exp_session]
    
    ###################
    sounds_dir = '' # stimulus files directory
    soundFiles_lst = os.listdir(sounds_dir) # list content of directory

    soundFiles = [stim.lstrip('STIM_') for stim in multi_repeat_stims] # remove initial "STIM_" string from stimulus IDs
    # compute cochleagrams and relative time and frequency axes for multi-repeat stimuli
    multi_rep_cochleagrams, t, f = get_cochleagrams_NS2(sounds_dir, soundFiles, bin_size, n_F, min_F, max_F, 
                                                        cochleagram_type, dataset, padding_bins, zero_padding)  

    soundFiles = [stim.lstrip('STIM_') for stim in single_repeat_stims] # remove initial "STIM_" string from stimulus IDs
    
    sounds_list = []
    fs_list = []
    for sID in soundFiles_lst: # loop over all stimuli in directory
        # read in sounds and their sampling frequencies
        fs, y = wav.read(sounds_dir + sID)
        sounds_list.append(y)
        fs_list.append(fs)   
    
    # remove responses to stimuli shorter or longer than 1 seconds, if any
    for stim in range(len(sounds_list)):
        if int(len(sounds_list[stim])/fs_list[stim]) != 1: # signal length/sampling frequency must be equal to 1
            mismatch_stim_ID = soundFiles_lst[stim]
            soundFiles.remove(mismatch_stim_ID) # remove stimuli from list of sound files
            
            for nID in range(len(single_rep_resps)):
                # remove responses to the stimuli for each neuron - only consider single repeats as multi repeats are not affected
                single_rep_resps[nID].pop('STIM_' + mismatch_stim_ID)
    
    # compute cochleagrams and relative time and frequency axes for single-repeat stimuli
    single_rep_cochleagrams, t, f = get_cochleagrams_NS2(sounds_dir, soundFiles, bin_size, n_F, min_F, max_F, 
                                                         cochleagram_type, dataset, padding_bins, zero_padding)  
    
    all_resps = []
    for nID in range(len(multi_rep_resps)):
        # responses to all stimuli for each neuron, as dicts with multi-repeat and single-repeat responses together
        all_resps.append({**multi_rep_resps[nID], **single_rep_resps[nID]})

    n_neurons = len(all_resps)
    n_multi_rep_stimuli = len(multi_rep_resps[0])
    n_single_rep_stimuli = len(single_rep_resps[0])

    neuron_response_psths = []
    neuron_firing_rates = np.zeros((n_neurons))
    
    # only calculate PSTHs of multi-repeat responses for noise ratio calculation
    for nID in range(n_neurons):
        n_spiketimes = multi_rep_resps[nID] # list of responses (spike times) of neuron to repeats of each stimulus
        # array to hold PSTHs, dimensions -> (stimuli, repeats, t)
        n_psths = np.zeros((n_multi_rep_stimuli, len(n_spiketimes[list(n_spiketimes.keys())[0]]), len(t)))
        
        stim_count = 0
        for sID in n_spiketimes:
            n_s_spiketimes = n_spiketimes[sID] # list of responses of neuron to each repeat of stimulus
            n_reps = len(n_s_spiketimes)
            
            for rID in range(n_reps):
                n_s_r_spiketimes = n_s_spiketimes[rID] # list of responses of neuron to repeat of stimulus
                # get PSTHs of response binned in time according to t
                n_s_r_psth, spikes_after_end = get_psth(n_s_r_spiketimes, t)
                n_psths[stim_count, rID, :] = np.append(n_s_r_psth[:], spikes_after_end)
            stim_count += 1
        
        # average firing rate of neuron (sum spikes and divide by stimulus length (1 s)) over all stimuli and repeats
        neuron_firing_rates[nID] = np.sum(n_psths)/(1*stim_count*n_reps)
        neuron_response_psths.append(n_psths)

    noise_ratios = []
    all_noisy_neurons = []

    non_noisy_neuron_data = []
    non_noisy_ratios = []
    non_noisy = 0
    
    subset_neur_array = []
    subset_sess_array = []
    
    for nID in range(n_neurons):
        # get noise ratio of PSTHs of responses to all stimuli and their repeats, for neuron nID
        # nrs is an array of NRs, one for each stimulus; nr is the NR over all stimuli
        nrs, nr = get_noise_ratio_NS2(neuron_response_psths[nID])
        #nrs, nr = get_noise_ratio_NS2(neuron_response_spiketrains[nID])
        noise_ratios.append(nr)

        if nr >= 40 or np.isnan(nr): # chosen noise threshold; also exclude neuron if noise ratio is NaN
            all_noisy_neurons.append(nID)
        else:
            if neuron_firing_rates[nID] > 0.5: # chosen average firing rate threshold to remove neurons that fire too rarely
                non_noisy += 1
                # keep neurons that survive both criteria
                non_noisy_neuron_data.append(all_resps[nID])
                non_noisy_ratios.append(nr)

                # keep neuron IDs and their corresponding session IDs
                subset_neur_array.append(subset_neuron_array_flat[nID])
                subset_sess_array.append(subset_session_array_flat[nID])

    # inconsistency criteria: remove neurons whose firing rate variance over repeats is over a certain threshold 
    # for more than half of the stimuli; threshold was determined by looking at average variances over repeats
    inconsistent = 0
    inconsistent_nID = []

    for nID in range(len(non_noisy_neuron_data)): # loop over neurons
        # array to hold firing rate variances for each stimulus, for one neuron
        n_av_fr_variance = np.zeros((len(list(multi_rep_resps[nID].keys()))))
        # number of stimuli - number of keys in dictionary of responses for each neuron
        n_stimuli = len(list(multi_rep_resps[nID].keys()))
        
        for sID in range(n_stimuli): # loop over stimuli
            # sum spike count over time bins for each repeat and divide by stimulus length to get average firing rate for 
            # each repeat (spikes/ms)
            av_fr_repeats = np.sum(neuron_response_psths[nID][sID], 1)/t[-1]

            # variance of average firing rate over repeats, for one stimulus
            n_av_fr_variance[sID] = np.var(av_fr_repeats, ddof=1)
        
        # count number of stimuli for which firing rate variance is above a certain threshold
        n_var_inconsistent = np.count_nonzero(n_av_fr_variance > 1e-4)
        
        if n_var_inconsistent > n_stimuli/2: # if the count is higher than half the stimuli, neuron is inconsistent
            inconsistent = inconsistent + 1
            inconsistent_nID.append(nID)
    
    # reverse order to avoid changing later indices when removing elements
    inconsistent_nID = sorted(inconsistent_nID, reverse=True)
    
    # remove data from inconsistent neurons across all relevant variables
    for i_nID in range(len(inconsistent_nID)):
        non_noisy_neuron_data[inconsistent_nID[i_nID]] = []
        non_noisy_neuron_data.remove(non_noisy_neuron_data[inconsistent_nID[i_nID]])
        non_noisy_ratios.remove(non_noisy_ratios[inconsistent_nID[i_nID]])
        subset_neur_array.remove(subset_neur_array[inconsistent_nID[i_nID]])
        subset_sess_array.remove(subset_sess_array[inconsistent_nID[i_nID]])
        
    n_neurons = len(non_noisy_neuron_data)
    n_repeats = 20

    subset_neur_array = list(zip(subset_neur_array, subset_sess_array)) # neuron and session IDs

    params = {'bin_size': bin_size, 'n_F': n_F, 'min_F': min_F, 'max_F': max_F, 'n_neurons' : n_neurons,
              'n_stimuli' : [n_multi_rep_stimuli, n_single_rep_stimuli], 'n_repeats' : n_repeats,
              'noise_ratios' : non_noisy_ratios, 'stimuli_order' : list(all_resps[0].keys()),
              'neurons_ID_sess' : subset_neur_array, 't_axis' : t}
    
    return [multi_rep_cochleagrams, single_rep_cochleagrams], non_noisy_neuron_data, params

#############################################################################################################################

def build_dataset_NS3_PEG(params, zero_padding):
    cochleagram_type = params['cochleagram_type']
    bin_size = params['bin_size']
    n_F = params['n_F']
    min_F = params['min_F']
    max_F = params['max_F']
    padding_bins = params['padding_bins']
    
    dataset = 'NS3_PEG'

    filename = 'Data/NS3_PEG/NS3_PEG_prebuild.pckl' # pre-built dataset, used to save time
    f = open(filename, 'rb')
    NS3_PEG_prebuild = pickle.load(f)
    f.close()

    unique_stim_list = NS3_PEG_prebuild['unique_stim_list'] # list of IDs of stimuli which will be used
    unique_stim_occ = NS3_PEG_prebuild['unique_stim_occ'] # number of repeats of each stimulus in unique_stim_list
    sessions = NS3_PEG_prebuild['sessions'] # list of session IDs
    neuron_IDs = NS3_PEG_prebuild['neuron_IDs'] # list of neuron IDs, by session
    # list of spike times by session, for each neuron, for each stimulus in chronological order of presentation
    all_sess_resps = NS3_PEG_prebuild['all_sess_resps']
    # list of stimulus IDs by session, in chronological order of presentation with repetitions
    stim_IDs = NS3_PEG_prebuild['stim_IDs']

    # list of stimuli, within unique_stim_list, presented more than once
    # occurrence == 20 is used as a criteria instead of occurrence > 1, because there are two sounds presented twice;
    # these two sounds and their responses, in all neurons, are excluded from the modelling
    multi_repeat_stims = [unique_stim_list[x] for x in range(len(unique_stim_occ)) if unique_stim_occ[x] == 20]
    # list of stimuli, within unique_stim_list, presented once
    single_repeat_stims = [unique_stim_list[x] for x in range(len(unique_stim_occ)) if unique_stim_occ[x] == 1]

    multi_rep_resps = []
    single_rep_resps = []

    subset_session_array = []

    sess_count = 0
    for session in range(len(sessions)): # loop over sessions
        for nID in range(len(neuron_IDs[session])): # loop over neurons in each session
            # dictionary to hold responses of neuron to multi-repeat stimuli
            multi_rep_dict = dict.fromkeys(multi_repeat_stims, [])
            # dictionary to hold responses of neuron to single-repeat stimuli
            single_rep_dict = dict.fromkeys(single_repeat_stims, [])
            
            # list of spike times of neuron in response to each stimulus, in chronological order of presentation
            n_resps = all_sess_resps[sess_count][nID]

            for stim in range(len(n_resps)): # loop over stimuli in chronological order of presentation
                sID = stim_IDs[session][stim] # stimulus ID
                n_resp_stim = n_resps[stim] # list of spike times of neuron in response to stimulus
                
                if sID in multi_rep_dict: # check if stimulus is a multi-repeat or a single-repeat
                    if not multi_rep_dict[sID]: # if stimulus field in dict is empty, update field with spike times
                        multi_rep_dict.update({sID: [n_resp_stim]})
                    else:
                        multi_rep_dict[sID].append(n_resp_stim) # if field is not empty, simply append spike times
                elif sID in single_rep_dict:
                    if not single_rep_dict[sID]:
                        single_rep_dict.update({sID: [n_resp_stim]})
                    else:
                        single_rep_dict[sID].append(n_resp_stim)
        
            multi_rep_resps.append(multi_rep_dict) # append to list containing data from all sessions
            single_rep_resps.append(single_rep_dict) # append to list containing data from all sessions
        
        subset_session_array.append([sess_count+1]*len(neuron_IDs[session])) # assign number to each session
        
        sess_count += 1
    
    subset_neuron_array_flat = [n_ID for exp_session in neuron_IDs for n_ID in exp_session] # flatten neuron ID list
    # flattened array of session numbers for each neuron; neurons with the same number were recorded in the same session
    subset_session_array_flat = [session_ID for exp_session in subset_session_array for session_ID in exp_session]

    ###################
    sounds_dir = '' # stimulus files directory
    soundFiles_lst = os.listdir(sounds_dir) # list content of directory

    soundFiles = [stim.lstrip('STIM_') for stim in multi_repeat_stims] # remove initial "STIM_" string from stimulus IDs
    # compute cochleagrams and relative time and frequency axes for multi-repeat stimuli
    multi_rep_cochleagrams, t, f = get_cochleagrams_NS2(sounds_dir, soundFiles, bin_size, n_F, min_F, max_F, 
                                                        cochleagram_type, dataset, padding_bins, zero_padding)  

    soundFiles = [stim.lstrip('STIM_') for stim in single_repeat_stims] # remove initial "STIM_" string from stimulus IDs

    sounds_list = []
    fs_list = []
    for sID in soundFiles_lst: # loop over all stimuli in directory
        # read in sounds and their sampling frequencies
        fs, y = wav.read(sounds_dir + sID)
        sounds_list.append(y)
        fs_list.append(fs)   
    
    # remove responses to stimuli shorter or longer than 1 seconds, if any
    for stim in range(len(sounds_list)):
        if int(len(sounds_list[stim])/fs_list[stim]) != 1: # signal length/sampling frequency must be equal to 1
            mismatch_stim_ID = soundFiles_lst[stim]
            soundFiles.remove(mismatch_stim_ID) # remove stimuli from list of sound files
            
            for nID in range(len(single_rep_resps)):
                # remove responses to the stimuli for each neuron - only consider single repeats as multi repeats are not affected
                single_rep_resps[nID].pop('STIM_' + mismatch_stim_ID)
    
    # compute cochleagrams and relative time and frequency axes for single-repeat stimuli
    single_rep_cochleagrams, t, f = get_cochleagrams_NS2(sounds_dir, soundFiles, bin_size, n_F, min_F, max_F, 
                                                         cochleagram_type, dataset, padding_bins, zero_padding)  
    
    all_resps = []
    for nID in range(len(multi_rep_resps)):
        # responses to all stimuli for each neuron, as dicts with multi-repeat and single-repeat responses together
        all_resps.append({**multi_rep_resps[nID], **single_rep_resps[nID]})

    n_neurons = len(all_resps)
    n_multi_rep_stimuli = len(multi_rep_resps[0])
    n_single_rep_stimuli = len(single_rep_resps[0])

    neuron_response_psths = []
    neuron_firing_rates = np.zeros((n_neurons))
    
    # only calculate PSTHs of multi-repeat responses for noise ratio calculation
    for nID in range(n_neurons):
        n_spiketimes = multi_rep_resps[nID] # list of responses (spike times) of neuron to repeats of each stimulus
        # array to hold PSTHs, dimensions -> (stimuli, repeats, t)
        n_psths = np.zeros((n_multi_rep_stimuli, len(n_spiketimes[list(n_spiketimes.keys())[0]]), len(t)))
        
        stim_count = 0
        for sID in n_spiketimes:
            n_s_spiketimes = n_spiketimes[sID] # list of responses of neuron to each repeat of stimulus
            n_reps = len(n_s_spiketimes)
            
            for rID in range(n_reps):
                n_s_r_spiketimes = n_s_spiketimes[rID] # list of responses of neuron to repeat of stimulus
                # get PSTHs of response binned in time according to t
                n_s_r_psth, spikes_after_end = get_psth(n_s_r_spiketimes, t)
                n_psths[stim_count, rID, :] = np.append(n_s_r_psth[:], spikes_after_end)
            stim_count += 1
        
        # average firing rate of neuron (sum spikes and divide by stimulus length (1 s)) over all stimuli and repeats
        neuron_firing_rates[nID] = np.sum(n_psths)/(1*stim_count*n_reps)
        neuron_response_psths.append(n_psths)

    noise_ratios = []
    all_noisy_neurons = []

    non_noisy_neuron_data = []
    non_noisy_ratios = []
    non_noisy = 0
    
    subset_neur_array = []
    subset_sess_array = []
    
    for nID in range(n_neurons):
        # get noise ratio of PSTHs of responses to all stimuli and their repeats, for neuron nID
        # nrs is an array of NRs, one for each stimulus; nr is the NR over all stimuli
        nrs, nr = get_noise_ratio_NS2(neuron_response_psths[nID])
        #nrs, nr = get_noise_ratio_NS2(neuron_response_spiketrains[nID])
        noise_ratios.append(nr)

        if nr >= 40 or np.isnan(nr): # chosen noise threshold; also exclude neuron if noise ratio is NaN
            all_noisy_neurons.append(nID)
        else:
            if neuron_firing_rates[nID] > 0.5: # chosen average firing rate threshold to remove neurons that fire too rarely
                non_noisy += 1
                # keep neurons that survive both criteria
                non_noisy_neuron_data.append(all_resps[nID])
                non_noisy_ratios.append(nr)

                # keep neuron IDs and their corresponding session IDs
                subset_neur_array.append(subset_neuron_array_flat[nID])
                subset_sess_array.append(subset_session_array_flat[nID])

    # inconsistency criteria: remove neurons whose firing rate variance over repeats is over a certain threshold 
    # for more than half of the stimuli; threshold was determined by looking at average variances over repeats
    inconsistent = 0
    inconsistent_nID = []

    for nID in range(len(non_noisy_neuron_data)): # loop over neurons
        # array to hold firing rate variances for each stimulus, for one neuron
        n_av_fr_variance = np.zeros((len(list(multi_rep_resps[nID].keys()))))
        # number of stimuli - number of keys in dictionary of responses for each neuron
        n_stimuli = len(list(multi_rep_resps[nID].keys()))
        
        for sID in range(n_stimuli): # loop over stimuli
            # sum spike count over time bins for each repeat and divide by stimulus length to get average firing rate for 
            # each repeat (spikes/ms)
            av_fr_repeats = np.sum(neuron_response_psths[nID][sID], 1)/t[-1]

            # variance of average firing rate over repeats, for one stimulus
            n_av_fr_variance[sID] = np.var(av_fr_repeats, ddof=1)
        
        # count number of stimuli for which firing rate variance is above a certain threshold
        n_var_inconsistent = np.count_nonzero(n_av_fr_variance > 1e-4)
        
        if n_var_inconsistent > n_stimuli/2: # if the count is higher than half the stimuli, neuron is inconsistent
            inconsistent = inconsistent + 1
            inconsistent_nID.append(nID)
    
    # reverse order to avoid changing later indices when removing elements
    inconsistent_nID = sorted(inconsistent_nID, reverse=True)
    
    # remove data from inconsistent neurons across all relevant variables
    for i_nID in range(len(inconsistent_nID)):
        non_noisy_neuron_data[inconsistent_nID[i_nID]] = []
        non_noisy_neuron_data.remove(non_noisy_neuron_data[inconsistent_nID[i_nID]])
        non_noisy_ratios.remove(non_noisy_ratios[inconsistent_nID[i_nID]])
        subset_neur_array.remove(subset_neur_array[inconsistent_nID[i_nID]])
        subset_sess_array.remove(subset_sess_array[inconsistent_nID[i_nID]])
        
    n_neurons = len(non_noisy_neuron_data)
    n_repeats = 20

    subset_neur_array = list(zip(subset_neur_array, subset_sess_array)) # neuron and session IDs

    params = {'bin_size': bin_size, 'n_F': n_F, 'min_F': min_F, 'max_F': max_F, 'n_neurons' : n_neurons,
              'n_stimuli' : [n_multi_rep_stimuli, n_single_rep_stimuli], 'n_repeats' : n_repeats,
              'noise_ratios' : non_noisy_ratios, 'stimuli_order' : list(all_resps[0].keys()),
              'neurons_ID_sess' : subset_neur_array, 't_axis' : t}
    
    return [multi_rep_cochleagrams, single_rep_cochleagrams], non_noisy_neuron_data, params