import numpy as np
import matplotlib.pyplot as plt
import torch

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def whosMy(*args):
    sequentialTypes = [dict, list, tuple] 
    for var in args:
        t=type(var)
        if t==np.ndarray:  
            print(type(var), var.dtype, var.shape)
        elif t in sequentialTypes: 
            print(type(var), len(var))
        else:
            print(type(var))
        
#############################################################################################################################

def extents(k):
    delta = k[1] - k[0]
    return [k[0] - delta/2, k[-1] + delta/2]
        
#############################################################################################################################

def random_train_val_test_split(n_stimuli, test_set_size, val_set_size, n_folds):
    '''
    Parameters
    ----------
    n_stimuli : total number of stimuli
    test_set_size : desired number of stimuli in test set
    val_set_size : desired number of stimuli in validation sets
    n_folds : number of cross-validation folds
    
    Returns
    -------
    training_stimuli : training sets (n_folds, n_stimuli-test_set_size-val_set_size)
    validation_stimuli : validation sets (n_folds, val_set_size)
    test_stimuli : test set (test_set_size)
    cv_stimuli : cross-validation set (n_stimuli-test_set_size)
    
    Random split of dataset into a test set and a training set; training set is also randomly split into a training set and 
    a validation set, with the number of folds specified by n_folds and without repetition
    '''
    
    import random
    
    # split dataset into a cross-validation and a test set
    stimuli = list(range(n_stimuli)) # list of all stimuli
    test_stimuli = random.sample(stimuli, k=test_set_size) # list of stimuli in the test set
    cv_stimuli = [sID for sID in stimuli if sID not in test_stimuli] # list of stimuli in the training + cv set

    # split cross-validation set 8-fold (non-overlapping) into training and validation sets
    folds = n_folds
    validation_stimuli = [] # list of lists of stimuli in the validation sets
    training_stimuli = [] # list of lists of stimuli in the training sets
    remaining_stimuli = cv_stimuli
    
    for i in range(folds):
        validation_fold = random.sample(remaining_stimuli, k=val_set_size)
        validation_stimuli.append(validation_fold)
        remaining_stimuli = [sID for sID in remaining_stimuli if sID not in validation_fold]
        training_stimuli.append([sID for sID in cv_stimuli if sID not in validation_fold])
        
    return training_stimuli, validation_stimuli, test_stimuli, cv_stimuli

#############################################################################################################################

def train_val_test_split(dataset, reduced, stim_dataset):
    '''
    Parameters
    ----------
    dataset : dataset ID
    reduced : boolean value; determines whether a reduced number of folds is used in cross-validation
    stim_dataset : single-repeat cochleagrams, only used for datasets that contain them

    Returns
    -------
    folds : number of cross-validation folds
    test_stimuli : test set stimuli
    cv_stimuli : cross-validation set stimuli
    validation_stimuli : validation sets stimuli (n_folds, stimuli)
    training_stimuli : training sets stimuli (n_folds, stimuli)
    stimuli : dictionary containing all (4) sets together
    
    Split of dataset into a test set and a training set; training set is also split into a training set and a validation set, 
    with the number of folds specified by n_folds and without repetition. Test sets for NS1 and NS2 correspond to those used
    in Rahman et al., 2020
    '''
        
    if dataset == 'NS2_include_single' or dataset == 'NS3':
        folds = 7
        test_stimuli = [2, 8, 11, 14]
        cv_stimuli = [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17]
        validation_stimuli = [[6, 3], [13, 7], [17, 9], [5, 0], [1, 4], [16, 10], [12, 15]]
        training_stimuli_multi = [[0, 1, 4, 5, 7, 9, 10, 12, 13, 15, 16, 17], 
                                  [0, 1, 3, 4, 5, 6, 9, 10, 12, 15, 16, 17],
                                  [0, 1, 3, 4, 5, 6, 7, 10, 12, 13, 15, 16],
                                  [1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 17],
                                  [0, 3, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17],
                                  [0, 1, 3, 4, 5, 6, 7, 9, 12, 13, 15, 17],
                                  [0, 1, 3, 4, 5, 6, 7, 9, 10, 13, 16, 17]]
        training_stimuli = [f_train_stim + list(range(np.size(stim_dataset, 2))) for f_train_stim in training_stimuli_multi]

        stimuli = {'training_stimuli' : training_stimuli, 'validation_stimuli' : validation_stimuli,
                   'cv_stimuli' : cv_stimuli, 'test_stimuli' : test_stimuli}
        
    elif dataset == 'NS3_PEG':
        folds = 7
        test_stimuli = [8, 14, 10, 17]
        cv_stimuli = [1, 4, 16, 12, 15, 7, 11, 3, 5, 6, 9, 13, 2, 0]
        validation_stimuli = [[7, 16], [9, 11], [0, 3], [15, 1], [4, 12], [2, 5], [6, 13]]
        training_stimuli_multi = [[1, 4, 12, 15, 11, 3, 5, 6, 9, 13, 2, 0], 
                                  [1, 4, 16, 12, 15, 7, 3, 5, 6, 13, 2, 0],
                                  [1, 4, 16, 12, 15, 7, 11, 5, 6, 9, 13, 2],
                                  [4, 16, 12, 7, 11, 3, 5, 6, 9, 13, 2, 0],
                                  [1, 16, 15, 7, 11, 3, 5, 6, 9, 13, 2, 0],
                                  [1, 4, 16, 12, 15, 7, 11, 3, 6, 9, 13, 0],
                                  [1, 4, 16, 12, 15, 7, 11, 3, 5, 9, 2, 0]]
        training_stimuli = [f_train_stim + list(range(np.size(stim_dataset, 2))) for f_train_stim in training_stimuli_multi]

        stimuli = {'training_stimuli' : training_stimuli, 'validation_stimuli' : validation_stimuli,
                   'cv_stimuli' : cv_stimuli, 'test_stimuli' : test_stimuli}
        
    return folds, test_stimuli, cv_stimuli, validation_stimuli, training_stimuli, stimuli

#############################################################################################################################

def get_cochleagrams_NS1(sounddir, bin_size, n_F, min_F, max_F, cochleagram_type, padding_bins, zero_padding):
    '''
    Parameters
    ----------
    sounddir : directory containing stimuli
    bin_size : temporal window used for binning stimuli
    n_F : number of desired cochleagram frequency channels
    min_F : minimum centre frequency of cochleagram channels
    max_F : maximum centre frequency of cochleagram channels
    cochleagram_type : cochleagram type (e.g., spec-power)
    padding_bins : number of bins for for zero-padding, if applicable
    zero_padding: boolean variable, controls zero-padding
        
    Returns
    -------
    cochleagrams : frequency-time representation of stimuli - numpy array of dimensions (frequency, time, stimuli)
    t : time axis used to produce cochleagrams
    f : frequency axis used to produce cochleagrams
    
    Returns the cochleagrams of the stimuli contained in the folder sounddir, with the frequency and time parameters 
    specified; an additional parameter is included that specifies how many, if any, time bins are to be clipped from the 
    beginning of the stimuli; also returns time and frequency axes for plotting the cochleagrams
    '''
    
    import os
    import numpy as np
    import scipy.io.wavfile as wav
    import cochlea_lorenzo
    
    dataset = 'NS1'
    
    cochleagram_fcn = 'cochleagram_' + cochleagram_type
    cochleagram = getattr(cochlea_lorenzo, cochleagram_fcn) # use cochleagram function for the chosen type
    
    soundFiles_lst = os.listdir(sounddir) # list content of sounddir
    soundFiles = sorted(soundFiles_lst) # sort content (NOT NECESSARILY NEEDED, MUST CHECK THIS)

    cochleagrams = np.array([])
    
    for sID in range(len(soundFiles)):
        # Read in raw sounds and their sampling frequencies
        fs, y = wav.read(sounddir + soundFiles[sID])
        
        if zero_padding:
            y = np.concatenate((np.zeros((int((padding_bins*bin_size/1000)*fs))), y)) # pad raw sounds with zeros

        # 'log' indicates log spacing of frequency channels
        X_ft, params = cochleagram(y, fs, bin_size, dataset, zero_padding, padding_bins, spacing='log', 
                                   freq_info=[min_F, max_F, n_F])
        
        '''
        # plot cochleagrams (and raw sounds) as they are computed
        plot_stimuli(y, fs, X_ft, bin_size, n_F, min_F, max_F)
        '''

        # if statement is necessary for concatenation - must share concatenated dimension
        if cochleagrams.shape[0] == 0:
            cochleagrams = np.expand_dims(X_ft, axis=2) # add 3rd (stimulus) dimension to cochleagram
        else:
            # concatenate cochleagrams along 3rd (stimulus) dimension
            cochleagrams = np.concatenate((cochleagrams, np.expand_dims(X_ft, axis=2)), axis=2)
    
    # time and frequency axes used for cochleagrams
    t = params['spectrogram_t'] 
    f = np.logspace(np.log10(min_F), np.log10(max_F), n_F) # logarithmically spaced frequency axis
    
    return cochleagrams, t, f

#############################################################################################################################

def get_cochleagrams_NS2(sounddir, soundFiles, bin_size, n_F, min_F, max_F, cochleagram_type, dataset, padding_bins, 
                         zero_padding):
    '''
    Parameters
    ----------
    sounddir : directory containing stimuli
    soundfiles : names of stimulus files
    bin_size : temporal window used for binning stimuli
    n_F : number of desired cochleagram frequency channels
    min_F : minimum centre frequency of cochleagram channels
    max_F : maximum centre frequency of cochleagram channels
    cochleagram_type : cochleagram type (e.g., spec-power)
    dataset : dataset ID
    padding_bins : number of history steps for zero-padding, if applicable
    zero_padding: boolean variable, controls zero-padding
        
    Returns
    -------
    cochleagrams : frequency-time representation of stimuli - numpy array of dimensions (frequency, time, stimuli)
    t : time axis used to produce cochleagrams
    f : frequency axis used to produce cochleagrams
    
    Returns the cochleagrams of the stimuli contained in the folder sounddir, named soundFiles, with the frequency and time 
    parameters specified; an additional parameter is included that specifies how many, if any, time bins are to be clipped 
    from the beginning of the stimuli; also returns time and frequency axes for plotting the cochleagrams
    '''
    
    import numpy as np
    import soundfile as sf
    import cochlea_lorenzo
    
    cochleagram_fcn = 'cochleagram_' + cochleagram_type
    cochleagram = getattr(cochlea_lorenzo, cochleagram_fcn) # use cochleagram function for the chosen type
    
    cochleagrams = np.array([])

    for sID in range(len(soundFiles)):
        # Read in sounds and their sampling frequencies
        y, fs = sf.read(sounddir + soundFiles[sID])
        
        if zero_padding:
            y = np.concatenate((np.zeros((int((padding_bins*bin_size/1000)*fs))), y)) # pad raw sounds with zeros
        
        # 'log' indicates log spacing of frequency channels
        X_ft, params = cochleagram(y, fs, bin_size, dataset, zero_padding, padding_bins, spacing='log', 
                                   freq_info=[min_F, max_F, n_F])
        
        # if statement is necessary for concatenation - must share concatenated dimension
        if cochleagrams.shape[0] == 0:
            cochleagrams = np.expand_dims(X_ft, axis=2) # add 3rd (stimulus) dimension to cochleagram
        else:
            # concatenate cochleagrams along 3rd (stimulus) dimension
            cochleagrams = np.concatenate((cochleagrams, np.expand_dims(X_ft, axis=2)), axis=2)
    
    # time and frequency axes used for cochleagrams
    t = params['spectrogram_t']
    f = np.logspace(np.log10(min_F), np.log10(max_F), n_F) # logarithmically spaced frequency axis
    
    return cochleagrams, t, f

#############################################################################################################################

def get_cochleagrams_NS3(sounddir, bin_size, n_F, min_F, max_F, cochleagram_type): # NOT IN USE
    '''
    Parameters
    ----------
    sounddir : directory containing stimulus data
    bin_size : temporal window used for binning stimulus data
    n_F : number of desired frequency channels of the cochleagrams
    min_F : minimum frequency of the cochleagrams
    max_F : maximum frequency of the cochleagrams
    cochleagram_type : cochleagram type (e.g., spec-power)
        
    Returns
    -------
    cochleagrams : frequency-time representation of stimuli - numpy array of dimensions (frequency, time, stimuli)
    t : time axis used to produce cochleagrams
    f : frequency axis used to produce cochleagrams
    
    Returns the cochleagrams of the stimuli contained in the folder sounddir, with the frequency and time parameters 
    specified; an additional parameter is included that specifies how many, if any, time bins are to be clipped from the 
    beginning of the stimuli; also returns time and frequenzy axes for plotting the cochleagrams
    '''
    
    import os
    import numpy as np
    import scipy.io.wavfile as wav
    import cochlea_lorenzo
    
    cochleagram_fcn = 'cochleagram_' + cochleagram_type
    cochleagram = getattr(cochlea_lorenzo, cochleagram_fcn)
    
    soundFiles_lst = os.listdir(sounddir) # lists content of sounddir
    soundFiles = sorted(soundFiles_lst) # sorts content (NOT NECESSARILY NEEDED, MUST CHECK THIS) 
    cochleagrams = np.array([])
    
    for sID in range(len(soundFiles)):
        # Read in raw sounds and their sampling frequencies
        fs, y = wav.read(sounddir + soundFiles[sID])
        
        # 'log' indicates log spacing of frequency channels
        X_ft, params = cochleagram(y, fs, bin_size, spacing='log', freq_info=[min_F, max_F, n_F])
        X_ft = X_ft[:, :int(1000/bin_size)] # only keep first second of the sounds, as this is all that was presented
        
        # if statement is necessary for concatenation - must share concatenated dimension
        if cochleagrams.shape[0] == 0:
            cochleagrams = np.expand_dims(X_ft, axis=2) # add dummy (concatenating) third dimension to cochleagram
        else:
            # concatenate cochleagram along 3rd (stimulus) dimension
            cochleagrams = np.concatenate((cochleagrams, np.expand_dims(X_ft, axis=2)), axis=2)
    
    last_sample_t = bin_size*(np.size(X_ft, 1)-1) # time (ms) corresponding to the final time bin
    t = np.linspace(0, last_sample_t, num=np.size(X_ft, 1)) # time axis
    f = np.logspace(np.log10(min_F), np.log10(max_F), n_F) # logarithmically spaced frequency axis
    
    return cochleagrams, t, f

#############################################################################################################################

def plot_stimuli(raw_sound, fs, cochleagram, bin_size, n_F, min_F, max_F):
    '''
    Parameters
    ----------
    raw_sound : raw sound pressure wave
    fs : sampling frequency of raw stimulus
    cochleagram : cochleagram of stimulus
    bin_size : cochleagram time bin size
    n_F : number of cochleagram frequency channels
    min_F : minimum centre frequency of cochleagram channels
    max_F : maximum centre frequency of cochleagram channels

    Returns
    -------
    None.
    '''
    
    last_sample_t = bin_size*(np.size(cochleagram, 1)-1)
    t = np.linspace(0, last_sample_t, num=np.size(cochleagram, 1))
    f = np.logspace(np.log10(min_F), np.log10(max_F), n_F)
    
    #t_raw = np.linspace(0, 4, num=4*fs)
    #t_raw = np.linspace(0, 1, num=4*fs)
    t_raw = np.linspace(0, len(raw_sound)/fs, num=len(raw_sound))
    
    plt.figure()
    #plt.plot(t_raw, raw_sound[:len(t_raw)])
    plt.plot(t_raw, raw_sound)
    plt.ylabel('Amplitude')
    plt.xlabel('time (s)')
    plt.show()
    
    fig, ax = plt.subplots()
    im = ax.imshow(cochleagram, aspect='auto', cmap='jet', interpolation='none', extent=extents(t) + extents(f), origin='lower')
    fig.colorbar(im, ax=ax)
    current_yticks = ax.get_yticks()
    
    fig, ax = plt.subplots()
    im = ax.imshow(cochleagram, aspect='auto', cmap='jet', interpolation='none', extent=extents(t) + extents(f), origin='lower')
    fig.colorbar(im, ax=ax)
    new_yticks = np.array([1250, 3750, 6250, 8750, 11250, 13750, 16250, 18750])
    ax.set_yticks(new_yticks)
    ax.set_yticklabels([str(round(f_chan)) for f_chan in f])

#############################################################################################################################

def get_all_responses(neurons, data_dir, num_neurons, num_stimuli, num_repeats):
    '''
    Parameters
    ----------
    neurons : units in the original dataset which will be used in the analysis (excludes non-single and excessively noisy - 
    NR > 40 - units)
    data_dir : directory containing response data files
    num_neurons : number of neurons
    num_stimuli : number of stimuli
    num_repeats : number of repeats of each stimulus
    
    Returns
    -------
    neuron_data_list : list of response data for all neurons
    
    Returns list of response data contained in directory data_dir, with dimensions (num_neurons, num_stimuli, num_repeats); 
    each element of the list is a numpy array of spike times; "neurons" specifies which units, out of the original dataset, 
    are usable for analysis
    '''
    
    neuron_data_list = []

    for nID in range(num_neurons):
        # load all spike times data for one neuron
        neuron_mat_contents = loadmat(data_dir + neurons[nID]['path'][0])
        data = neuron_mat_contents['data']
        
        stimulus_responses = []
        for sID in range(num_stimuli):
            # response data from neuron nID in response to stimulus sID, all repeats
            neuron_stimulus_repeats = data['set'][sID].repeats
            
            response_repeats_list = []
            for rID in range(num_repeats):
                # spike times from neuron nID in response to stimulus sID on repeat rID
                response_repeats_list.append(neuron_stimulus_repeats[rID].t)
            
            stimulus_responses.append(response_repeats_list)
            
        neuron_data_list.append(stimulus_responses)
        
    return neuron_data_list
        
def loadmat(filename):
    """
    FUNCTION OBTAINED FROM https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    
    this function should be called instead of direct spio.loadmat as it cures the problem of not properly recovering python 
    dictionaries from mat files. It calls the function check keys to cure all entries which are still mat-objects
    """
    
    import scipy.io as sio
    
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    """
    FUNCTION OBTAINED FROM https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries
    """
    
    import scipy.io as sio
    
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    """
    FUNCTION OBTAINED FROM https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    
    A recursive function which constructs from matobjects nested dictionaries
    """
    
    import scipy.io as sio
    
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

#############################################################################################################################

def get_stim_response_grid_concat(response_data, stim_data, stimuli, fold, his_steps, bin_size, stage, norm, mean_all, 
                                  std_all, bins_clip, bins_shift, zero_padding, padding_bins, t_axis):
    '''
    Parameters
    ----------
    response_data : list of NumPy arrays of spike times for a given neuron in response to stimuli in fold, for each repeat
    stim_data : cochleagrams of all stimuli (dimensions -> (f, t, stimuli))
    stimuli : stimuli to be used (training, validation, cross-validation, or testing)
    fold : cross-validation fold (None if the function is used for CV or test data)
    his_steps : history steps for stimulus tensorisation
    bin_size : time bin size for binning spike times, same as the one used for producing cochleagrams
    stage : stage of training-validation-testing pipeline (one of "train", "val", and "test")
    norm : parameter which determines whether cochleagrams are normalised to zero-mean and unit variance w.r.t. the mean and 
    std of a given set ('all') or w.r.t their own mean and std for each cochleagram ('each')
    mean_all : mean of a given stimulus set, only used if norm='all'
    std_all : std of a given stimulus set, only used if norm='all'
    bins_clip : number of bins to be clipped from the start of the responses
    bins_shift : number of bins to be shifted in the responses compared to the cochleagrams to account for neuronal latency
    zero_padding : boolean variable, controls zero-padding
    padding_bins : number of padding steps for zero-padding, if applicable
    t_axis : time axis used to compute cochleagrams

    Returns
    -------
    stim_grid : NumPy array of concatenated normalised and tensorised cochleagrams of desired stimuli; dimensions -> 
    (f, history, concatenated t)
    response_grid : NumPy array of concatenated binned spike counts (PSTHs) of responses to desired stimuli, for a given 
    neuron; if stage is "train", the PSTHs are averaged over repeats, and the dimensions are (concatenated t); if it is 
    "val" or "test", the PSTHs are computed for each repeat, and the dimensions are (concatenated t, repeats), to enable
    CCnorm calculation
    
    Returns NumPy arrays of stimulus and response data, for a given neuron and desired stimuli, concatenated along their 
    time dimension
    '''
    
    import numpy as np
    from tensorize_mod import tensorize_mod
    
    response_grid = np.array([])
    stim_grid = np.array([])
    
    # time axis used for computing cochleagrams, to use for binning responses into PSTHs
    t = t_axis
    
    for stim in range(len(response_data)): # loop over stimuli
        if fold is None: # if fold is None, stimuli are either the complete CV set or the test set  
            X_ft = stim_data[:, :, stimuli[stim]]
        else:
            X_ft = stim_data[:, :, stimuli[fold][stim]]

        # stimulus normalisation to have zero mean and unit variance
        if norm == 'all': # if "all", normalise stimuli using whole set (usually, cross-validation set)
            X_ft_normalized = (X_ft - mean_all) / std_all
        elif norm == 'each': # if "each", normalise stimuli stimulus-by-stimulus
            if zero_padding: # if stimuli are zero-padded, exclude padding_bins from mean and std calculations
                X_ft_normalized = (X_ft - np.mean(X_ft[:, padding_bins:][:])) / np.std(X_ft[:, padding_bins:][:], ddof=1)
            else:
                X_ft_normalized = (X_ft - np.mean(X_ft[:])) / np.std(X_ft[:], ddof=1)
        
        # stimulus tensorisation; X_fht dimensions -> (f, history, t); X_fht.shape[2] = X_ft.shape[1] - (his_steps-1)
        # if X_ft is zero-padded, and padding_bins = his_steps-1, the bins removed for tensorisations are the zero-padding 
        # ones
        X_fht = tensorize_mod(X_ft_normalized, his_steps)

        # if statement is needed because X_fht[:, :, :-0] does not work
        if bins_shift != 0:
            X_fht = X_fht[:, :, :-bins_shift] # remove final bins of stimulus to account for latency
        
        # concatenate cochleagrams of different stimuli along their time dimension
        if stim_grid.shape[0] == 0: # if statement is necessary for concatenation - must share concatenated dimension
            stim_grid = X_fht
        else:
            stim_grid = np.concatenate((stim_grid, X_fht), axis=-1)

        # list of spike time arrays in response to a given stimulus for each repeat, for a given neuron
        spike_times = response_data[stim]
        # ensure all entries in spike times list are lists
        spike_times_1 = [[o] if type(o) is float else o for o in spike_times] 
        
        # build PSTH grid to be used in training: responses are averaged over repeats
        if stage == 'train':
            all_spiketimes = [j for sub in spike_times_1 for j in sub] # flatten spike times list, remove repeat dimension
            
            response = np.histogram(all_spiketimes, t)[0] # get PSTH of responses to all repeats of stimulus, for one neuron
            response = response[bins_clip:] # clip desired number of time bins from the start of the response
            response = response[bins_shift:] # remove initial bins of response to account for latency
            response_av = response[:]/len(spike_times) # divide by number of repeats to get average spike count
            
            if not zero_padding: # if stimuli were not zero-padded, clip first his_steps-1 bins to match cochleagram length
                response_av = response_av[his_steps-1:]
            
            # concatenate PSTHs to different stimuli along their time dimension
            response_grid = np.concatenate((response_grid, response_av))
        
        # build PSTH grid to be used in validation or testing: response repeats are kept separate, and not averaged, to be 
        # able to compute ccnorm
        elif stage == 'test' or stage == 'val': 
            stim_repeats_resp = np.array([])
            
            for rID in range(len(spike_times_1)): # loop over repeats
                # array of spike times in response to a repeat of a given stimulus, for a given neuron
                spiketimes_rep = spike_times_1[rID]
                
                # get PSTH of responses to one repeat of stimulus, for one neuron
                response_rep = np.histogram(spiketimes_rep, t)[0]
                response_rep = response_rep[bins_clip:] # clip desired number of time bins from the start of the response
                response_rep = response_rep[bins_shift:] # remove initial bins of response to account for latency
                
                # if stimuli were not zero-padded, clip first his_steps-1 bins to match cochleagram length
                if not zero_padding:
                    response_rep = response_rep[his_steps-1:]

                # if statement is necessary for concatenation - must share concatenated dimension
                # concatenate PSTHs along their repeat dimension (dimensions -> (t, repeats))
                if stim_repeats_resp.shape[0] == 0:
                    stim_repeats_resp = np.expand_dims(response_rep, axis=1)
                else:
                    stim_repeats_resp = np.concatenate((stim_repeats_resp, np.expand_dims(response_rep, axis=1)), axis=1)
                  
            # if statement is necessary for concatenation - must share concatenated dimension
            # concatenate PSTHs to different stimuli along their time dimension (dimensions -> (concatenated t, repeats))
            if response_grid.shape[0] == 0:
                response_grid = stim_repeats_resp
            else:
                response_grid = np.concatenate((response_grid, stim_repeats_resp), axis=0)
            
        else:
            # only accept "train", "val", and "test" as stages
            raise ValueError('"stage" must be one of: "train", "val", and "test"')
        
    return stim_grid, response_grid

def get_stim_response_grid(response_data, stim_data, stimuli, fold, his_steps, bin_size, stage, norm, mean_all, std_all, 
                           bins_clip, bins_shift, zero_padding, padding_bins, t_axis):
    '''
    Parameters
    ----------
    response_data : list of NumPy arrays of spike times for a given neuron in response to stimuli in fold, for each repeat
    stim_data : cochleagrams of all stimuli (dimensions -> (f, t, stimuli))
    stimuli : stimuli to be used (training, validation, cross-validation, or testing)
    fold : cross-validation fold (None if the function is used for CV or test data)
    his_steps : history steps for stimulus tensorisation
    bin_size : time bin size for binning spike times, same as the one used for producing cochleagrams
    stage : stage of training-validation-testing pipeline (one of "train", "val", and "test")
    norm : parameter which determines whether cochleagrams are normalised to zero-mean and unit variance w.r.t. the mean and 
    std of a given set ('all') or w.r.t their own mean and std for each cochleagram ('each')
    mean_all : mean of a given stimulus set, only used if norm='all'
    std_all : std of a given stimulus set, only used if norm='all'
    bins_clip : number of bins to be clipped from the start of the responses
    bins_shift : number of bins to be shifted in the responses compared to the cochleagrams to account for neuronal latency
    zero_padding : boolean variable, controls zero-padding
    padding_bins : number of padding steps for zero-padding, if applicable
    t_axis : time axis used to compute cochleagrams

    Returns
    -------
    stim_grid : NumPy array of normalised and tensorised cochleagrams of desired stimuli; dimensions -> (stimuli, f, history, 
    t)
    response_grid : NumPy array of binned spike counts (PSTHs) of responses to desired stimuli, for a given neuron; if stage 
    is "train", the PSTHs are averaged over repeats, and the dimensions are (stimuli, t); if it is "val" or "test", the 
    PSTHs are computed for each repeat, and the dimensions are (stimuli, t, repeats), to enable CCnorm calculation
    
    Returns NumPy arrays of stimulus and response data, for a given neuron and desired stimuli
    '''
    
    import numpy as np
    from tensorize_mod import tensorize_mod
    
    response_grid = np.array([])
    stim_grid = np.array([])
    
    # time axis used for computing cochleagrams, to use for binning responses into PSTHs
    t = t_axis
    
    for stim in range(len(response_data)): # loop over stimuli
        if fold is None: # if fold is None, stimuli are either the complete CV set or the test set  
            X_ft = stim_data[:, :, stimuli[stim]]
        else:
            X_ft = stim_data[:, :, stimuli[fold][stim]]

        # stimulus normalisation to have zero mean and unit variance
        if norm == 'all': # if "all", normalise stimuli using whole set (usually, cross-validation set)
            X_ft_normalized = (X_ft - mean_all) / std_all
        elif norm == 'each': # if "each", normalise stimuli stimulus-by-stimulus
            if zero_padding: # if stimuli are zero-padded, exclude padding_bins from mean and std calculations
                X_ft_normalized = (X_ft - np.mean(X_ft[:, padding_bins:][:])) / np.std(X_ft[:, padding_bins:][:], ddof=1)
            else:
                X_ft_normalized = (X_ft - np.mean(X_ft[:])) / np.std(X_ft[:], ddof=1)
        
        # stimulus tensorisation; X_fht dimensions -> (f, history, t); X_fht.shape[2] = X_ft.shape[1] - (his_steps-1)
        # if X_ft is zero-padded, and padding_bins = his_steps-1, the bins removed for tensorisations are the zero-padding 
        # ones
        X_fht = tensorize_mod(X_ft_normalized, his_steps)

        # if statement is needed because X_fht[:, :, :-0] does not work
        if bins_shift != 0:
            X_fht = X_fht[:, :, :-bins_shift] # remove final bins of stimulus to account for latency
        
        # concatenate cochleagrams of different stimuli
        if stim_grid.shape[0] == 0: # if statement is necessary for concatenation - must share concatenated dimension
            stim_grid = np.expand_dims(X_fht, axis=0) # add stimulus dimension
        else:
            # concatenate along stimulus dimension
            stim_grid = np.concatenate((stim_grid, np.expand_dims(X_fht, axis=0)), axis=0)

        # list of spike time arrays in response to a given stimulus for each repeat, for a given neuron
        spike_times = response_data[stim]
        # ensure all entries in spike times list are lists
        spike_times_1 = [[o] if type(o) is float else o for o in spike_times] 
        
        # build PSTH grid to be used in training: responses are averaged over repeats
        if stage == 'train':
            all_spiketimes = [j for sub in spike_times_1 for j in sub] # flatten spike times list, remove repeat dimension
            
            response = np.histogram(all_spiketimes, t)[0] # get PSTH of responses to all repeats of stimulus, for one neuron
            response = response[bins_clip:] # clip desired number of time bins from the start of the response
            response = response[bins_shift:] # remove initial bins of response to account for latency
            response_av = response[:]/len(spike_times) # divide by number of repeats to get average spike count
            
            if not zero_padding: # if stimuli were not zero-padded, clip first his_steps-1 bins to match cochleagram length
                response_av = response_av[his_steps-1:]
            
            # concatenate PSTHs to different stimuli
            if response_grid.shape[0] == 0: # if statement is necessary for concatenation - must share concatenated dimension
                response_grid = np.expand_dims(response_av, axis=0) # add stimulus dimension
            else:
                # concatenate along stimulus dimension
                response_grid = np.concatenate((response_grid, np.expand_dims(response_av, axis=0)), axis=0)
        
        # build PSTH grid to be used in validation or testing: response repeats are kept separate, and not averaged, to be 
        # able to compute ccnorm
        elif stage == 'test' or stage == 'val': 
            stim_repeats_resp = np.array([])
            
            for rID in range(len(spike_times_1)): # loop over repeats
                # array of spike times in response to a repeat of a given stimulus, for a given neuron
                spiketimes_rep = spike_times_1[rID]
                
                # get PSTH of responses to one repeat of stimulus, for one neuron
                response_rep = np.histogram(spiketimes_rep, t)[0]
                response_rep = response_rep[bins_clip:] # clip desired number of time bins from the start of the response
                response_rep = response_rep[bins_shift:] # remove initial bins of response to account for latency
                
                # if stimuli were not zero-padded, clip first his_steps-1 bins to match cochleagram length
                if not zero_padding:
                    response_rep = response_rep[his_steps-1:]

                # if statement is necessary for concatenation - must share concatenated dimension
                # concatenate PSTHs along their repeat dimension (dimensions -> (t, repeats))
                if stim_repeats_resp.shape[0] == 0:
                    stim_repeats_resp = np.expand_dims(response_rep, axis=1) # add repeat dimension
                else:
                    # concatenate along repeat dimension
                    stim_repeats_resp = np.concatenate((stim_repeats_resp, np.expand_dims(response_rep, axis=1)), axis=1)
                  
            # if statement is necessary for concatenation - must share concatenated dimension
            # concatenate PSTHs to different stimuli (dimensions -> (stimuli, t, repeats))
            if response_grid.shape[0] == 0:
                response_grid = np.expand_dims(stim_repeats_resp, axis=0) # add stimulus dimension
            else:
                # concatenate along stimulus dimension
                response_grid = np.concatenate((response_grid, np.expand_dims(stim_repeats_resp, axis=0)), axis=0)
            
        else:
            # only accept "train", "val", and "test" as stages
            raise ValueError('"stage" must be one of: "train", "val", and "test"')
        
    return stim_grid, response_grid

#############################################################################################################################

def get_psth(spiketimes, t):
    '''
    Parameters
    ----------
    spiketimes : list of spike times
    t : time axis for binning spike times

    Returns
    -------
    response : PSTH of spiketimes, binned using t
    spikes_after_end : number of spikes recorded after the last time point in t
    '''
    
    spikes_all = len(spiketimes) # number of spikes in spiketimes

    spikes_before = len(spiketimes) # number of spikes after discarding those after the last time point in t
    spikes_after_end = spikes_all - spikes_before # number of spikes recorded after the stimulus has ended
    
    response = np.histogram(spiketimes, t)[0] # PSTH of all spike times over one repeat
    
    return response, spikes_after_end

#############################################################################################################################

def get_noise_ratio_NS2(psths):
    '''
    Parameters
    ----------
    psths : numpy array of PSTHs of responses for a given neuron, separated by stimulus and repeat (dimensions are (stimuli, 
    repeats, t))
    
    Returns
    -------
    noise_ratios : numpy array of noise ratios for a given neuron, for each stimulus
    noise_ratio_all : noise ratio for a given neuron, calculated over all stimuli
    '''
    
    import numpy as np
    
    n_stimuli = np.size(psths, 0)
    n_repeats = np.size(psths, 1)
    
    noise_ratios = np.zeros((n_stimuli))
    
    for stim in range(n_stimuli): # loop over stimuli
        stim_psths = psths[stim, :, :] # PSTHs in response to a given stimulus, for a given neuron
        
        # total power of response PSTHs (average power in response to each repeat)
        total_power = np.mean(np.var(stim_psths, axis=1, ddof=1))
        # signal power of response PSTHs (power in average response to all repeats)
        signal_power = (1/(n_repeats - 1))*(n_repeats*np.var(np.mean(stim_psths, 0), ddof=1) - total_power)
        
        noise_ratios[stim] = (total_power - signal_power)/signal_power # noise ratio: noise power over signal power
    
    psths_tr = np.transpose(psths, [1, 0, 2]) # change the PSTH array to have dimensions (repeats, stimuli, time)
    # reshape the PSTH by concatenating stimuli over time (dimensions (repeats, stimuli*time))
    psths = np.reshape(psths_tr, (n_repeats, -1)) 
    
    total_power_all = np.mean(np.var(psths, axis=1, ddof=1)) # total power of response PSTHs (see above)
    
    # signal power of response PSTHs (see above)
    signal_power_all = (1/(n_repeats - 1))*(n_repeats*np.var(np.mean(psths, 0), ddof=1) - total_power_all)
    # if signal power is negative, set it to NaN
    if signal_power_all < 0:
        signal_power_all = np.nan
    
    noise_ratio_all = (total_power_all - signal_power_all)/signal_power_all # noise ratio (see above)

    return noise_ratios, noise_ratio_all

#############################################################################################################################

def get_psths(response_data, t):
    '''
    Parameters
    ----------
    response_data : numpy array containing spike trains binned at 1 ms for a given neuron, for all stimuli and their repeats
    t : time axis used to produce cochleagrams, which will also be used to produce PSTHs

    Returns
    -------
    psths : PSTHs of responses for a given neuron, dimensions are (stimuli, repeats, time)
    average_psth : PSTHs of responses for a given neuron, averaged over repeats (dimensions are (stimuli, time))

    Returns PSTHs of responses of a neuron to all stimuli; returns both PSTHs for separate repeats and averaged over repeats
    '''
    
    import numpy as np
    
    n_stimuli = np.size(response_data, 0)
    n_repeats = np.size(response_data, 1)
    
    psths = np.zeros((n_stimuli, n_repeats, np.size(t, 0)-1))
    
    for stim in range(n_stimuli): # loop over stimuli
        raw_responses = response_data[stim, :, :] # array of spike trains in response to a given stimulus, for a given neuron
        
        all_spiketimes = []
        
        for rID in range(n_repeats): # loop over repeats
            # find spike times (in ms) from spike train (add 1 to adjust for Python indexing)
            # this only works like this if spike trains are binned at 1 ms
            spiketimes_rep = np.where(raw_responses[rID, :] != 0)[0] + 1
            # get spike times over repeats - used to calculated the repeat-averaged PSTH later on
            all_spiketimes.append(spiketimes_rep)
            
            response_psth = np.histogram(spiketimes_rep, t)[0] # PSTH of all spike times over one repeat
            
            psths[stim, rID, :] = response_psth
            
        all_spiketimes = [j for sub in all_spiketimes for j in sub] # flatten list of spike time arrays for each repeat
        
        average_psth = np.histogram(all_spiketimes, t)[0] # PSTH of all spike times over all repeats
        average_psth = np.append(average_psth[:]/n_repeats, 0) # divide by number of repeats to get average spike count
    
    return psths, average_psth

#############################################################################################################################

def get_spiketimes(response_data):
    '''
    Parameters
    ----------
    response_data : numpy array containing spike trains for a given neuron, for all stimuli and their repeats
    
    Returns
    -------
    spiketimes : list of numpy arrays of spike times for a given neuron, for all stimuli and their repeats
    
    Returns list of spike times of responses of a neuron to each repeat of each stimulus
    '''
    
    import numpy as np
    
    n_stimuli = np.size(response_data, 0)
    n_repeats = np.size(response_data, 1)
    
    spiketimes = []
    
    for stim in range(n_stimuli): # loop over stimuli
        raw_responses = response_data[stim, :, :] # array of spike trains in response to a given stimulus, for a given neuron
        
        spiketimes_stim = []
        
        for rID in range(n_repeats): # loop over repeats
            # find spike times from spike train (add 1 to adjust for Python indexing)
            spiketimes_rep = np.where(raw_responses[rID, :] != 0)[0] + 1
            
            spiketimes_stim.append(spiketimes_rep)
            
        spiketimes.append(spiketimes_stim)
    
    return spiketimes

#############################################################################################################################

def build_grid_NS1_NS2_concat(neuron_data_list, stimuli_cochleagrams, stimuli, his_steps, bin_size, norm, mean_all, std_all, 
                              clip_bins, shift_bins, type_data, zero_padding, padding_bins, t_axis):
    '''
    Parameters
    ----------
    neuron_data_list : list of NumPy arrays of spike times (dimensions -> neurons x stimuli x repeats)
    stimuli_cochleagrams : NumPy array of cochleagrams (dimensions -> f, t, stimuli)
    stimuli : dictionary containing training, validation, cross-validation and testing stimuli
    his_steps : number of history steps for tensorisation
    bin_size : time step used to bin stimuli and responses
    norm : type of stimulus normalisation, either 'all' or 'each'
    mean_cv : mean over all stimuli in the cross-validation set
    std_cv : standard deviation over all stimuli in the cross-validation set
    clip_bins : number of bins to be clipped from the start of the stimuli and responses
    shift_bins : number of bins that the response is to be shifted by with respect to the stimuli
    type_data : determines whether datasets are to be built as NumPy arrays or torch tensors
    zero_padding : boolean variable, indicates whether stimuli were zero-padded
    padding_bins : number of bins added to zero-padded stimuli
    t_axis : time axis used to compute cochleagrams

    Returns
    -------
    train_stim_dataset : torch tensor/NumPy array of time-concatenated tensorised stimuli in training sets 
    (dimensions -> folds x f x history x concatenated t)
    train_resp_dataset : torch tensor/NumPy array of time-concatenated repeat-averaged responses to stimuli in training sets 
    (dimensions -> neurons x folds x concatenated t)
    val_stim_dataset : torch tensor/NumPy array of time-concatenated tensorised stimuli in val sets 
    (dimensions -> folds x f x history x concatenated t)
    val_resp_dataset : torch tensor/NumPy array of time-concatenated repeat-averaged responses to stimuli in val sets 
    (dimensions -> neurons x folds x concatenated t)
    cv_stim_dataset : torch tensor/NumPy array of time-concatenated tensorised stimuli in cross-val set 
    (dimensions -> f x history x concatenated t)
    cv_resp_dataset : torch tensor/NumPy array of time-concatenated repeat-averaged responses to stimuli in cross-val set 
    (dimensions -> neurons x concatenated t)
    test_stim_dataset : torch tensor/NumPy array of time-concatenated tensorised stimuli in test set 
    (dimensions -> f x history x concatenated t)
    test_resp_dataset : torch tensor/NumPy array of time-concatenated repeat-averaged responses to stimuli in test set 
    (dimensions -> neurons x concatenated t)
    '''
    
    folds = len(stimuli['training_stimuli'])
    training_stimuli = stimuli['training_stimuli']
    validation_stimuli = stimuli['validation_stimuli']
    cv_stimuli = stimuli['cv_stimuli']
    test_stimuli = stimuli['test_stimuli']
    
    train_stim_dataset = torch.tensor([], device=device)
    train_resp_dataset = torch.tensor([], device=device)

    val_stim_dataset = torch.tensor([], device=device)
    val_resp_dataset = torch.tensor([], device=device)
    
    cv_stim_dataset = torch.tensor([], device=device)
    cv_resp_dataset = torch.tensor([], device=device)
    
    test_stim_dataset = torch.tensor([], device=device)
    test_resp_dataset = torch.tensor([], device=device)
    
    for nID in range(len(neuron_data_list)): # loop over neurons
        print("nID = ")
        print(nID)
        
        neuron_train_resp_data = torch.tensor([], device=device)
        neuron_val_resp_data = torch.tensor([], device=device)
        
        for fID in range(folds): # loop over cross-validation folds
            # list of spike time responses of neuron to training stimuli in fold, for each repeat
            training_data = [neuron_data_list[nID][stim] for stim in training_stimuli[fID]]
            
            # get stimulus and response grids; stimulus grid contains training set cochleagrams (dimensions -> (f, history,
            # concatenated t)); response grid contains PSTHs of responses in training set averaged over repeats 
            # (dimensions -> concatenated t))
            train_X_fht_grid, train_resp_grid = get_stim_response_grid_concat(training_data, stimuli_cochleagrams, 
                                                                              training_stimuli, fID, his_steps, bin_size, 
                                                                              'train', norm, mean_all, std_all, clip_bins, 
                                                                              shift_bins, zero_padding, padding_bins, t_axis)
            
            # convert NumPy array grids to torch tensors
            train_X_fht_grid = torch.tensor(train_X_fht_grid, device=device)
            train_resp_grid = torch.tensor(train_resp_grid, device=device)
            
            # stimulus grid does not depend on neurons - only need to put in dataset tensor the first time it is computed
            if nID == 0:
                # append grid to tensor containing whole stimulus dataset for each fold
                train_stim_dataset = torch.cat((train_stim_dataset, train_X_fht_grid.unsqueeze(0)), dim=0)
            
            # append grid to tensor of response dataset for each fold, for one neuron
            neuron_train_resp_data = torch.cat((neuron_train_resp_data, train_resp_grid.unsqueeze(0)), dim=0)
            
            
            # list of spike time responses of neuron to validation stimuli in fold, for each repeat
            val_data = [neuron_data_list[nID][stim] for stim in validation_stimuli[fID]]
            
            # get stimulus and response grids; stimulus grid contains validation set cochleagrams (dimensions -> (f, history
            # concatenated t)); response grid contains PSTHs of responses in validation set averaged over repeats 
            # (dimensions -> concatenated t))
            val_X_fht_grid, val_resp_grid = get_stim_response_grid_concat(val_data, stimuli_cochleagrams, validation_stimuli, 
                                                                          fID, his_steps, bin_size, 'train', norm, mean_all, 
                                                                          std_all, clip_bins, shift_bins, zero_padding, 
                                                                          padding_bins, t_axis)
            
            # convert NumPy array grids to torch tensors
            val_X_fht_grid = torch.tensor(val_X_fht_grid, device=device)
            val_resp_grid =  torch.tensor(val_resp_grid, device=device)
            
            # stimulus grid does not depend on neurons - only need to put in dataset tensor the first time it is computed
            if nID == 0:
                # append grid to tensor containing whole stimulus dataset for each fold
                val_stim_dataset = torch.cat((val_stim_dataset, val_X_fht_grid.unsqueeze(0)), dim=0)
            
            # append grid to tensor of response dataset for each fold, for one neuron
            neuron_val_resp_data = torch.cat((neuron_val_resp_data, val_resp_grid.unsqueeze(0)), dim=0)
        
        # append training and validation response tensors for one neuron to tensor of responses for each neuron
        train_resp_dataset = torch.cat((train_resp_dataset, neuron_train_resp_data.unsqueeze(0)), dim=0)
        val_resp_dataset = torch.cat((val_resp_dataset, neuron_val_resp_data.unsqueeze(0)), dim=0)
        

        # list of spike time responses of neuron to cross-validation stimuli, for each repeat
        cv_data = [neuron_data_list[nID][stim] for stim in cv_stimuli]
        
        # get stimulus and response grids; stimulus grid contains cross-validation set cochleagrams (dimensions -> (f, 
        # history, concatenated t)); response grid contains PSTHs of responses in cross-validation set averaged over repeats
        # (dimensions -> concatenated t))
        cv_X_fht_grid, cv_resp_grid = get_stim_response_grid_concat(cv_data, stimuli_cochleagrams, cv_stimuli, None, 
                                                                    his_steps, bin_size, 'train', norm, mean_all, std_all, 
                                                                    clip_bins, shift_bins, zero_padding, padding_bins, 
                                                                    t_axis)
        
        # stimulus grid does not depend on neurons - only need to put in dataset tensor the first time it is computed
        if nID == 0:
            cv_stim_dataset = torch.tensor(cv_X_fht_grid, device=device)
        
        cv_resp_grid = torch.tensor(cv_resp_grid, device=device)
        
        # append grid to tensor of response dataset for each neuron
        cv_resp_dataset = torch.cat((cv_resp_dataset, cv_resp_grid.unsqueeze(0)), dim=0)
        
        
        # list of spike time responses of neuron to test stimuli, for each repeat
        test_data = [neuron_data_list[nID][stim] for stim in test_stimuli]
        
        # get stimulus and response grids; stimulus grid contains test set cochleagrams (dimensions -> (f, 
        # history, concatenated t)); response grid contains PSTHs of responses in test set averaged over repeats
        # (dimensions -> concatenated t))
        test_X_fht_grid, test_resp_grid = get_stim_response_grid_concat(test_data, stimuli_cochleagrams, test_stimuli, None, 
                                                                        his_steps, bin_size, 'train', norm, mean_all, 
                                                                        std_all, clip_bins, shift_bins, zero_padding, 
                                                                        padding_bins, t_axis)
        
        # stimulus grid does not depend on neurons - only need to put in dataset tensor the first time it is computed
        if nID == 0:
            test_stim_dataset = torch.tensor(test_X_fht_grid, device=device)
            
        test_resp_grid = torch.tensor(test_resp_grid, device=device)
        
        # append grid to tensor of response dataset for each neuron
        test_resp_dataset = torch.cat((test_resp_dataset, test_resp_grid.unsqueeze(0)), dim=0)
    
    # convert data to suitable torch float type
    train_stim_dataset = train_stim_dataset.to(torch.float32)
    train_resp_dataset = train_resp_dataset.to(torch.float32)
    val_stim_dataset = val_stim_dataset.to(torch.float32)
    val_resp_dataset = val_resp_dataset.to(torch.float32)
    cv_stim_dataset = cv_stim_dataset.to(torch.float32)
    cv_resp_dataset = cv_resp_dataset.to(torch.float32)
    test_stim_dataset = test_stim_dataset.to(torch.float32)
    test_resp_dataset = test_resp_dataset.to(torch.float32)
    
    if type_data == 'numpy': # if data is to be used as NumPy arrays, convert it back to that format
        train_stim_dataset = train_stim_dataset.detach().cpu().numpy()
        train_resp_dataset = train_resp_dataset.detach().cpu().numpy()
        val_stim_dataset = val_stim_dataset.detach().cpu().numpy()
        val_resp_dataset = val_resp_dataset.detach().cpu().numpy()
        cv_stim_dataset = cv_stim_dataset.detach().cpu().numpy()
        cv_resp_dataset = cv_resp_dataset.detach().cpu().numpy()
        test_stim_dataset = test_stim_dataset.detach().cpu().numpy()
        test_resp_dataset = test_resp_dataset.detach().cpu().numpy()
    
    return train_stim_dataset, train_resp_dataset, val_stim_dataset, val_resp_dataset, cv_stim_dataset, cv_resp_dataset, \
        test_stim_dataset, test_resp_dataset
        
def build_grid_NS1_NS2(neuron_data_list, stimuli_cochleagrams, stimuli, his_steps, bin_size, norm, mean_all, std_all, 
                       clip_bins, shift_bins, type_data, zero_padding, padding_bins, t_axis):
    '''
    Parameters
    ----------
    neuron_data_list : list of NumPy arrays of spike times (dimensions -> neurons x stimuli x repeats)
    stimuli_cochleagrams : NumPy array of cochleagrams (dimensions -> f, t, stimuli)
    stimuli : dictionary containing training, validation, cross-validation and testing stimuli
    his_steps : number of history steps for tensorisation
    bin_size : time step used to bin stimuli and responses
    norm : type of stimulus normalisation, either 'all' or 'each'
    mean_cv : mean over all stimuli in the cross-validation set
    std_cv : standard deviation over all stimuli in the cross-validation set
    clip_bins : number of bins to be clipped from the start of the stimuli and responses
    shift_bins : number of bins that the response is to be shifted by with respect to the stimuli
    type_data : determines whether datasets are to be built as NumPy arrays or torch tensors
    zero_padding : boolean variable, indicates whether stimuli were zero-padded
    padding_bins : number of bins added to zero-padded stimuli
    t_axis : time axis used to compute cochleagrams

    Returns
    -------
    train_stim_dataset : torch tensor/NumPy array of tensorised stimuli in training sets 
    (dimensions -> folds x stimuli f x history x t)
    train_resp_dataset : torch tensor/NumPy array of repeat-averaged responses to stimuli in training sets 
    (dimensions -> neurons x folds x stimuli x t)
    val_stim_dataset : torch tensor/NumPy array of tensorised stimuli in val sets 
    (dimensions -> folds x stimuli x f x history x t)
    val_resp_dataset : torch tensor/NumPy array of repeat-averaged responses to stimuli in val sets 
    (dimensions -> neurons x folds x stimuli x t)
    cv_stim_dataset : torch tensor/NumPy array of tensorised stimuli in cross-val set 
    (dimensions -> stimuli x f x history x t)
    cv_resp_dataset : torch tensor/NumPy array of repeat-averaged responses to stimuli in cross-val set 
    (dimensions -> neurons x stimuli x t)
    test_stim_dataset : torch tensor/NumPy array of tensorised stimuli in test set 
    (dimensions -> stimuli x f x history x t)
    test_resp_dataset : torch tensor/NumPy array of repeat-averaged responses to stimuli in test set 
    (dimensions -> neurons x stimuli x t)
    '''
    
    folds = len(stimuli['training_stimuli'])
    training_stimuli = stimuli['training_stimuli']
    validation_stimuli = stimuli['validation_stimuli']
    cv_stimuli = stimuli['crossval_stimuli']
    test_stimuli = stimuli['test_stimuli']
    
    train_stim_dataset = torch.tensor([], device=device)
    train_resp_dataset = torch.tensor([], device=device)

    val_stim_dataset = torch.tensor([], device=device)
    val_resp_dataset = torch.tensor([], device=device)
    
    cv_stim_dataset = torch.tensor([], device=device)
    cv_resp_dataset = torch.tensor([], device=device)
    
    test_stim_dataset = torch.tensor([], device=device)
    test_resp_dataset = torch.tensor([], device=device)
    
    for nID in range(len(neuron_data_list)): # loop over neurons
        print("nID = ")
        print(nID)
        
        neuron_train_resp_data = torch.tensor([], device=device)
        neuron_val_resp_data = torch.tensor([], device=device)
        
        for fID in range(folds): # loop over cross-validation folds
            # list of spike time responses of neuron to training stimuli in fold, for each repeat
            training_data = [neuron_data_list[nID][stim] for stim in training_stimuli[fID]]
            
            # get stimulus and response grids; stimulus grid contains training set cochleagrams (dimensions -> (stimuli, f, 
            # history, t)); response grid contains PSTHs of responses in training set averaged over repeats (dimensions -> 
            # (stimuli, t))
            train_X_fht_grid, train_resp_grid = get_stim_response_grid(training_data, stimuli_cochleagrams, training_stimuli, 
                                                                       fID, his_steps, bin_size, 'train', norm, mean_all, 
                                                                       std_all, clip_bins, shift_bins, zero_padding, 
                                                                       padding_bins, t_axis)
            
            # convert NumPy array grids to torch tensors
            train_X_fht_grid = torch.tensor(train_X_fht_grid, device=device)
            train_resp_grid = torch.tensor(train_resp_grid, device=device)
            
            # stimulus grid does not depend on neurons - only need to put in dataset tensor the first time it is computed
            if nID == 0:
                # append grid to tensor containing whole stimulus dataset for each fold
                train_stim_dataset = torch.cat((train_stim_dataset, train_X_fht_grid.unsqueeze(0)), dim=0)
            
            # append grid to tensor of response dataset for each fold, for one neuron
            neuron_train_resp_data = torch.cat((neuron_train_resp_data, train_resp_grid.unsqueeze(0)), dim=0)
            
            
            # list of spike time responses of neuron to validation stimuli in fold, for each repeat
            val_data = [neuron_data_list[nID][stim] for stim in validation_stimuli[fID]]
            
            # get stimulus and response grids; stimulus grid contains validation set cochleagrams (dimensions -> (stimuli, f, 
            # history, t)); response grid contains PSTHs of responses in validation set averaged over repeats (dimensions -> 
            # (stimuli, t))
            val_X_fht_grid, val_resp_grid = get_stim_response_grid(val_data, stimuli_cochleagrams, validation_stimuli, fID, 
                                                                   his_steps, bin_size, 'train', norm, mean_all, std_all, 
                                                                   clip_bins, shift_bins, zero_padding, padding_bins, t_axis)
            
            # convert NumPy array grids to torch tensors
            val_X_fht_grid = torch.tensor(val_X_fht_grid, device=device)
            val_resp_grid =  torch.tensor(val_resp_grid, device=device)
            
            # stimulus grid does not depend on neurons - only need to put in dataset tensor the first time it is computed
            if nID == 0:
                # append grid to tensor containing whole stimulus dataset for each fold
                val_stim_dataset = torch.cat((val_stim_dataset, val_X_fht_grid.unsqueeze(0)), dim=0)
            
            # append grid to tensor of response dataset for each fold, for one neuron
            neuron_val_resp_data = torch.cat((neuron_val_resp_data, val_resp_grid.unsqueeze(0)), dim=0)
        
        # append training and validation response tensors for one neuron to tensor of responses for each neuron
        train_resp_dataset = torch.cat((train_resp_dataset, neuron_train_resp_data.unsqueeze(0)), dim=0)
        val_resp_dataset = torch.cat((val_resp_dataset, neuron_val_resp_data.unsqueeze(0)), dim=0)
        

        # list of spike time responses of neuron to cross-validation stimuli, for each repeat
        cv_data = [neuron_data_list[nID][stim] for stim in cv_stimuli]
        
        # get stimulus and response grids; stimulus grid contains cross-validation set cochleagrams (dimensions -> (stimuli, 
        # f, history, t)); response grid contains PSTHs of responses in cross-validation set averaged over repeats
        # (dimensions -> stimuli, t))
        cv_X_fht_grid, cv_resp_grid = get_stim_response_grid(cv_data, stimuli_cochleagrams, cv_stimuli, None, his_steps, 
                                                             bin_size, 'train', norm, mean_all, std_all, clip_bins, 
                                                             shift_bins, zero_padding, padding_bins, t_axis)
        
        # stimulus grid does not depend on neurons - only need to put in dataset tensor the first time it is computed
        if nID == 0:
            cv_stim_dataset = torch.tensor(cv_X_fht_grid, device=device)
        
        cv_resp_grid = torch.tensor(cv_resp_grid, device=device)
        
        # append grid to tensor of response dataset for each neuron
        cv_resp_dataset = torch.cat((cv_resp_dataset, cv_resp_grid.unsqueeze(0)), dim=0)
        
        
        # list of spike time responses of neuron to test stimuli, for each repeat
        test_data = [neuron_data_list[nID][stim] for stim in test_stimuli]
        
        # get stimulus and response grids; stimulus grid contains test set cochleagrams (dimensions -> (stimuli, f, history, 
        # t)); response grid contains PSTHs of responses in test set averaged over repeats (dimensions -> (stimuli, t))
        test_X_fht_grid, test_resp_grid = get_stim_response_grid(test_data, stimuli_cochleagrams, test_stimuli, None, 
                                                                 his_steps, bin_size, 'train', norm, mean_all, std_all, 
                                                                 clip_bins, shift_bins, zero_padding, padding_bins, t_axis)
        
        # stimulus grid does not depend on neurons - only need to put in dataset tensor the first time it is computed
        if nID == 0:
            test_stim_dataset = torch.tensor(test_X_fht_grid, device=device)
            
        test_resp_grid = torch.tensor(test_resp_grid, device=device)
        
        # append grid to tensor of response dataset for each neuron
        test_resp_dataset = torch.cat((test_resp_dataset, test_resp_grid.unsqueeze(0)), dim=0)
    
    # convert data to suitable torch float type
    train_stim_dataset = train_stim_dataset.to(torch.float32)
    train_resp_dataset = train_resp_dataset.to(torch.float32)
    val_stim_dataset = val_stim_dataset.to(torch.float32)
    val_resp_dataset = val_resp_dataset.to(torch.float32)
    cv_stim_dataset = cv_stim_dataset.to(torch.float32)
    cv_resp_dataset = cv_resp_dataset.to(torch.float32)
    test_stim_dataset = test_stim_dataset.to(torch.float32)
    test_resp_dataset = test_resp_dataset.to(torch.float32)
    
    if type_data == 'numpy': # if data is to be used as NumPy arrays, convert it back to that format
        train_stim_dataset = train_stim_dataset.detach().cpu().numpy()
        train_resp_dataset = train_resp_dataset.detach().cpu().numpy()
        val_stim_dataset = val_stim_dataset.detach().cpu().numpy()
        val_resp_dataset = val_resp_dataset.detach().cpu().numpy()
        cv_stim_dataset = cv_stim_dataset.detach().cpu().numpy()
        cv_resp_dataset = cv_resp_dataset.detach().cpu().numpy()
        test_stim_dataset = test_stim_dataset.detach().cpu().numpy()
        test_resp_dataset = test_resp_dataset.detach().cpu().numpy()
    
    return train_stim_dataset, train_resp_dataset, val_stim_dataset, val_resp_dataset, cv_stim_dataset, cv_resp_dataset, \
        test_stim_dataset, test_resp_dataset

#############################################################################################################################

def build_grid_NS2_include_single_NS3_concat(single_rep_cochleagrams, multi_rep_cochleagrams, n_neurons, resp_dataset, 
                                             stimuli, params, norm, mean_cv, std_cv, clip_bins, shift_bins, type_data, 
                                             zero_padding, padding_bins, t_axis):
    '''
    Parameters
    ----------
    single_rep_cochleagrams : cochleagrams of single-repeat stimuli (dimensions -> (f, t, stimuli))
    multi_rep_cochleagrams : cochleagrams of multi-repeat stimuli (dimensions -> (f, t, stimuli))
    n_neurons : number of neurons in dataset
    resp_dataset : spike time responses to each repeat of multi- and single-repeat stimuli, for each neuron (dimensions ->
    (neurons, stimuli, repeats))
    stimuli : training, validation, cross-validation, and test set stimuli
    params : dict containing cochleagram parameters and tensorisation parameter
    norm : type of stimulus normalisation, either 'all' or 'each'
    mean_cv : mean over all stimuli in the cross-validation set
    std_cv : standard deviation over all stimuli in the cross-validation set
    clip_bins : number of bins to be clipped from the start of the stimuli and responses
    shift_bins : number of bins that the response is to be shifted by with respect to the stimuli
    type_data : determines whether datasets are to be built as NumPy arrays or torch tensors
    zero_padding : boolean variable, indicates whether stimuli were zero-padded
    padding_bins : number of bins added to zero-padded stimuli
    t_axis : time axis used to compute cochleagrams

    Returns
    -------
    train_stim_dataset : torch tensor/NumPy array of time-concatenated tensorised stimuli in training sets 
    (dimensions -> folds x f x history x concatenated t)
    train_resp_dataset : torch tensor/NumPy array of time-concatenated repeat-averaged responses to stimuli in training sets 
    (dimensions -> neurons x folds x concatenated t)
    val_stim_dataset : torch tensor/NumPy array of time-concatenated tensorised stimuli in val sets 
    (dimensions -> folds x f x history x concatenated t)
    val_resp_dataset : torch tensor/NumPy array of time-concatenated repeat-averaged responses to stimuli in val sets 
    (dimensions -> neurons x folds x concatenated t)
    cv_stim_dataset : torch tensor/NumPy array of time-concatenated tensorised stimuli in crossval sets 
    (dimensions -> f x history x concatenated t)
    cv_resp_dataset : torch tensor/NumPy array of time-concatenated repeat-averaged responses to stimuli in crossval sets 
    (dimensions -> neurons x concatenated t)
    test_stim_dataset : torch tensor/NumPy array of time-concatenated tensorised stimuli in test sets 
    (dimensions -> f x history x concatenated t)
    test_resp_dataset : torch tensor/NumPy array of time-concatenated repeat-averaged responses to stimuli in test sets 
    (dimensions -> neurons x concatenated t)
    multi_rep_resps : spike time responses to each repeat of multi-repeat stimuli, for each neuron (dimensions -> (neurons, 
    stimuli, repeats))
    single_rep_resps : spike time responses to single-repeat stimuli, for each neuron (dimensions -> (neurons, stimuli, 
    repeats))
    '''
    
    training_stimuli = stimuli['training_stimuli']
    validation_stimuli = stimuli['validation_stimuli']
    cv_stimuli = stimuli['cv_stimuli']
    test_stimuli = stimuli['test_stimuli']
    folds = len(training_stimuli)
    
    bin_size = params['bin_size']
    n_F = params['n_F']
    his_steps = params['n_h']
    
    multi_rep_resps = []
    single_rep_resps = []
    for nID in range(n_neurons):
        n_multi_rep_resps = []
        n_single_rep_resps = []

        for stim in resp_dataset[nID]:
            if len(resp_dataset[nID][stim]) > 1: # if stimulus was repeated more than once, response is a multi-repeat
                n_multi_rep_resps.append(resp_dataset[nID][stim])
            else: # if stimulus was repeated only once, response is a single-repeat
                n_single_rep_resps.append(resp_dataset[nID][stim])
                
        multi_rep_resps.append(n_multi_rep_resps)
        single_rep_resps.append(n_single_rep_resps)
    
    # NumPy arrays to hold datasets
    # if stimuli were zero-padded, and padding_bins is not equal to his_steps-1 (as is the case for convolutional models), 
    # stimuli and response datasets have different (time) lengths
    
    # training stimuli, dimensions -> (folds, f, history, concatenated t)
    train_stim_dataset = np.zeros((folds, n_F, his_steps, 
                                   len(training_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))
    # validation stimuli, dimensions -> (folds, f, history, concatenated t)
    val_stim_dataset = np.zeros((folds, n_F, his_steps, 
                                 len(validation_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))
    
    if padding_bins != his_steps-1 and zero_padding is True:
        # training responses, dimensions -> (neurons, folds, concatenated t)
        train_resp_dataset = np.zeros((n_neurons, folds, 
                                       len(training_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-padding_bins)))
        # validation responses, dimensions -> (neurons, folds, concatenated t)
        val_resp_dataset = np.zeros((n_neurons, folds, 
                                     len(validation_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-padding_bins)))
        # cross-validation responses, dimensions -> (neurons, concatenated t)
        cv_resp_dataset = np.zeros((n_neurons, 
                                    (len(cv_stimuli)+np.size(single_rep_cochleagrams, 2))*(np.size(single_rep_cochleagrams, 
                                                                                                   1)-padding_bins)))
        # test responses, dimensions -> (neurons, concatenated t)
        test_resp_dataset = np.zeros((n_neurons, len(test_stimuli)*(np.size(single_rep_cochleagrams, 1)-padding_bins)))
            
    else:
        train_resp_dataset = np.zeros((n_neurons, folds, 
                                       len(training_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))
        val_resp_dataset = np.zeros((n_neurons, folds, 
                                     len(validation_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))
        cv_resp_dataset = np.zeros((n_neurons, 
                                    (len(cv_stimuli)+np.size(single_rep_cochleagrams, 2))*(np.size(single_rep_cochleagrams, 
                                                                                                   1)-his_steps+1)))
        test_resp_dataset = np.zeros((n_neurons, len(test_stimuli)*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))

    for nID in range(n_neurons):
        print('nID = ' + str(nID))
        n_train_resp_data_single = single_rep_resps[nID]
        
        # get stimulus and response grids for all single-repeat stimuli, for one neuron
        stim_dataset_single, n_resp_single = get_stim_response_grid_concat(n_train_resp_data_single, single_rep_cochleagrams, 
                                                                           training_stimuli[0][7:], None, his_steps, 
                                                                           bin_size, 'train', norm, mean_cv, std_cv, 
                                                                           clip_bins, shift_bins, zero_padding, padding_bins, 
                                                                           t_axis)
        
        for fID in range(folds):
            # responses to each repeat of each multi-repeat stimulus in training fold, for one neuron
            n_train_resp_data_multi = [multi_rep_resps[nID][stim] for stim in training_stimuli[fID][:7]]
            
            # get stimulus and response grids for training fold multi-repeat stimuli, for one neuron
            train_stim_dataset_multi, n_train_resp_multi = get_stim_response_grid_concat(n_train_resp_data_multi, 
                                                                                         multi_rep_cochleagrams, 
                                                                                         list([training_stimuli[0][:7]]), fID, 
                                                                                         his_steps, bin_size, 'train', norm, 
                                                                                         mean_cv, std_cv, clip_bins, 
                                                                                         shift_bins, zero_padding, 
                                                                                         padding_bins, t_axis)

            # training stimulus and response sets for each fold are made of all single-repeat data and fold-specific
            # multi-repeat data; multi- and single- repeat data are concatenated along their time dimension
            train_stim_dataset[fID] = np.concatenate((train_stim_dataset_multi, stim_dataset_single), axis=-1)
            train_resp_dataset[nID, fID] = np.concatenate((n_train_resp_multi, n_resp_single), axis=-1)
            
            # responses to each repeat of each multi-repeat stimulus in validation fold, for one neuron
            n_val_resp_data_multi = [multi_rep_resps[nID][stim] for stim in validation_stimuli[fID]] 
            
            # get stimulus and response grids for validation fold multi-repeat stimuli, for one neuron
            val_stim_dataset_multi, n_val_resp_multi = get_stim_response_grid_concat(n_val_resp_data_multi, 
                                                                                     multi_rep_cochleagrams, 
                                                                                     validation_stimuli, fID, his_steps, 
                                                                                     bin_size, 'train', norm, mean_cv, 
                                                                                     std_cv, clip_bins, shift_bins, 
                                                                                     zero_padding, padding_bins, t_axis)
            
            val_stim_dataset[fID] = val_stim_dataset_multi
            val_resp_dataset[nID, fID, :] = n_val_resp_multi
        
        # responses to each repeat of each multi-repeat stimulus in cross-validation set, for one neuron
        n_cv_resp_data_multi = [multi_rep_resps[nID][stim] for stim in cv_stimuli]
        
        # get stimulus and response grids for cross-validation set multi-repeat stimuli, for one neuron
        cv_stim_dataset_multi, n_cv_resp_multi = get_stim_response_grid_concat(n_cv_resp_data_multi, multi_rep_cochleagrams, 
                                                                               cv_stimuli, None, his_steps, bin_size, 
                                                                               'train', norm, mean_cv, std_cv, clip_bins, 
                                                                               shift_bins, zero_padding, padding_bins, 
                                                                               t_axis)

        # crossval stimulus and response sets are made of all single-repeat data and some multi-repeat data; multi- and 
        # single- repeat data are concatenated along their time dimension
        cv_stim_dataset = np.concatenate((cv_stim_dataset_multi, stim_dataset_single), axis=-1)
        cv_resp_dataset[nID] = np.concatenate((n_cv_resp_multi, n_resp_single), axis=-1)
        
        # responses to each repeat of each multi-repeat stimulus in test set, for one neuron
        n_test_resp_data_multi = [multi_rep_resps[nID][stim] for stim in test_stimuli]
        
        # get stimulus and response grids for cross-validation set multi-repeat stimuli, for one neuron
        test_stim_dataset, n_test_resp = get_stim_response_grid_concat(n_test_resp_data_multi, multi_rep_cochleagrams, 
                                                                       test_stimuli, None, his_steps, bin_size, 'train', 
                                                                       norm, mean_cv, std_cv, clip_bins, shift_bins, 
                                                                       zero_padding, padding_bins, t_axis)
        
        test_resp_dataset[nID] = n_test_resp
    
    # convert NumPy array datasets to torch tensors of suitable torch float type
    train_stim_dataset = torch.tensor(train_stim_dataset).to(torch.float32).to(device)
    train_resp_dataset = torch.tensor(train_resp_dataset).to(torch.float32).to(device)
    val_stim_dataset = torch.tensor(val_stim_dataset).to(torch.float32).to(device)
    val_resp_dataset = torch.tensor(val_resp_dataset).to(torch.float32).to(device)
    cv_stim_dataset = torch.tensor(cv_stim_dataset).to(torch.float32).to(device)
    cv_resp_dataset = torch.tensor(cv_resp_dataset).to(torch.float32).to(device)
    test_stim_dataset = torch.tensor(test_stim_dataset).to(torch.float32).to(device)
    test_resp_dataset = torch.tensor(test_resp_dataset).to(torch.float32).to(device)
    
    if type_data == 'numpy': # if data is to be used as NumPy arrays, convert it back to that format
        train_stim_dataset = train_stim_dataset.detach().cpu().numpy()
        train_resp_dataset = train_resp_dataset.detach().cpu().numpy()
        val_stim_dataset = val_stim_dataset.detach().cpu().numpy()
        val_resp_dataset = val_resp_dataset.detach().cpu().numpy()
        cv_stim_dataset = cv_stim_dataset.detach().cpu().numpy()
        cv_resp_dataset = cv_resp_dataset.detach().cpu().numpy()
        test_stim_dataset = test_stim_dataset.detach().cpu().numpy()
        test_resp_dataset = test_resp_dataset.detach().cpu().numpy()
    
    return train_stim_dataset, train_resp_dataset, val_stim_dataset, val_resp_dataset, cv_stim_dataset, cv_resp_dataset, \
        test_stim_dataset, test_resp_dataset, multi_rep_resps, single_rep_resps

def build_grid_NS2_include_single_NS3(single_rep_cochleagrams, multi_rep_cochleagrams, n_neurons, resp_dataset, stimuli, 
                                      params, norm, mean_cv, std_cv, clip_bins, shift_bins, type_data, zero_padding, 
                                      padding_bins, t_axis):
    '''
    Parameters
    ----------
    single_rep_cochleagrams : cochleagrams of single-repeat stimuli (dimensions -> (f, t, stimuli))
    multi_rep_cochleagrams : cochleagrams of multi-repeat stimuli (dimensions -> (f, t, stimuli))
    n_neurons : number of neurons in dataset
    resp_dataset : spike time responses to each repeat of multi- and single-repeat stimuli, for each neuron (dimensions ->
    (neurons, stimuli, repeats))
    stimuli : training, validation, cross-validation, and test set stimuli
    params : dict containing cochleagram parameters and tensorisation parameter
    norm : type of stimulus normalisation, either 'all' or 'each'
    mean_cv : mean over all stimuli in the cross-validation set
    std_cv : standard deviation over all stimuli in the cross-validation set
    clip_bins : number of bins to be clipped from the start of the stimuli and responses
    shift_bins : number of bins that the response is to be shifted by with respect to the stimuli
    type_data : determines whether datasets are to be built as NumPy arrays or torch tensors
    zero_padding : boolean variable, indicates whether stimuli were zero-padded
    padding_bins : number of bins added to zero-padded stimuli
    t_axis : time axis used to compute cochleagrams

    Returns
    -------
    train_stim_dataset : torch tensor/NumPy array of tensorised stimuli in training sets 
    (dimensions -> folds x stimuli x f x history x t)
    train_resp_dataset : torch tensor/NumPy array of repeat-averaged responses to stimuli in training sets 
    (dimensions -> neurons x folds x stimuli x t)
    val_stim_dataset : torch tensor/NumPy array of tensorised stimuli in val sets 
    (dimensions -> folds x stimuli x f x history x t)
    val_resp_dataset : torch tensor/NumPy array of repeat-averaged responses to stimuli in val sets 
    (dimensions -> neurons x folds x stimuli x t)
    multi_rep_resps : spike time responses to each repeat of multi-repeat stimuli, for each neuron (dimensions -> (neurons, 
    stimuli, repeats))
    single_rep_resps : spike time responses to single-repeat stimuli, for each neuron (dimensions -> (neurons, stimuli, 
    repeats))
    '''
    
    training_stimuli = stimuli['training_stimuli']
    validation_stimuli = stimuli['validation_stimuli']
    folds = len(training_stimuli)
    
    bin_size = params['bin_size']
    n_F = params['n_F']
    his_steps = params['n_h']
    
    multi_rep_resps = []
    single_rep_resps = []
    for nID in range(n_neurons):
        n_multi_rep_resps = []
        n_single_rep_resps = []

        for stim in resp_dataset[nID]:
            if len(resp_dataset[nID][stim]) > 1: # if stimulus was repeated more than once, response is a multi-repeat
                n_multi_rep_resps.append(resp_dataset[nID][stim])
            else: # if stimulus was repeated only once, response is a single-repeat
                n_single_rep_resps.append(resp_dataset[nID][stim])
                
        multi_rep_resps.append(n_multi_rep_resps)
        single_rep_resps.append(n_single_rep_resps)
    
    # NumPy arrays to hold datasets
    
    # training stimuli, dimensions -> (folds, training stimuli, f, history, t)
    train_stim_dataset = np.zeros((folds, len(training_stimuli[0]), n_F, his_steps, 
                                   (np.size(single_rep_cochleagrams, 1)-his_steps+1)))
    # validation stimuli, dimensions -> (folds, validation_stimuli, f, history, t)
    val_stim_dataset = np.zeros((folds, len(validation_stimuli[0]), n_F, his_steps,
                                 (np.size(single_rep_cochleagrams, 1)-his_steps+1)))
    
    # if stimuli were zero-padded, and padding_bins is not equal to his_steps-1 (as is the case for convolutional models), 
    # response datasets have different (time) lengths
    if padding_bins != his_steps-1 and zero_padding is True:
        # training responses, dimensions -> (neurons, folds, training stimuli, t)
        train_resp_dataset = np.zeros((n_neurons, folds, len(training_stimuli[0]),
                                       (np.size(single_rep_cochleagrams, 1)-padding_bins)))
        # validation responses, dimensions -> (neurons, folds, validation stimuli, t)
        val_resp_dataset = np.zeros((n_neurons, folds, len(validation_stimuli[0]),
                                     (np.size(single_rep_cochleagrams, 1)-padding_bins)))
    else:
        # training responses, dimensions -> (neurons, folds, training stimuli, t)
        train_resp_dataset = np.zeros((n_neurons, folds, len(training_stimuli[0]),
                                       (np.size(single_rep_cochleagrams, 1)-his_steps+1)))
        # validation responses, dimensions -> (neurons, folds, validation stimuli, t)
        val_resp_dataset = np.zeros((n_neurons, folds, len(validation_stimuli[0]),
                                     (np.size(single_rep_cochleagrams, 1)-his_steps+1)))
    
    for nID in range(n_neurons):
        print('nID = ' + str(nID))
        n_train_resp_data_single = single_rep_resps[nID]
        
        # get stimulus and response grids for all single-repeat stimuli, for one neuron
        train_stim_dataset_single, n_train_resp_single = get_stim_response_grid(n_train_resp_data_single, 
                                                                                single_rep_cochleagrams, 
                                                                                training_stimuli[0][7:], None, his_steps, 
                                                                                bin_size, 'train', norm, mean_cv, std_cv, 
                                                                                clip_bins, shift_bins, zero_padding, 
                                                                                padding_bins, t_axis)
        
        for fID in range(folds):
            # responses to each repeat of each multi-repeat stimulus in training fold, for one neuron
            n_train_resp_data_multi = [multi_rep_resps[nID][stim] for stim in training_stimuli[fID][:7]]
            
            # get stimulus and response grids for training fold multi-repeat stimuli, for one neuron
            train_stim_dataset_multi, n_train_resp_multi = get_stim_response_grid(n_train_resp_data_multi, 
                                                                                  multi_rep_cochleagrams, 
                                                                                  training_stimuli[:7], fID, his_steps, 
                                                                                  bin_size, 'train', norm, mean_cv, std_cv, 
                                                                                  clip_bins, shift_bins, zero_padding, 
                                                                                  padding_bins, t_axis)
            
            # training stimulus and response sets for each fold are made of all single-repeat data and fold-specific
            # multi-repeat data; multi- and single- repeat data are concatenated along their time dimension
            train_stim_dataset[fID] = np.concatenate((train_stim_dataset_multi, train_stim_dataset_single), axis=0)
            train_resp_dataset[nID, fID] = np.concatenate((n_train_resp_multi, n_train_resp_single), axis=0)
            
            # responses to each repeat of each multi-repeat stimulus in validation fold, for one neuron
            n_val_resp_data_multi = [multi_rep_resps[nID][stim] for stim in validation_stimuli[fID]] 
            
            # get stimulus and response grids for validation fold multi-repeat stimuli, for one neuron
            val_stim_dataset_multi, n_val_resp_multi = get_stim_response_grid(n_val_resp_data_multi, multi_rep_cochleagrams, 
                                                                              validation_stimuli, fID, his_steps, bin_size, 
                                                                              'train', norm, mean_cv, std_cv, clip_bins, 
                                                                              shift_bins, zero_padding, padding_bins, t_axis)
            
            val_stim_dataset[fID] = val_stim_dataset_multi
            val_resp_dataset[nID, fID] = n_val_resp_multi
    
    # convert NumPy array datasets to torch tensors of suitable torch float type
    train_stim_dataset = torch.tensor(train_stim_dataset).to(torch.float32).to(device)
    train_resp_dataset = torch.tensor(train_resp_dataset).to(torch.float32).to(device)
    val_stim_dataset = torch.tensor(val_stim_dataset).to(torch.float32).to(device)
    val_resp_dataset = torch.tensor(val_resp_dataset).to(torch.float32).to(device)
    
    if type_data == 'numpy': # if data is to be used as NumPy arrays, convert it back to that format
        train_stim_dataset = train_stim_dataset.detach().cpu().numpy()
        train_resp_dataset = train_resp_dataset.detach().cpu().numpy()
        val_stim_dataset = val_stim_dataset.detach().cpu().numpy()
        val_resp_dataset = val_resp_dataset.detach().cpu().numpy()
    
    return train_stim_dataset, train_resp_dataset, val_stim_dataset, val_resp_dataset, multi_rep_resps, single_rep_resps

#############################################################################################################################

def build_grid_NS2_include_single_NS3_sessions(single_rep_cochleagrams, multi_rep_cochleagrams, n_neurons, sess_resp_dataset, 
                                               stimuli, params, norm, mean_cv, std_cv, clip_bins, shift_bins, type_data, 
                                               zero_padding, padding_bins, t_axis):
    '''
    Parameters
    ----------
    single_rep_cochleagrams : cochleagrams of single-repeat stimuli (dimensions -> (f, t, stimuli))
    multi_rep_cochleagrams : cochleagrams of multi-repeat stimuli (dimensions -> (f, t, stimuli))
    n_neurons : number of neurons in dataset
    resp_dataset : spike time responses to each repeat of multi- and single-repeat stimuli, for each neuron (dimensions ->
    (neurons, stimuli, repeats))
    stimuli : training, validation, cross-validation, and test set stimuli
    params : dict containing cochleagram parameters and tensorisation parameter
    norm : type of stimulus normalisation, either 'all' or 'each'
    mean_cv : mean over all stimuli in the cross-validation set
    std_cv : standard deviation over all stimuli in the cross-validation set
    clip_bins : number of bins to be clipped from the start of the stimuli and responses
    shift_bins : number of bins that the response is to be shifted by with respect to the stimuli
    type_data : determines whether datasets are to be built as NumPy arrays or torch tensors
    zero_padding : boolean variable, indicates whether stimuli were zero-padded
    padding_bins : number of bins added to zero-padded stimuli
    t_axis : time axis used to compute cochleagrams

    Returns
    -------
    train_stim_dataset : torch tensor/NumPy array of tensorised stimuli in training sets 
    (dimensions -> folds x stimuli x f x history x t)
    train_resp_dataset : torch tensor/NumPy array of repeat-averaged responses to stimuli in training sets 
    (dimensions -> neurons x folds x stimuli x t)
    val_stim_dataset : torch tensor/NumPy array of tensorised stimuli in val sets 
    (dimensions -> folds x stimuli x f x history x t)
    val_resp_dataset : torch tensor/NumPy array of repeat-averaged responses to stimuli in val sets 
    (dimensions -> neurons x folds x stimuli x t)
    multi_rep_resps : spike time responses to each repeat of multi-repeat stimuli, for each neuron (dimensions -> (neurons, 
    stimuli, repeats))
    single_rep_resps : spike time responses to single-repeat stimuli, for each neuron (dimensions -> (neurons, stimuli, 
    repeats))
    '''
    
    training_stimuli = stimuli['training_stimuli']
    validation_stimuli = stimuli['validation_stimuli']
    folds = len(training_stimuli)
    
    bin_size = params['bin_size']
    n_F = params['n_F']
    his_steps = params['n_h']
    
    sess_multi_rep_resps = []
    sess_single_rep_resps = []
    
    sess_train_resp_dataset = []
    sess_val_resp_dataset = []
    
    for session in range(len(sess_resp_dataset)):
        multi_rep_resps = []
        single_rep_resps = []
        
        resp_dataset = sess_resp_dataset[session]
        for nID in resp_dataset:
            n_multi_rep_resps = []
            n_single_rep_resps = []
    
            for stim in resp_dataset[nID]:
                if len(resp_dataset[nID][stim]) > 1: # if stimulus was repeated more than once, response is a multi-repeat
                    n_multi_rep_resps.append(resp_dataset[nID][stim])
                else: # if stimulus was repeated only once, response is a single-repeat
                    n_single_rep_resps.append(resp_dataset[nID][stim])
                    
            multi_rep_resps.append(n_multi_rep_resps)
            single_rep_resps.append(n_single_rep_resps)
            
        sess_multi_rep_resps.append(multi_rep_resps)
        sess_single_rep_resps.append(single_rep_resps)
        
        
        # NumPy arrays to hold datasets
        # if stimuli were zero-padded, and padding_bins is not equal to his_steps-1 (as is the case for convolutional models), 
        # stimuli and response datasets have different (time) lengths
        # training stimuli, dimensions -> (folds, f, history, concatenated t)
        train_stim_dataset = np.zeros((folds, n_F, his_steps, 
                                       len(training_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))
        # validation stimuli, dimensions -> (folds, f, history, concatenated t)
        val_stim_dataset = np.zeros((folds, n_F, his_steps, 
                                     len(validation_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))
        
        if padding_bins != his_steps-1 and zero_padding is True:
            # training responses, dimensions -> (neurons, folds, concatenated t)
            train_resp_dataset = np.zeros((len(resp_dataset), folds, 
                                           len(training_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-padding_bins)))
            # validation responses, dimensions -> (neurons, folds, concatenated t)
            val_resp_dataset = np.zeros((len(resp_dataset), folds, 
                                         len(validation_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-padding_bins)))
        else:
            train_resp_dataset = np.zeros((len(resp_dataset), folds, 
                                           len(training_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))
            val_resp_dataset = np.zeros((len(resp_dataset), folds, 
                                         len(validation_stimuli[0])*(np.size(single_rep_cochleagrams, 1)-his_steps+1)))
        
        for nID in range(len(resp_dataset)):
            print('nID = ' + str(nID))
            n_train_resp_data_single = single_rep_resps[nID]
            
            # get stimulus and response grids for all single-repeat stimuli, for one neuron
            train_stim_dataset_single, n_train_resp_single = get_stim_response_grid_concat(n_train_resp_data_single, 
                                                                                           single_rep_cochleagrams, 
                                                                                           training_stimuli[0][7:], None, 
                                                                                           his_steps, bin_size, 'train', norm, 
                                                                                           mean_cv, std_cv, clip_bins, 
                                                                                           shift_bins, zero_padding, 
                                                                                           padding_bins, t_axis)
            
            for fID in range(folds):
                # responses to each repeat of each multi-repeat stimulus in training fold, for one neuron
                n_train_resp_data_multi = [multi_rep_resps[nID][stim] for stim in training_stimuli[fID][:7]]
                
                # get stimulus and response grids for training fold multi-repeat stimuli, for one neuron
                train_stim_dataset_multi, n_train_resp_multi = get_stim_response_grid_concat(n_train_resp_data_multi, 
                                                                                             multi_rep_cochleagrams, 
                                                                                             training_stimuli[:7], fID, 
                                                                                             his_steps, bin_size, 'train', norm, 
                                                                                             mean_cv, std_cv, clip_bins, 
                                                                                             shift_bins, zero_padding, 
                                                                                             padding_bins, t_axis)
                
                # training stimulus and response sets for each fold are made of all single-repeat data and fold-specific
                # multi-repeat data; multi- and single- repeat data are concatenated along their time dimension
                train_stim_dataset[fID] = np.concatenate((train_stim_dataset_multi, train_stim_dataset_single), axis=-1)
                train_resp_dataset[nID, fID] = np.concatenate((n_train_resp_multi, n_train_resp_single), axis=-1)
                
                # responses to each repeat of each multi-repeat stimulus in validation fold, for one neuron
                n_val_resp_data_multi = [multi_rep_resps[nID][stim] for stim in validation_stimuli[fID]] 
                
                # get stimulus and response grids for validation fold multi-repeat stimuli, for one neuron
                val_stim_dataset_multi, n_val_resp_multi = get_stim_response_grid_concat(n_val_resp_data_multi, 
                                                                                         multi_rep_cochleagrams, 
                                                                                         validation_stimuli, fID, his_steps, 
                                                                                         bin_size, 'train', norm, mean_cv, 
                                                                                         std_cv, clip_bins, shift_bins, 
                                                                                         zero_padding, padding_bins, t_axis)
                
                val_stim_dataset[fID] = val_stim_dataset_multi
                val_resp_dataset[nID, fID, :] = n_val_resp_multi
        
        # convert NumPy array datasets to torch tensors of suitable torch float type
        train_stim_dataset = torch.tensor(train_stim_dataset).to(torch.float32).to(device)
        train_resp_dataset = torch.tensor(train_resp_dataset).to(torch.float32).to(device)
        val_stim_dataset = torch.tensor(val_stim_dataset).to(torch.float32).to(device)
        val_resp_dataset = torch.tensor(val_resp_dataset).to(torch.float32).to(device)
        
        if type_data == 'numpy': # if data is to be used as NumPy arrays, convert it back to that format
            train_stim_dataset = train_stim_dataset.detach().cpu().numpy()
            train_resp_dataset = train_resp_dataset.detach().cpu().numpy()
            val_stim_dataset = val_stim_dataset.detach().cpu().numpy()
            val_resp_dataset = val_resp_dataset.detach().cpu().numpy()
        
        sess_train_resp_dataset.append(train_resp_dataset)
        sess_val_resp_dataset.append(val_resp_dataset)
        
    return train_stim_dataset, sess_train_resp_dataset, val_stim_dataset, sess_val_resp_dataset, sess_multi_rep_resps, \
        sess_single_rep_resps

#############################################################################################################################

def build_grid_stream_data(stim_dataset, resp_dataset, t_axis, his_steps, bins_shift, bins_clip, zero_padding):
    from tensorize_mod import tensorize_mod
    
    # time axis used for computing cochleagrams, to use for binning responses into PSTHs
    t = t_axis
    X_ft = stim_dataset

    # stimulus normalisation to have zero mean and unit variance
    X_ft_normalized = (X_ft - np.mean(X_ft[:])) / np.std(X_ft[:], ddof=1)
    
    # stimulus tensorisation; X_fht dimensions -> (f, history, t); X_fht.shape[2] = X_ft.shape[1] - (his_steps-1)
    # if X_ft is zero-padded, and padding_bins = his_steps-1, the bins removed for tensorisations are the zero-padding 
    # ones
    X_fht = tensorize_mod(X_ft_normalized, his_steps)

    # if statement is needed because X_fht[:, :, :-0] does not work
    if bins_shift != 0:
        X_fht = X_fht[:, :, :-bins_shift] # remove final bins of stimulus to account for latency
    
    resp_grid = np.zeros((len(resp_dataset), len(t)-1-(his_steps - 1)))
    
    # list of spike time arrays in response to a given stimulus for each repeat, for a given neuron
    n_count = 0
    
    for nID in resp_dataset:
        spike_times = resp_dataset[nID]
        # ensure all entries in spike times list are lists
        spike_times_1 = [[o] if type(o) is float else o for o in spike_times] 
        
        response = np.histogram(spike_times_1, t)[0] # get PSTH of responses to all repeats of stimulus, for one neuron
        response = response[bins_clip:] # clip desired number of time bins from the start of the response
        response = response[bins_shift:] # remove initial bins of response to account for latency
        
        if not zero_padding: # if stimuli were not zero-padded, clip first his_steps-1 bins to match cochleagram length
            response = response[his_steps-1:]
            
        resp_grid[n_count] = response
        
        n_count += 1
    
    #X_fht = torch.tensor(X_fht).to(torch.float32).to(device)
    #resp_grid = torch.tensor(resp_grid).to(torch.float32).to(device)
    
    return X_fht, resp_grid

#############################################################################################################################

def build_grid_NS3_preprocessed(stim_dataset, resp_dataset, n_stim_array, n_h, n_F, n_n, mean_all, std_all): # NOT IN USE
    from tensorize_mod import tensorize_mod
    
    if n_h <= 25:
        stim_val_array = stim_dataset['val_s_np'][:, 25-n_h:]
        stim_train_tune_array = stim_dataset['train_tune_s_np'][:, 25-n_h:]
        
    else:
        stim_val_array = stim_dataset['val_s_np']
        stim_train_tune_array = stim_dataset['train_tune_s_np']
    
    n_stimuli_tune = n_stim_array[3]
    
    stim_train_array = stim_train_tune_array[:, :, :-n_stimuli_tune]
    stim_tune_array = stim_train_tune_array[:, :, -n_stimuli_tune:]
    
    resp_val_array = resp_dataset['val_r_np'][:, 25:]
    resp_train_tune_array = resp_dataset['train_tune_r_np'][:, 25:]
    resp_train_array = resp_train_tune_array[:, :, :-n_stimuli_tune]
    resp_tune_array = resp_train_tune_array[:, :, -n_stimuli_tune:]
    
    data = {0:{0:stim_train_tune_array, 1:resp_train_tune_array}, 
            1:{0:stim_val_array, 1:resp_val_array},
            2:{0:stim_train_array, 1:resp_train_array}, 
            3:{0:stim_tune_array, 1:resp_tune_array}}
    
    concat_data = {}
    
    for i in range(4):
        n_stim = n_stim_array[i]
        stim_data = data[i][0]
        resp_data = data[i][1]
        
        concat_data_i = {}
        
        len_stim_resp = np.size(stim_data, 1)-n_h

        concat_stim_array = np.zeros((n_F, n_h, n_stim*len_stim_resp))
        
        tr_resp_array = np.transpose(resp_data, (0, 2, 1))
        concat_resp_array = np.reshape(tr_resp_array, (n_n, -1))
        
        for stim in range(n_stim):
            stim_data_ft = (stim_data[:, :, stim] - mean_all) / std_all
            stim_data_fht = tensorize_mod(stim_data_ft, n_h)
            
            concat_stim_array[:, :, stim*len_stim_resp:(stim+1)*len_stim_resp] = stim_data_fht[:, :, 1:]
        
        concat_data_i[0] = torch.tensor(concat_stim_array).to(torch.float32).to(device)
        concat_data_i[1] = torch.tensor(concat_resp_array).to(torch.float32).to(device)
        
        concat_data[i] = concat_data_i
        
    train_stim_dataset = concat_data[2][0]
    train_resp_dataset = concat_data[2][1]
    tune_stim_dataset = concat_data[3][0]
    tune_resp_dataset = concat_data[3][1]
    val_stim_dataset = concat_data[1][0]
    val_resp_dataset = concat_data[1][1]
    train_tune_stim_dataset = concat_data[0][0]
    train_tune_resp_dataset = concat_data[0][1]
        
    return train_stim_dataset, train_resp_dataset, tune_stim_dataset, tune_resp_dataset, \
        val_stim_dataset, val_resp_dataset, train_tune_stim_dataset, train_tune_resp_dataset

#############################################################################################################################

def redblue(m=None):
    """
    def redblue(m)
    
    This is a Python adaptation of the MATLAB function redblue.m, which returns
    a bilinear red and blue colour map.

    REDBLUE(M) returns an M-by-3 matrix containing the colormap.
    """
    
    import numpy as np
    from matplotlib.colors import ListedColormap
    import colorsys
        
    top = np.floor(m/2)
    bot = m - top
    
    btop = np.concatenate((1.0*np.ones((int(top), 1)), 
                           np.expand_dims((np.transpose(np.append(np.arange(1, int(top)), top))/top)**2, axis=1), 
                           np.ones((int(top), 1))), axis=1) #1.0 hue good red
    bbot = np.concatenate((0.7*np.ones((int(bot), 1)), 
                           np.expand_dims((1-np.transpose(np.append(np.arange(1, int(bot)), bot)/bot))**2, axis=1), 
                           np.ones((int(bot), 1))), axis=1) # 0.7 hue good blue
    b_hsv = np.concatenate((bbot, btop), axis=0)
    b_rgba = np.ones((int(top+bot), 4))
    for cID in range(int(top+bot)):
        b_rgba[cID, 0:3] = colorsys.hsv_to_rgb(b_hsv[cID, 0], b_hsv[cID, 1], b_hsv[cID, 2])
        
    redblue_map = ListedColormap(b_rgba)
    
    return redblue_map

#############################################################################################################################

def calc_CC_norm(y_td, y_hat):
    """
    BENLIB_PY function
    
    Calculate CC_norm, CC_abs, CC_max of a y_td matrix where t is time and d are repeats
    """
    
    n_t, n = y_td.shape
    y = np.mean(y_td, axis=1)
    Ey = np.mean(y)
    Eyhat = np.mean(y_hat)
    Vy = np.sum(np.multiply((y-Ey), (y-Ey)))/n_t
    Vyhat = np.sum(np.multiply((y_hat-Eyhat), (y_hat-Eyhat)))/n_t
    Cyyhat = np.sum(np.multiply((y-Ey), (y_hat-Eyhat)))/n_t
    SP = (np.var(np.sum(y_td, axis=1), ddof=1)-np.sum(np.var(y_td, axis=0, ddof=1)))/(n*(n-1))
    CCabs = Cyyhat/np.sqrt(Vy*Vyhat)
    CCnorm = Cyyhat/np.sqrt(SP*Vyhat)
    CCmax = np.sqrt(SP/Vy)
    if SP <= 0:
        #print('SP less than or equal to zero - CCmax and CCnorm cannot be calculated.')
        CCnorm = np.nan
        CCmax = 0
    # added criterion - prevent highly negative and highly positive CCnorms
    if CCnorm > 2 or CCnorm < -2:
        CCnorm = np.nan
    return CCnorm, CCabs, CCmax

#############################################################################################################################

def calc_CC_norm_test(y_td, y_hat):
    """
    BENLIB_PY function
    
    Calculate CC_norm, CC_abs, CC_max of a y_td matrix where t is time and d are repeats
    """
    
    import pdb
    pdb.set_trace()
    n_t, n = y_td.shape
    y = np.mean(y_td, axis=1)
    Ey = np.mean(y)
    Eyhat = np.mean(y_hat)
    Vy = np.sum(np.multiply((y-Ey), (y-Ey)))/n_t
    Vyhat = np.sum(np.multiply((y_hat-Eyhat), (y_hat-Eyhat)))/n_t
    Cyyhat = np.sum(np.multiply((y-Ey), (y_hat-Eyhat)))/n_t
    SP = (np.var(np.sum(y_td, axis=1), ddof=1)-np.sum(np.var(y_td, axis=0, ddof=1)))/(n*(n-1))
    CCabs = Cyyhat/np.sqrt(Vy*Vyhat)
    CCnorm = Cyyhat/np.sqrt(SP*Vyhat)
    CCmax = np.sqrt(SP/Vy)
    if SP <= 0:
        #print('SP less than or equal to zero - CCmax and CCnorm cannot be calculated.')
        CCnorm = np.nan
        CCmax = 0
    # ADDED CRITERION -> PREVENT HIGHLY NEGATIVE OR HIGHLY POSITIVE CCNORMS DUE TO VALIDATION FOLD COMBINATIONS FROM
    # AFFECTING AVERAGES
    if CCnorm > 2 or CCnorm < -2:
        CCnorm = np.nan
    return CCnorm, CCabs, CCmax

#############################################################################################################################

def AddSysPath(new_path):
    """ AddSysPath(new_path): adds a directory to Python's sys.path

    Does not add the directory if it does not exist or if it's already on
    sys.path. Returns 1 if OK, -1 if new_path does not exist, 0 if it was
    already on sys.path.
    """
    import sys, os

    # Avoid adding nonexistent paths
    if not os.path.exists(new_path): return -1

    # Standardize the path. Windows is case-insensitive, so lowercase
    # for definiteness.
    new_path = os.path.abspath(new_path)
    if sys.platform == 'win32':
        new_path = new_path.lower(  )

    # Check against all currently available paths
    for x in sys.path:
        x = os.path.abspath(x)
        if sys.platform == 'win32':
            x = x.lower(  )
        if new_path in (x, x + os.sep):
            return 0
    sys.path.append(new_path)
    return 1

#############################################################################################################################

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])