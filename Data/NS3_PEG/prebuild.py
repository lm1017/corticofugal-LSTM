import os
import csv
import json
import h5py
import numpy as np
import pickle

def unique(list1):
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            
    return unique_list

data_dir = '' # main data directory
sessions = os.listdir(data_dir) # list contents of data directory

meta_ID = '.resp.json' # session metadata file ID
epoch_ID = '.resp.epoch.csv' # session file ID - describes presentation order of stimuli (with pre- and post-stim silence)
data_ID = '.resp.h5' # responses file ID

neuron_IDs = []
stim_IDs = []
unique_stim_IDs = []
unique_stim_num = []
stim_times = []

all_sess_resps = []
for session in sessions:
    print(session)
    sess_dir = data_dir + session # session directory
    sess_dir_spec = os.listdir(sess_dir) # list directory contents (only one sub-directory)
    sess_dir = sess_dir + '/' + sess_dir_spec[0] + '/' # full path to session data folder
    
    meta_file = sess_dir + sess_dir_spec[0] + meta_ID # full path to session metadata file

    f = open(meta_file)
    data_nID = json.load(f) # session metadata dict
    
    neuron_IDs.append(data_nID['chans']) # list of neuron IDs from session
    
    ###########
    epoch_file = sess_dir + sess_dir_spec[0] + epoch_ID # full path to session file
    
    with open(epoch_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader) # load session data -> list of dicts with information for each time point in session
    
    # list of "names" presented at each time point in session: includes stims, silence etc.
    name_list = [line['name'] for line in data]
    # list of stims presented in session in chronological order, with repetitions, extracted from name_list
    stim_list = [n for n in name_list if "STIM" in n]
    print(len(stim_list))
    # remove repetitions and chronological info from stim_list, only keep all unique IDs
    unique_stim_list = unique(stim_list)
    print(len(unique_stim_list))
    # list of number of repeats for each stimulus in unique_stim_list
    unique_stim_occ = [stim_list.count(sID) for sID in unique_stim_list]
    
    # append stimuli-related lists to lists containing all sessions
    stim_IDs.append(stim_list)
    unique_stim_IDs.append(unique_stim_list)
    unique_stim_num.append(unique_stim_occ)
    
    pre_stim_end = []
    for i, j in enumerate(name_list):
        if j == 'PreStimSilence':
            pre_stim_end.append(float(data[i]['end'])) # list of end times of pre-stim silence
            
    post_stim_start = []
    for i, j in enumerate(name_list):
        if j == 'PostStimSilence':
            post_stim_start.append(float(data[i]['start'])) # list of start times of post-stim silence
    
    times = list(zip(pre_stim_end, post_stim_start)) # list of tuples of start and end times for each stimulus
    
    stim_times.append(times) # append stimulus times to list containing all sessions
    
    ###########
    data_file = sess_dir + sess_dir_spec[0] + data_ID # full path to responses file
    # load stimuli from h5 file
    data_in = h5py.File(data_file, 'r') # load response data from h5 file
    
    data_dict = {}
    for key, dataset in data_in.items():
        # response file contains spike times (in seconds) for each neuron over the whole session
        data_dict[key] = np.array(dataset[:])
    
    neuron_spikes = []
    for nID in data_nID['chans']: # loop over neurons in session
        print(nID)
        spiketimes = data_dict[nID] # spike times for one neuron over whole session
        
        stim_spikes = [None]*len(times) # create empty list to hold spike times for the neuron, for each stimulus
        
        for interval in range(len(times)): # loop over time intervals, in the session, when stimuli were presented
            stim_spikes[interval] = []
            for spike in spiketimes: # loop over spike times
                if (times[interval][0] <= spike) and (spike <= times[interval][1]): # spikes falling in interval
                    # convert absolute spike time (in s) to spike time relative to start of interval (in ms)    
                    stim_spikes[interval].append(int(np.ceil((spike-times[interval][0])*1000)))
        
        neuron_spikes.append(stim_spikes)
    all_sess_resps.append(neuron_spikes)
        
NS3_PEG_prebuild = {'unique_stim_list': unique_stim_list, 'unique_stim_occ': unique_stim_occ, 'sessions': sessions, 
                    'unique_stim_num': unique_stim_num, 'neuron_IDs': neuron_IDs, 'all_sess_resps': all_sess_resps, 
                    'stim_IDs': stim_IDs}

# save pre-built dataset file
'''f = open('NS3_PEG_prebuild.pckl', 'wb')
pickle.dump(NS3_PEG_prebuild, f)
f.close()'''