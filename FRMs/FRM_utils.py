import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from util_funcs import extents

def sine_with_cosine_ramp(freq, duration, fs, on_ramp_duration, off_ramp_duration, dB_intensity):
    t = np.arange(0, duration, 1/fs)
    
    # Sine wave
    signal = np.sin(2*np.pi*freq*t)
    
    # Ramp length in samples
    on_ramp_len = int(on_ramp_duration*fs)
    off_ramp_len = int(off_ramp_duration*fs)
    
    # Cosine ramp (Hann half-window)
    on_ramp = 0.5*(1-np.cos(np.pi*np.arange(on_ramp_len)/on_ramp_len))
    off_ramp = 0.5*(1-np.cos(np.pi*np.arange(off_ramp_len)/off_ramp_len))

    # Create full envelope
    envelope = np.ones_like(signal)
    
    # Apply fade-in
    envelope[:on_ramp_len] = on_ramp
    
    # Apply fade-out
    envelope[-off_ramp_len:] = off_ramp[::-1]
    
    Pa = 10**(dB_intensity/20)*2e-5
    
    return signal*envelope*Pa


def get_cochleagram(y, fs, bin_size, n_F, min_F, max_F, cochleagram_type, padding_bins, zero_padding):
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
    
    import numpy as np
    import cochlea_lorenzo
    
    dataset = 'tones'
    cochleagram_fcn = 'cochleagram_' + cochleagram_type
    cochleagram = getattr(cochlea_lorenzo, cochleagram_fcn) # use cochleagram function for the chosen type

    #y_new_array = np.zeros((20, 1252)) # for checking dB SPL
        
    if zero_padding:
        y = np.concatenate((np.zeros((int((padding_bins*bin_size/1000)*fs))), y)) # pad raw sounds with zeros

    # 'log' indicates log spacing of frequency channels
    X_ft, params = cochleagram(y, fs, bin_size, dataset, zero_padding, padding_bins, spacing='log', 
                               freq_info=[min_F, max_F, n_F])
    
    # time and frequency axes used for cochleagrams
    t = params['spectrogram_t'] 
    
    return X_ft, t


def save_FRMs_pdf(FRMs, freqs, dBs, dataset_ID, lambdas):
    n_lambdas = np.size(FRMs, 0)
    n_HUs = np.size(FRMs, 1)
    
    HUs_per_page = 16
    rows = 4
    cols = 4
    num_pages = n_HUs // HUs_per_page + 1
    
    file_num = 0
    file_name = 'FRM_pdfs/' + dataset_ID + '_lamb_'
    '''while True:
        if os.path.isfile(file_name + str(file_num) + '.pdf'):
            file_num += 1
        else:
            break
    file_name = file_name + str(file_num)'''

    for lamb in range(n_lambdas):
        lamb_file_name = file_name + str(lambdas[lamb])

        with PdfPages(lamb_file_name + '.pdf') as pdf:
            HU = 0
            for page in range(num_pages):
                f, axs = plt.subplots(rows, cols)
                
                for row_idx in range(rows):
                    if HU == n_HUs:
                        break
                    for col_idx in range(cols):
                        HU = HUs_per_page*page + row_idx*rows + col_idx
                        if HU == n_HUs:
                            break
                        
                        FRM = np.transpose(FRMs[lamb, HU])
                        FRM[FRM < 0] = 0
                        '''im = axs[row_idx, col_idx].imshow(FRM, cmap='hot_r', aspect='auto', interpolation='none', 
                                                          extent=extents(freqs) + extents(dBs), origin='lower')'''
                        
                        im = axs[row_idx, col_idx].imshow(FRM, cmap='hot_r', aspect='auto', interpolation='none', 
                                                          origin='lower')
                        
                        # Find index of maximum value
                        max_idx = np.argmax(FRM)
                        row, col = np.unravel_index(max_idx, FRM.shape)
                        
                        rect = Rectangle((col - 0.5, row - 0.5), 1, 1, edgecolor='blue', facecolor='none', linewidth=0.5)
                        axs[row_idx, col_idx].add_patch(rect)


                        if row_idx == rows - 1:
                            xtick_idx = np.linspace(0, len(freqs)-1, 6, dtype=int)
                            axs[row_idx, col_idx].set_xticks(xtick_idx, [f"{int(freqs[i])}" for i in xtick_idx])
                            axs[row_idx, col_idx].set_xlabel('Frequency (Hz)', fontsize=5)
                            axs[row_idx, col_idx].xaxis.set_tick_params(labelsize=5)
                        else:
                            axs[row_idx, col_idx].xaxis.set_tick_params(labelcolor='none')
                        if col_idx == 0:
                            ytick_idx = np.array([0, 2, 4, 6])
                            axs[row_idx, col_idx].set_yticks(ytick_idx, [f"{int(dBs[i])}" for i in ytick_idx])
                            axs[row_idx, col_idx].set_ylabel('Intensity (dB)', fontsize=5)
                            axs[row_idx, col_idx].yaxis.set_tick_params(labelsize=5)
                        else:
                            axs[row_idx, col_idx].yaxis.set_tick_params(labelcolor='none')
                            
                        axs[row_idx, col_idx].set_title('HU ' + str(HU), fontsize=5)
                
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close(f)
        
            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'FRMs, lambda ' + str(lambdas[lamb])