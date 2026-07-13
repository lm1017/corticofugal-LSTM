import numpy as np
import random
from scipy import stats

def get_oddball(cf, fs, tone_dB, tone_dur, rise_dur, fall_dur, isi_dur, resp_win_dur, n_conds, params, n_blocks, n_tones, 
                start):
    isi = 0*np.arange(0, isi_dur, 1/fs)
    
    conds = {}
    for cond in range(n_conds):
        cond_params = params[cond+1]
        df = cond_params['df']

        # Sine waves
        f1 = 0.5*(-df*cf + cf*np.sqrt(np.square(df)+4))
        f2 = np.square(cf)/f1
        tone1 = sine_with_cosine_ramp(f1, tone_dur, fs, rise_dur, fall_dur, tone_dB)
        tone2 = sine_with_cosine_ramp(f2, tone_dur, fs, rise_dur, fall_dur, tone_dB)
        
        cond_blocks = {}
        for block in range(n_blocks):
            if block == 0:
                f1_prob = cond_params['prob']/100
            elif block == 1:
                f1_prob = (100 - cond_params['prob'])/100
            elif block == 2:
                f1_prob = 50/100
                
            f2_prob = 1 - f1_prob
            f_sequence = np.random.choice([1, 2], n_tones, p=[f1_prob, f2_prob])
            
            if start == 'sound':
                tone_sequence = np.array([])
                win_sequence = np.array([])
            elif start == 'silence':
                tone_sequence = isi
                win_sequence = np.zeros_like(isi)
                
            for tone_idx in range(n_tones):
                tone_f = f_sequence[tone_idx]

                tone_isi = np.concatenate((locals()['tone' + str(tone_f)], isi))
                tone_sequence = np.concatenate((tone_sequence, tone_isi))
                
                resp_win = np.zeros_like(tone_isi)
                resp_win[:int(resp_win_dur*fs)] += 1
                win_sequence = np.concatenate((win_sequence, resp_win))
                
            cond_blocks['block ' + str(block)] = [tone_sequence, f_sequence, win_sequence]
        conds['cond ' + str(cond)] = cond_blocks
        
    return conds

def get_oddball_cond_2(cf, fs, tone_dB, tone_dur, rise_dur, fall_dur, isi_dur, resp_win_dur, params, blocks_f_seq, n_blocks, 
                       n_tones, start, fix_f_seq):
    isi = 0*np.arange(0, isi_dur, 1/fs)
    
    cond = 2
    cond_params = params[cond+1]
    df = cond_params['df']
    
    # Sine waves
    f1 = 0.5*(-df*cf + cf*np.sqrt(np.square(df)+4))
    f2 = np.square(cf)/f1
    tone1 = sine_with_cosine_ramp(f1, tone_dur, fs, rise_dur, fall_dur, tone_dB)
    tone2 = sine_with_cosine_ramp(f2, tone_dur, fs, rise_dur, fall_dur, tone_dB)

    blocks = {}
    for block in range(n_blocks):
        if fix_f_seq:
            f_sequence = blocks_f_seq[block]
        else:
            if block == 0:
                f1_prob = cond_params['prob']/100
            elif block == 1:
                f1_prob = (100 - cond_params['prob'])/100
            elif block == 2:
                f1_prob = 50/100
                
            f2_prob = 1 - f1_prob
            f_sequence = np.random.choice([1, 2], n_tones, p=[f1_prob, f2_prob])
        
        if start == 'sound':
            tone_sequence = np.array([])
            win_sequence = np.array([])
        elif start == 'silence':
            tone_sequence = isi
            win_sequence = np.zeros_like(isi)
            
        for tone_idx in range(n_tones):
            tone_f = f_sequence[tone_idx]

            tone_isi = np.concatenate((locals()['tone' + str(tone_f)], isi))
            tone_sequence = np.concatenate((tone_sequence, tone_isi))
            
            resp_win = np.zeros_like(tone_isi)
            resp_win[:int(resp_win_dur*fs)] += 1
            win_sequence = np.concatenate((win_sequence, resp_win))
            
        blocks['block ' + str(block)] = [tone_sequence, f_sequence, win_sequence]
        
    return blocks

def get_switch_oddball(cf, fs, tone_dB, tone_dur, rise_dur, fall_dur, isi_dur, resp_win_dur, params, n_blocks, n_tones, 
                       start):
    isi = 0*np.arange(0, isi_dur, 1/fs)

    df = params['df']

    # Sine waves
    f1 = 0.5*(-df*cf + cf*np.sqrt(np.square(df)+4))
    f2 = np.square(cf)/f1
    tone1 = sine_with_cosine_ramp(f1, tone_dur, fs, rise_dur, fall_dur, tone_dB)
    tone2 = sine_with_cosine_ramp(f2, tone_dur, fs, rise_dur, fall_dur, tone_dB)
    
    f1_prob = params['prob']/100
    f2_prob = 1 - f1_prob
    f_sequence_1 = np.random.choice([1, 2], int(n_tones/2), p=[f1_prob, f2_prob])
    f_sequence_2 = np.random.choice([1, 2], int(n_tones/2), p=[f2_prob, f1_prob])
    f_sequence = np.concatenate((f_sequence_1, f_sequence_2))
    
    if start == 'sound':
        tone_sequence = np.array([])
        win_sequence = np.array([])
    elif start == 'silence':
        tone_sequence = isi
        win_sequence = np.zeros_like(isi)
        
    for tone_idx in range(n_tones):
        tone_f = f_sequence[tone_idx]
        
        tone_isi = np.concatenate((locals()['tone' + str(tone_f)], isi))
        tone_sequence = np.concatenate((tone_sequence, tone_isi))

        resp_win = np.zeros_like(tone_isi)
        resp_win[:int(resp_win_dur*fs)] += 1
        win_sequence = np.concatenate((win_sequence, resp_win))
        
    whole_sequence = np.tile(tone_sequence, n_blocks)
    whole_win_sequence = np.tile(win_sequence, n_blocks)
    return [whole_sequence, f_sequence, whole_win_sequence]

def get_resp_curve(cf, fs, tone_dB, tone_dur, rise_dur, fall_dur, isi_dur, resp_win_dur, n_freqs, oct_range, n_reps, 
                   start):
    isi = 0*np.arange(0, isi_dur, 1/fs)

    # Sine waves
    f_low = np.sqrt(np.square(cf)/np.power(2, oct_range))
    f_high = np.square(cf)/f_low
    freqs = np.logspace(np.log10(f_low), np.log10(f_high), num=n_freqs)
    
    tones = {}
    for f_idx in range(n_freqs):
        freq = freqs[f_idx]
        tone = sine_with_cosine_ramp(freq, tone_dur, fs, rise_dur, fall_dur, tone_dB)
        tones[f_idx] = tone
    
    f_array = np.zeros((n_freqs*n_reps))
    for tone_idx in range(len(f_array)):
        f_array[tone_idx] = tone_idx // n_reps
    random.shuffle(f_array)

    if start == 'sound':
        tone_sequence = np.array([])
        win_sequence = np.array([])
    elif start == 'silence':
        tone_sequence = isi
        win_sequence = np.zeros_like(isi)
        
    for tone_idx in range(len(f_array)):
        f_idx = f_array[tone_idx]
        tone_f = tones[int(f_idx)]
        
        tone_isi = np.concatenate((tone_f, isi))
        tone_sequence = np.concatenate((tone_sequence, tone_isi))

        resp_win = np.zeros_like(tone_isi)
        resp_win[:int(resp_win_dur*fs)] += 1
        win_sequence = np.concatenate((win_sequence, resp_win))
        
    return [tone_sequence, f_array, win_sequence]

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

def pitman_morgan_test(x, y):
    """
    Pitman-Morgan test for equality of variances
    in paired samples.

    Parameters
    ----------
    x, y : array-like
        Paired observations.

    Returns
    -------
    t_stat : float
        Test statistic.
    p_value : float
        Two-sided p-value.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) < 3:
        raise ValueError("Need at least 3 paired observations")

    s = x + y
    d = x - y

    r = np.corrcoef(s, d)[0, 1]

    n = len(x)

    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

    return t_stat, p_value