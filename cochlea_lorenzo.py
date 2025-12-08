'''
Cochleagram - adapted from the cochlea module of benlib-py (https://github.com/ben-willmore/benlib-py)
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from scipy.signal import spectrogram
from matplotlib import mlab
import matlab.engine
import scipy.sparse as sps


def whosMy(*args):
    sequentialTypes = [dict, list, tuple]
    for var in args:
        t = type(var)
        if t == np.ndarray:
            print(type(var), var.dtype, var.shape)
        elif t in sequentialTypes:
            print(type(var), len(var))
        else:
            print(type(var))


def frq2erb(frq):
    '''
    See voicebox
    FRQ2ERB  Convert Hertz to ERB frequency scale ERB=(FRQ)
    '''
    frq = np.array(frq)
    g = np.abs(frq)
    erb = 11.17268 * np.sign(frq) * np.log(1+46.06538*g/(g+14678.49))
    bnd = 6.23e-6 * g**2 + 93.39e-3 * g + 28.52
    return erb, bnd


def erb2frq(erb):
    '''
    See voicebox
    ERB2FRQ  Convert ERB frequency scale to Hertz FRQ=(ERB)
    '''
    erb = np.array(erb)
    frq = np.sign(erb) * (676170.4 * (47.06538 -
                                      np.exp(0.08950404*np.abs(erb)))**(-1) - 14678.49)
    bnd = 6.23e-6 * frq**2 + 93.39e-3 * np.abs(frq) + 28.52
    return erb, bnd


def frq2erb_cat(frq):
    '''
    Equivalent of frq2erb for cat
    '''
    frq = np.array(frq)
    erb = 1000 * 13.7 * (frq**0.362)
    return erb


def erb2frq_cat(erb):
    '''
    Equivalent of erb2frq for cat
    '''
    frq = erb / ((13.7 * 1000))**(1./0.362)
    return frq


def melbankbw(n_filters, n_fft, f_s, f_lo, f_hi, spacing='log'):
    '''
    Melbank. Frequencies are equally spaced in the specified space (log etc)
    The middle of one filter is the bottom of the next and the top of the previous
    one, and they are triangular in shape.
    '''
    
    if spacing == 'log':
        freq2scale = np.log10
        def scale2freq(x): return np.power(10, x)
    elif spacing == 'erb':
        freq2scale = frq2erb
        scale2freq = erb2frq
    elif spacing == 'cat':
        freq2scale = frq2erb_cat
        scale2freq = erb2frq_cat

    # lo and hi filter frequencies in the desired units
    melfreq_lo, melfreq_hi = freq2scale([f_lo, f_hi])
    # mel range
    melrng = np.dot(np.array([melfreq_lo, melfreq_hi]), np.array([-1, 1]))
    # bin index of highest positive frequency (Nyquist if n is even)
    fn2 = np.floor(n_fft/2)

    if n_filters is None:
        n_filters = np.ceil(4.6*np.log10(f_s))

    # fixed increment required to get from melfreq_lo to melfreq_hi in n_filters steps
    melinc = (melfreq_hi - melfreq_lo)/(n_filters-1)
    if n_filters < 1:
        n_filters = np.round(melrng/(n_filters*1000)) + 1
    melinc = melrng/(n_filters - 1)

    # add 2 extra increments for the top and bottom because we specified the
    # centre frequencies
    melfreq_lo, melfreq_hi = melfreq_lo-melinc, melfreq_hi+melinc

    a = [0, 1, n_filters, n_filters+1]
    blim = scale2freq(melfreq_lo + [i*melinc for i in a])*n_fft/f_s

    mc = melfreq_lo + np.arange(1, n_filters)*melinc
    mc = np.append(mc, melfreq_lo + n_filters*melinc)

    # lowest FFT bin_0 required might be negative)
    b1 = int(np.floor(blim[0]) + 1)
    b4 = int(min(fn2, np.ceil(blim[3])-1))    # highest FFT bin_0 required

    # now map all the useful FFT bins_0 to filter1 centres
    pf = (freq2scale(np.arange(b1, b4)*f_s/n_fft) - melfreq_lo)/melinc
    pf = np.append(pf, (freq2scale(b4*f_s/n_fft) - melfreq_lo)/melinc)
    # remove any incorrect entries in pf due to rounding errors
    if pf[0] < 0:
        pf = np.delete(pf, 0)
        b1 = b1 + 1
    if pf[-1] >= n_filters + 1:
        pf = np.delete(pf, -1)
        b4 = b4 - 1
    
    fp = np.floor(pf) # FFT bin_0 i contributes to filters_1 fp(1+i-b1)+[0 1]
    pm = pf - fp # multiplier for upper filter
    k2 = np.argwhere(fp > 0)[0][0] # FFT bin_1 k2+b1 is the first to contribute to both upper and lower filters
    k3 = np.argwhere(fp < n_filters)[-1][0] # FFT bin_1 k3+b1 is the last to contribute to both upper and lower filters
    k4 = np.size(fp) # FFT bin_1 k4+b1 is the last to contribute to any filters
    if k2 == []:
        k2 = k4 + 1
    if k3 == []:
        k3 = 0
    
    r = np.concatenate((1 + fp[0:k3+1], fp[k2:k4])) # filter number_1
    r = [int(i)-1 for i in r]
    c = np.concatenate((np.append(np.arange(0, k3), k3), np.arange(k2, k4))) # FFT bin_1 - b1; no need to append k4 as it is not an index, just a size
    c = [int(i) for i in c]
    v = np.concatenate((pm[0:k3+1], 1 - pm[k2:k4]))
    mn = b1 + 1 # lowest fft bin_1
    mx = b4 + 1 # highest fft bin_1
    
    if b1 < 0:
        c = abs(c + b1 - 1) - b1 + 1; # convert negative frequencies into positive
    
    filters = sps.csr_matrix((v,(r, c)))
    filters.eliminate_zeros()
    
    sx = np.sum(filters, axis=1)
    sx_0 = sx + (sx == 0)
    a = np.tile(sx_0, (1, np.size(filters, axis=1)))
    a = np.squeeze(np.asarray(a))
    
    filters = filters.toarray()
    filters = np.divide(filters, a)
    
    # filter frequencies. freqs[0:-2] are the bottoms of the filters,
    # freqs[1:-1] are the middles, freqs[2:] are the tops
    freqs = scale2freq(melfreq_lo + np.arange(0, n_filters+2) * melinc)

    return filters, freq2scale(freqs[1:-1]), mn, mx


def melbank(n_filters, n_fft, f_s, f_lo, f_hi, spacing='log'):
    '''
    Melbank. Frequencies are equally spaced in the specified space (log etc)
    The middle of one filter is the bottom of the next and the top of the previous
    one, and they are triangular in shape.
    '''
    if spacing == 'log':
        freq2scale = np.log10
        def scale2freq(x): return np.power(10, x)
    elif spacing == 'erb':
        freq2scale = frq2erb
        scale2freq = erb2frq
    elif spacing == 'cat':
        freq2scale = frq2erb_cat
        scale2freq = erb2frq_cat

    # lo and hi filter frequencies in the desired units
    melfreq_lo, melfreq_hi = freq2scale([f_lo, f_hi])

    # fixed increment required to get from melfreq_lo to melfreq_hi in n_filters steps
    melinc = (melfreq_hi - melfreq_lo)/(n_filters-1)

    # add 2 extra increments for the top and bottom because we specified the
    # centre frequencies
    melfreq_lo, melfreq_hi = melfreq_lo-melinc, melfreq_hi+melinc

    # filter frequencies. freqs[0:-2] are the bottoms of the filters,
    # freqs[1:-1] are the middles, freqs[2:] are the tops
    freqs = scale2freq(melfreq_lo + np.arange(0, n_filters+2) * melinc)

    # artificially long fft_freqs to avoid clipping
    fft_freqs = np.arange(0, n_fft+1)/n_fft*f_s

    filters = []
    for idx in range(freqs.shape[0]-2):
        filt = np.zeros(fft_freqs.shape)
        [f_lo, f_mid, f_hi] = freqs[idx:idx+3]
        f_lo_idx = np.where(fft_freqs > f_lo)[0][0]
        f_mid_idx = np.where(fft_freqs > f_mid)[0][0]
        f_hi_idx = np.where(fft_freqs > f_hi)[0][0]

        # ramp up, then down again, about centre frequency
        if f_mid_idx > f_lo_idx:
            filt[f_lo_idx:f_mid_idx+1] = \
                np.arange(f_mid_idx+1-f_lo_idx)/(f_mid_idx-f_lo_idx)
        if f_hi_idx > f_mid_idx:
            filt[f_mid_idx:f_hi_idx+1] = \
                1 - np.arange(f_hi_idx+1-f_mid_idx)/(f_hi_idx-f_mid_idx)

        # clip filter to appropriate length
        filt = filt[:int(n_fft/2+1)]
        if np.sum(filt) > 0:
            filt = filt / np.sum(filt)
        filters.append(filt)

    return np.stack(filters, axis=0), freqs[1:-1]

def hill_function(x, n, c, SAT):
# y=hill_function(x,n,c)
    if n is None:
        n = 1.77
        
    if c is None:
        c = 1e-2

    if SAT is None:
        SAT = 0.16
    
    x_to_n = np.power((c*x), n)
    y = np.divide(x_to_n, (SAT + x_to_n))

    return y

###############################################################################

def cochleagram_spec_power(y_t, fs_hz, dt_ms, dataset, zero_padding, padding_bins, spacing='log', freq_info=None):
    '''
    Melbank-based log cochleagram. Need to check it thoroughly against
    matlab, and sort out threshold which is arbitrary
    '''
    
    if dataset == 'NS1' or dataset == 'DRC':
        params = {'spacing': spacing,
                  'fs_hz': fs_hz,
                  'threshold': -70}
    elif dataset == 'NS2' or dataset =='NS2_include_single' or dataset == 'NS3' or dataset == 'NS3_PEG':
        params = {'spacing': spacing,
                  'fs_hz': fs_hz,
                  'threshold': -70}
    
    if freq_info is not None:
        params['f_min'], params['f_max'], params['n_f'] = freq_info

    if spacing == 'log':
        params['nfft_mult'] = 4
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 31

    elif spacing == 'cat-erb':
        params['nfft_mult'] = 1
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 23

    # get actual dt (which is an integer number of samples)
    dt_sec_nominal = dt_ms/1000
    dt_bins = dt_sec_nominal*params['fs_hz']
    if dt_bins % 1 != 0:
        print('Warning -- rounding dt to an integer number of samples')
        dt_bins = np.round(dt_bins)
        
    params['dt_sec'] = dt_bins/params['fs_hz']

    # get window, overlap sizes
    t_window_bins = dt_bins * 2
    params['t_window_sec'] = t_window_bins/params['fs_hz']
    t_overlap_bins = t_window_bins - dt_bins

    # melbank
    while True:
        '''[filts, melfreqs] = melbank(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)'''
    
        [filts, melfreqs, na, nb] = melbankbw(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)
        
        if np.all(np.sum(filts[:10, :], axis=1) > 0):
            break
        params['nfft_mult'] = params['nfft_mult'] * 2
        print('boosting nfft_mult to %d' % params['nfft_mult'])

    ham_win = np.hamming(int(t_window_bins))
    win = ham_win
    spectrum_scaling = np.sqrt(1/(ham_win.sum()**2))

    [freqs, t, spec] = spectrogram(y_t, fs=fs_hz, window=win, nperseg=int(t_window_bins), noverlap=int(t_overlap_bins),
                                   nfft=int(t_window_bins*params['nfft_mult']), detrend=False, scaling='spectrum', 
                                   mode='complex')
    
    if zero_padding:
        t = t[:-padding_bins]

    mag_spec = abs(spec)/spectrum_scaling

    X_ft = np.maximum(10*np.log10(np.dot(filts, np.square(mag_spec[na-1:nb, :]))), params['threshold'])

    params['melbank'] = filts
    params['melfreqs'] = melfreqs
    params['spectrogram'] = mag_spec
    params['spectrogram_freqs'] = freqs
    params['spectrogram_t'] = 1000*np.append(t, t[-1]+np.diff(t)[0])
    
    return X_ft, params

###############################################################################

def cochleagram_multithreshold(y_t, fs_hz, dt_ms, dataset, spacing='log', freq_info=None):
    '''
    Melbank-based log cochleagram. Need to check it thoroughly against
    matlab, and sort out threshold which is arbitrary
    '''
    params = {'spacing': spacing,
              'fs_hz': fs_hz}
    if freq_info is not None:
        params['f_min'], params['f_max'], params['n_f'] = freq_info

    if spacing == 'log':
        params['nfft_mult'] = 4
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 31

    elif spacing == 'cat-erb':
        params['nfft_mult'] = 1
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 23

    # get actual dt (which is an integer number of samples)
    dt_sec_nominal = dt_ms/1000
    dt_bins = dt_sec_nominal*params['fs_hz']
    if dt_bins % 1 != 0:
        print('Warning -- rounding dt to an integer number of samples')
        dt_bins = np.round(dt_bins)

    params['dt_sec'] = dt_bins/params['fs_hz']

    # get window, overlap sizes
    t_window_bins = dt_bins * 2
    params['t_window_sec'] = t_window_bins/params['fs_hz']
    t_overlap_bins = t_window_bins - dt_bins

    # melbank
    while True:
        '''[filts, melfreqs] = melbank(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)'''
    
        [filts, melfreqs, na, nb] = melbankbw(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)
        
        if np.all(np.sum(filts[:10, :], axis=1) > 0):
            break
        params['nfft_mult'] = params['nfft_mult'] * 2
        print('boosting nfft_mult to %d' % params['nfft_mult'])

    ham_win = np.hamming(int(t_window_bins))
    win = ham_win
    spectrum_scaling = np.sqrt(1/(ham_win.sum()**2))

    [freqs, t, spec] = spectrogram(y_t, fs=fs_hz, window=win, nperseg=int(t_window_bins),
                                   noverlap=int(t_overlap_bins),
                                   nfft=int(t_window_bins*params['nfft_mult']), detrend=False, scaling='spectrum', mode='complex')
    
    params['melbank'] = {'filts' : filts, 'na' : na, 'nb' : nb}
    params['melfreqs'] = melfreqs
    
    # Loop through AN fibertypes
    #params.threshold = [0 -20 -40]
    params['threshold'] = [-15, -30, -40]
    # params['SAT'] = [8e-6, 2e-6, 4e-7]
    # params['SAT'] = [2e-5, 8e-6, 4e-6] # v1
    # params['SAT'] = [6e-5, 3e-5, 1e-5] # v2
    params['SAT'] = [8e-5, 5e-5, 3e-5] # v3
    # params['SAT'] = [4.6e-5, 4.6e-5, 4.6e-5]# v4
    
    modeldata = np.zeros((len(params['threshold']), params['n_f'], np.size(spec, 1)))
    
    for fiberType in range(3):
        threshold = params['threshold'][fiberType]
        SAT = params['SAT'][fiberType]
        modeldata[fiberType, :, :] = model_fiber_type(params, spec, threshold, SAT, spectrum_scaling)
    
    X_ft_multithreshold = np.reshape(modeldata, (3*np.size(modeldata, 1), np.size(modeldata, 2)))

    fiber_percent = [0.15, 0.25, 0.6] # ColburnCarney-JARO-2003
    #fiber_percent = 0.33*ones(1,3)
    X_ft = np.zeros(np.shape(modeldata[0]))
    for ii in range(np.size(modeldata, 0)):
        X_ft = X_ft + modeldata[ii]*fiber_percent[ii]
    
    #return X_ft, params
    return X_ft_multithreshold, params

def model_fiber_type(params, spec, threshold, SAT, scaling):
    filts = params['melbank']['filts']
    na = params['melbank']['na']
    nb = params['melbank']['nb']
    
    modeldata = np.empty((np.size(filts, 0), np.size(spec, 1)))
    modeldata[:] = np.nan
    
    mag_spec = abs(spec)/scaling
    modeldata = np.maximum(10*np.log10(np.dot(filts, np.square(mag_spec[na-1:nb, :]))), threshold) - threshold
    n = 1.77
    c = 1e-4
    modeldata = hill_function(modeldata, n, c, SAT)
    
    return modeldata

###############################################################################

def cochleagram_spec_Hill(y_t, fs_hz, dt_ms, dataset, spacing='log', freq_info=None):
    '''
    Melbank-based log cochleagram. Need to check it thoroughly against
    matlab, and sort out threshold which is arbitrary
    '''
    
    if dataset == 'NS1' or dataset == 'DRC':
        params = {'spacing': spacing,
                  'fs_hz': fs_hz,
                  'threshold': -40}
    elif dataset == 'NS2':
        params = {'spacing': spacing,
                  'fs_hz': fs_hz,
                  'threshold': -50}
        
    if freq_info is not None:
        params['f_min'], params['f_max'], params['n_f'] = freq_info

    if spacing == 'log':
        params['nfft_mult'] = 4
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 31

    elif spacing == 'cat-erb':
        params['nfft_mult'] = 1
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 23

    # get actual dt (which is an integer number of samples)
    dt_sec_nominal = dt_ms/1000
    dt_bins = dt_sec_nominal*params['fs_hz']
    if dt_bins % 1 != 0:
        print('Warning -- rounding dt to an integer number of samples')
        dt_bins = np.round(dt_bins)

    params['dt_sec'] = dt_bins/params['fs_hz']

    # get window, overlap sizes
    t_window_bins = dt_bins * 2
    params['t_window_sec'] = t_window_bins/params['fs_hz']
    t_overlap_bins = t_window_bins - dt_bins

    # melbank
    while True:
        '''[filts, melfreqs] = melbank(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)'''
    
        [filts, melfreqs, na, nb] = melbankbw(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)
        
        if np.all(np.sum(filts[:10, :], axis=1) > 0):
            break
        params['nfft_mult'] = params['nfft_mult'] * 2
        print('boosting nfft_mult to %d' % params['nfft_mult'])

    ham_win = np.hamming(int(t_window_bins))
    win = ham_win
    spectrum_scaling = np.sqrt(1/(ham_win.sum()**2))

    [freqs, t, spec] = spectrogram(y_t, fs=fs_hz, window=win, nperseg=int(t_window_bins),
                                   noverlap=int(t_overlap_bins),
                                   nfft=int(t_window_bins*params['nfft_mult']), detrend=False, scaling='spectrum', mode='complex')

    mag_spec = abs(spec)/spectrum_scaling

    X_ft = np.maximum(10*np.log10(np.dot(filts, np.square(mag_spec[na-1:nb, :]))), params['threshold']) - params['threshold']
    
    SAT = 0.16 # This was found from crossvalidation
    X_ft = hill_function(X_ft, 1.77, 1e-2, SAT)

    params['melbank'] = filts
    params['melfreqs'] = melfreqs
    params['spectrogram'] = mag_spec
    params['spectrogram_freqs'] = freqs
    params['spectrogram_t'] = t
    
    return X_ft, params

###############################################################################

def cochleagram_spec_log(y_t, fs_hz, dt_ms, dataset, spacing='log', freq_info=None):
    '''
    Melbank-based log cochleagram. Need to check it thoroughly against
    matlab, and sort out threshold which is arbitrary
    '''
    
    if dataset == 'NS1' or dataset == 'DRC':
        params = {'spacing': spacing,
                  'fs_hz': fs_hz,
                  'threshold': -40}
    elif dataset == 'NS2':
        params = {'spacing': spacing,
                  'fs_hz': fs_hz,
                  'threshold': -50}
        
    if freq_info is not None:
        params['f_min'], params['f_max'], params['n_f'] = freq_info

    if spacing == 'log':
        params['nfft_mult'] = 4
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 31

    elif spacing == 'cat-erb':
        params['nfft_mult'] = 1
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 23

    # get actual dt (which is an integer number of samples)
    dt_sec_nominal = dt_ms/1000
    dt_bins = dt_sec_nominal*params['fs_hz']
    if dt_bins % 1 != 0:
        print('Warning -- rounding dt to an integer number of samples')
        dt_bins = np.round(dt_bins)

    params['dt_sec'] = dt_bins/params['fs_hz']

    # get window, overlap sizes
    t_window_bins = dt_bins * 2
    params['t_window_sec'] = t_window_bins/params['fs_hz']
    t_overlap_bins = t_window_bins - dt_bins

    # melbank
    while True:
        '''[filts, melfreqs] = melbank(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)'''
    
        [filts, melfreqs, na, nb] = melbankbw(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)
        
        if np.all(np.sum(filts[:10, :], axis=1) > 0):
            break
        params['nfft_mult'] = params['nfft_mult'] * 2
        print('boosting nfft_mult to %d' % params['nfft_mult'])

    ham_win = np.hamming(int(t_window_bins))
    win = ham_win
    spectrum_scaling = np.sqrt(1/(ham_win.sum()**2))

    [freqs, t, spec] = spectrogram(y_t, fs=fs_hz, window=win, nperseg=int(t_window_bins),
                                   noverlap=int(t_overlap_bins),
                                   nfft=int(t_window_bins*params['nfft_mult']), detrend=False, scaling='spectrum', mode='complex')

    mag_spec = abs(spec)/spectrum_scaling
    
    X_ft = np.maximum(20*np.log10(np.dot(filts, mag_spec[na-1:nb, :])), params['threshold'])

    params['melbank'] = filts
    params['melfreqs'] = melfreqs
    params['spectrogram'] = mag_spec
    params['spectrogram_freqs'] = freqs
    params['spectrogram_t'] = t
    
    return X_ft, params

###############################################################################

def cochleagram_spec_log1plus(y_t, fs_hz, dt_ms, dataset, spacing='log', freq_info=None):
    '''
    Melbank-based log cochleagram. Need to check it thoroughly against
    matlab, and sort out threshold which is arbitrary
    '''
    
    if dataset == 'NS1' or dataset == 'DRC':
        params = {'spacing': spacing,
                  'fs_hz': fs_hz,
                  'threshold': -40}
    elif dataset == 'NS2':
        params = {'spacing': spacing,
                  'fs_hz': fs_hz,
                  'threshold': -50}
        
    if freq_info is not None:
        params['f_min'], params['f_max'], params['n_f'] = freq_info

    if spacing == 'log':
        params['nfft_mult'] = 4
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 31

    elif spacing == 'cat-erb':
        params['nfft_mult'] = 1
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 23

    # get actual dt (which is an integer number of samples)
    dt_sec_nominal = dt_ms/1000
    dt_bins = dt_sec_nominal*params['fs_hz']
    if dt_bins % 1 != 0:
        print('Warning -- rounding dt to an integer number of samples')
        dt_bins = np.round(dt_bins)

    params['dt_sec'] = dt_bins/params['fs_hz']

    # get window, overlap sizes
    t_window_bins = dt_bins * 2
    params['t_window_sec'] = t_window_bins/params['fs_hz']
    t_overlap_bins = t_window_bins - dt_bins

    # melbank
    while True:
        '''[filts, melfreqs] = melbank(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)'''
    
        [filts, melfreqs, na, nb] = melbankbw(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)
        
        if np.all(np.sum(filts[:10, :], axis=1) > 0):
            break
        params['nfft_mult'] = params['nfft_mult'] * 2
        print('boosting nfft_mult to %d' % params['nfft_mult'])

    ham_win = np.hamming(int(t_window_bins))
    win = ham_win
    spectrum_scaling = np.sqrt(1/(ham_win.sum()**2))

    [freqs, t, spec] = spectrogram(y_t, fs=fs_hz, window=win, nperseg=int(t_window_bins),
                                   noverlap=int(t_overlap_bins),
                                   nfft=int(t_window_bins*params['nfft_mult']), detrend=False, scaling='spectrum', mode='complex')

    mag_spec = abs(spec)/spectrum_scaling
    
    X_ft = np.dot(filts, mag_spec[na-1:nb, :])
    
    X_ft = np.log10(X_ft + 1);

    params['melbank'] = filts
    params['melfreqs'] = melfreqs
    params['spectrogram'] = mag_spec
    params['spectrogram_freqs'] = freqs
    params['spectrogram_t'] = t
    
    return X_ft, params