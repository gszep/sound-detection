from numpy import array,ndarray,mean,std,exp,cos,sqrt,pi,gradient,vstack,amax,arange,linspace,meshgrid,log10,zeros,hstack,abs
from numpy.fft import rfft,rfftfreq
from scipy.signal import fftconvolve

from itertools import islice, chain
from multiprocessing import Pool,cpu_count


def standardise(x,axis=None) :
    """
    Return array standardised across given axis.

    Paramters
    ---------
    x : <numpy.ndarray>
        array of inputs standardised
    axis : None | int | (int,..,int)
         axes along which the standardisation is performed
    Returns
    ---------
    output : <numpy.ndarray>
        array of outputs

    ---------
    """
    epsilon = 10e-8
    if axis == 0 :
        return (x-mean(x,axis=axis))/(std(x,axis=axis)+epsilon)
    if axis == 1 :
        return ((x.T-mean(x,axis=axis))/(std(x,axis=axis)+epsilon)).T
    else :
        raise NotImplementedError('only axis=0 or axis=1 work for now')


def gaussian(x,mean=0.0,std=1.0) :
    """
    Calculate values of gaussian f(x) at x.
    f(x) = exp(-((x-mean)/std)**2) / sqrt(2*pi*std**2)

    Paramters
    ---------
    x : <numpy.ndarray>
        array of inputs to be passed to function
    mean : <float>
        mean of gaussian, defaults to zero
    std : <float>
        standard deviation of gaussian, defaults to one

    Returns
    ---------
    f(x) : <numpy.ndarray>
        array out outputs with same shape as inputs

    ---------
    """
    return exp(-((x-mean)/std)**2) / sqrt(2*pi*std**2)


def stft(signal, window=128, step=65 ):
    """
    Perform a shoft-time fourier transform with a Hann window
    https://en.wikipedia.org/wiki/Short-time_Fourier_transform

    Paramters
    ---------
    signal : <numpy.ndarray>
        array of shape length n_points of a single-channel signal

    window : int
        number of samples per window

    step : int
        stride of window, determines overlap

    Returns
    ---------
    time,frequency,stft : <numpy.ndarray>,<numpy.ndarray>,<numpy.ndarray>
        time and frequency mesh aligned with stft spectorgram
        assuming 44100Hz sampling rate

    ---------
    """

    signal -= signal.mean()
    freq = rfftfreq(window, d = 1/44100.0 )

    # construct hann windows
    windows,time = overlap(signal, window, step)
    windows *= 0.54-0.46*cos(2*pi*arange(window)/(window-1))

    # return rfft along window axis
    return time,freq,rfft(windows).T


def overlap(signal, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    signal : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")

    # pad signal with zeros
    padded_signal = hstack((zeros(window_size//2),signal,zeros(window_size//2)))

    n_windows = len(signal) // window_step+1
    out = ndarray((int(n_windows),int(window_size)),dtype=signal.dtype)

    for i in range(int(n_windows)):
        # "slide" the window along the samples
        start = int( i * window_step )
        stop = int( start + window_size )
        out[i] = padded_signal[start : stop]

    time = linspace(0,len(signal),n_windows)/44100.0
    return out,time

def spectrogram(d,window=2048,step=2048/16,thresh=4):
    """
    creates a spectrogram
    thresh: threshold minimum power for log spectrogram
    """

    # calculate
    time,freq,fft = stft(d,window,step)
    specgram = abs(fft)

    # threshold
    specgram = specgram[freq<4200]
    freq = freq[freq<4200]

    # normalise
    specgram /= specgram.max() # volume normalize to max 1
    specgram = log10(specgram) # take log
    specgram[specgram < -thresh] = -thresh

    time_grid,freq_grid = meshgrid(time,freq,copy=False)
    return time_grid,freq_grid,specgram
