from numpy import array,mean,std,exp,sqrt,pi,gradient,vstack,amax,linspace,meshgrid
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
        NotImplemented('only axis=0 or axis=1 work for now')


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


def stft(signal,time,window=0.1) :
    """
    Perform a shoft-time fourier transform with a window function
    given by the default parameters specified in MusicNet
    https://en.wikipedia.org/wiki/Short-time_Fourier_transform

    Paramters
    ---------
    signal : <numpy.ndarray>
        array of shape length n_points of a single-channel signal

    time : <numpy.ndarray>
        one dimensional array of timepoints at which signal was sampled

    freq : <numpy.ndarray>
        one dimensional array of freqencies to evaluate; defaults to all
        up to the nyquist frequency, at the limiting resolution.

    Returns
    ---------
    time,frequency,stft : <numpy.ndarray>,<numpy.ndarray>,<numpy.ndarray>
        time and frequency mesh aligned with stft spectorgram

    ---------
    """

    # getting frequency-time grid
    dt = mean(gradient(time))
    f_nyquist = 0.5/dt

    # logarithmic binning
    notes = linspace(0,108,108*4)
    freq = 2**((notes-69)/12.0)*440.0

    assert amax(freq) <= f_nyquist,'maximal frequency {}Hz must be less than nyquist {}Hz'.format(
        amax(freq),f_nyquist)

    # construct window function
    window = array(int(44100*window)*[1])

    # multithreaded batched fftconvolve
    pool = Pool(cpu_count())
    results = []

    batch_size = freq.size/(10*cpu_count())+1
    for freq_batch in batch(freq,batch_size) :

        results += [
            pool.apply_async(
                fftconvolvebatch, args=(signal,time,freq_batch,window)) ]

    # wait for threads to finish
    pool.close()
    pool.join()

    # collect and return results
    results = vstack([ array(result.get()) for result in results ])
    time_grid,freq_grid = meshgrid(time,freq,copy=False)
    return time_grid,freq_grid,results


def fftconvolvebatch(signal,time,freq_batch,window,mode='same') :
    """
    Batch compute <scipy.signal.fftconvolve>;
    see its docstring for further details

    Paramters
    ---------
    signal_batch : [ <numpy.ndarray> ... <numpy.ndarray> ]
        list of one dimensional arrays of length n_points to convolve

    window : <numpy.ndarray>
        one dimensional array to perform convolution with

    Returns
    ---------
    results : [ <numpy.ndarray> ... <numpy.ndarray> ]
        results of <scipy.signal.fftconvolve> for batch

    ---------
    """

    results = []
    for freq in freq_batch :
        results += [ fftconvolve(signal*exp(-2j*pi*time*freq),window,mode) ]

    return results


def batch(l, n):
    """Yield successive n-sized batches from array"""
    for i in range(0, len(l), n):
        yield l[i:i + n]
