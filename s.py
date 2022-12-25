# Copyright (C) 2021 Xiyuan Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import scipy.signal

def sTransform(ts, sample_rate, frange=[0, 500], frate = 1, downsample='none',
                                onesided=True, elevated=True, elevation=10e-8):
    '''Compute the S Transform
    Input:
                ts                 (ndarray)            time series data
                sample_rate        (int)                ts data sample rate
                frange             (list, optional)     frequency range (Hz)
                frate              (int, optional)      frequency sampling rate
                downsample         (str, int, optional) down-sampled length
                                                        in time ['none',
                                                        length(int)]
                onesided           (bool, optional)     include only the left 
                                                        side of the FFT if True
                elevated           (bool, optional)     when True, add elevation
                                                        to the Gaussian
                elevation          (float, optional)    magnitude of the 
                                                        elevation   
    Output:
                amp                (ndarray)            spectrogram table
    Note:
                amp                                     shape [frequency, 
                                                        time], lower -> higher
    '''

    length = len(ts)               # length of the input ts
    Nfreq = [int(frange[0]*length/sample_rate), 
                        int(frange[1]*length/sample_rate)]
    tsVal = np.copy(ts)            # copy of the input ts
    number_freq = abs(int(Nfreq[1]-Nfreq[0]))   # total number of freq bins
    _scaled_frate = int(np.ceil(frate*length/sample_rate)) # rescaled frate
    #_scaled_frate = frate              # freq bin number
    tsFFT = np.fft.fft(tsVal)                   # FFT of the original ts

    # time domain downsampling
    if downsample=='none':
        tsFFT_cut = np.concatenate((tsFFT, tsFFT))
        downsampled_length = len(tsFFT)
        normalize_cut = 1
    elif isinstance(downsample, int):
        tsFFT_positive = tsFFT[Nfreq[0]:Nfreq[1]]          # positive half
        tsFFT_negative = tsFFT[-Nfreq[1]:len(tsFFT)
                                            -Nfreq[0]]     # negative half
        if downsample < 2*number_freq:
            # perform the max allowed lower and higher frequency cut-off
            # note: np.fft.fft gives array (of complex values) with of length: 
            # 2*ts_length;
            pass

        else:
            # 0 padding to make up for the high and low-passed freq elements
            tsFFT_positive = np.concatenate((tsFFT_positive, 
                        np.zeros(downsample//2-number_freq)))  
            # 0 padding again
            tsFFT_negative = np.concatenate((tsFFT_negative, 
                        np.zeros(downsample//2-number_freq)))

        # connect high and low-passed coefficients
        tsFFT_cut = np.concatenate((tsFFT_positive, tsFFT_negative))
        downsampled_length = len(tsFFT_cut)                 # new length 
        # normalization coefficient
        normalize_cut = 1 - downsampled_length/length

    vec = np.hstack((tsFFT_cut, tsFFT_cut))      

    # empty (0) spectrogram array
    amp = np.zeros((int(number_freq/_scaled_frate)+1,downsampled_length), 
                                                                dtype='c8')
    # set the lowest frequency row to small values => 'zero' frequency 
    # this step is solely for the purpose of 'recoverS()' function    
    amp[0] = np.fft.ifft(vec[0:downsampled_length]
                            *_window_normal(downsampled_length, 0, 
                                elevated=elevated, elevation=elevation))
    for i in range(_scaled_frate, number_freq+1, _scaled_frate):                       
        amp[int(i/_scaled_frate)] = np.fft.ifft(
                    vec[Nfreq[0]+i:Nfreq[0]+i+downsampled_length]
                    *_window_normal(downsampled_length, Nfreq[0]+i, factor=1, 
                        elevated=elevated, elevation=elevation)*normalize_cut)
    
    return amp

def _window_normal(length, freq, factor = 1, elevated=True, elevation =10e-8):
    '''Gaussian Window function w/ elevation
    Input: 
                length              (int)               length of the Gaussian 
                                                        window
                freq                (int)               frequency at which this
                                                        window is to be applied
                                                        to
                factor              (int, float)        normalizing factor of 
                                                        the Gaussian; default 
                                                        set to 1
                elevated            (bool, optional)    when True, add elevation
                                                        to the Gaussian
                elevation           (float, optional)   magnitude of the 
                                                        elevation     
    Output:
                win                 (ndarray)           split gaussian window
Note:
                win                                     not your typical 
                                                        Gaussian => split + 
                                                        elevated (if True)
    '''
    gauss = scipy.signal.gaussian(length,std=freq/(2*np.pi))*factor
    if elevated:
        elevated_gauss = np.where(gauss<elevation, elevation,gauss)
        win = np.hstack((elevated_gauss,
                            elevated_gauss))[length//2:length//2+length]
    else:
        win = np.hstack((gauss,gauss))[length//2:length//2+length]

    return win

def recoverS(table, lowFreq = 0, elevated=True, elevation=10e-8):
    '''Quick 'Perfect' Recovery of Time-Series from S Transform Spectrogram 
        Generated using sTransform
    Input:
                table               (ndarray)           spectrogram table
                lowFreq             (int, optional)     starting frequency
                elevated            (bool, optional)    when True, add elevation
                                                        to the Gaussian
                elevation           (float, optional)   magnitude of the 
                                                        elevation     
    Output:
                ts_recovered        (ndarray)           recovered time series
    Note:
                *only when [0] frequency row encodes 
                    full time-series information
    '''
    tablep = np.copy(table)                 
    length = tablep.shape[1]
    s_row = tablep[0]
    tsFFT_recovered = np.fft.fft(s_row)/_window_normal(length, lowFreq, 
                                    elevated=elevated, elevation=elevation)

    ts_recovered = np.fft.ifft(tsFFT_recovered)

    return ts_recovered

def inverseS(table, lowFreq = 0, elevated=True, elevation =10e-8):
    '''The True Inverse S Transform (without optimization)
    Input:
                table               (ndarray)           spectrogram table
                lowFreq             (int, optional)     starting frequency
                elevated            (bool, optional)    when True, add elevation
                                                        to the Gaussian
                elevation           (float, optional)   magnitude of the 
                                                        elevation     
    Output:
                ts_recovered        (ndarray)           recovered time series
                recovered_tsFFT     (ndarray)           the recovered FFT of 
                                                        the time series (left 
                                                        side only)
    Note:
                    ts_recovered is not optimized
    '''
    tablep = np.copy(table)
    length = tablep.shape[1]
    recovered_tsFFT = np.zeros(length, dtype='c8')
    
    #full_tsFFT = np.fft.fft(tablep[0])/_window_normal(length, 0)

    for i in range(tablep.shape[0]):
        recovered_tsFFT[i] = np.fft.fft(tablep[i])[0]

    recovered_ts = np.fft.ifft(recovered_tsFFT)*2       
    # assuming the input is real
    return recovered_ts, recovered_tsFFT