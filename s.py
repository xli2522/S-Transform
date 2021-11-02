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

def sTransform(ts, sample_rate, frange=[0, 500], frate = 1, onesided=True, elevated=True, elevation=10e-15):
    '''Compute the S Transform
    Input:
                    ts                 (ndarray)            time series data
                    sample_rate        (int)                sample rate
                    frange             (list, optional)     frequency range (Hz)
                    frate              (int, optional)      frequency sampling rate
                    onesided           (bool, optional)     include only the left side of the FFT if True
                    elevated           (bool, optional)    when True, add elevation to the Gaussian
                    elevation          (float, optional)   magnitude of the elevation   
    Output:
                    amp                (ndarray)            spectrogram table
    Note:
                    amp                                     shape [frequency, time], lower -> higher
    '''

    length = len(ts)               
    Nfreq = [int(frange[0]*length/sample_rate), int(frange[1]*length/sample_rate)]     
    tsVal = np.copy(ts)            
    ampL = np.zeros((int((Nfreq[1]-Nfreq[0])/frate)+1,length), dtype='c8')                    
    tsFFT = np.fft.fft(tsVal)               
    vec = np.hstack((tsFFT, tsFFT))         

    # set the lowest frequency row to small values => 'zero' frequency     
    ampL[0] = np.fft.ifft(vec[0:length]*_window_normal(length, 0, elevated=elevated, elevation=elevation))             
    for i in range(frate, (Nfreq[1]-Nfreq[0])+1, frate):                       
        ampL[int(i/frate)] = np.fft.ifft(vec[Nfreq[0]+i:Nfreq[0]+i+length]*_window_normal(length, Nfreq[0]+i, factor=1, elevated=elevated, elevation=elevation))  
    if not onesided:
        ampR = np.zeros((int((Nfreq[1]-Nfreq[0])/frate)+1,length), dtype='c8') 
        for i in range(frate, (Nfreq[1]-Nfreq[0])+1, frate):
            ampR[int(i/frate)] = np.fft.ifft(vec[Nfreq[0]+length//2+i:Nfreq[0]+length//2+i+length]*_window_normal(length, Nfreq[0]+i, factor=1, elevated=elevated, elevation=elevation))  
        return ampL, ampR
    return ampL

def _window_normal(length, freq, factor = 1, elevated=True, elevation = 10e-15):
    '''Gaussian Window function w/ elevation
    Input: 
                    length              (int)               length of the Gaussian window
                    freq                (int)               frequency at which this window is to be applied to
                    factor              (int, float)        normalizing factor of the Gaussian; default set to 1
                    elevated            (bool, optional)    when True, add elevation to the Gaussian
                    elevation           (float, optional)   magnitude of the elevation     
    Output:
                    win                 (ndarray)           split gaussian window
    Note:
                    win                                     not your typical Gaussian => split + elevated (if True)
    '''
    gauss = scipy.signal.gaussian(length,std=(freq)/(2*np.pi*factor))
    if elevated:
        elevated_gauss = np.where(gauss<elevation, elevation,gauss)
        win = np.hstack((elevated_gauss,elevated_gauss))[length//2:length//2+length]
    else:
        win = np.hstack((gauss,gauss))[length//2:length//2+length]

    return win

def recoverS(table, lowFreq = 0):
    '''Quick 'Perfect' Recovery of Time-Series from S Transform Spectrogram Generated using sTransform
    Input:
                    table               (ndarray)           spectrogram table
                    lowFreq             (int, optional)     starting frequency
    Output:
                    ts_recovered        (ndarray)           recovered time series
    Note:
                    *only when [0] frequency row encodes full time-series information
    '''
    tablep = np.copy(table)                 
    length = tablep.shape[1]
    s_row = tablep[0]
    tsFFT_recovered = np.fft.fft(s_row)/_window_normal(length, lowFreq)

    ts_recovered = np.fft.ifft(tsFFT_recovered)

    return ts_recovered

def inverseS(table, lowFreq = 0):
    '''The True Inverse S Transform (without optimization)
    Input:
                    table               (ndarray)           spectrogram table
                    lowFreq             (int, optional)     starting frequency
    Output:
                    ts_recovered        (ndarray)           recovered time series
                    recovered_tsFFT     (ndarray)           the recovered FFT of the time series (left side only)
    Note:
                    ts_recovered is not optimized
                    recovered_tsFFT has errors of approximatly 10e-7 times the time series amplitude
    '''
    tablep = np.copy(table)
    length = tablep.shape[1]
    vec = np.zeros(length)              
    
    full_tsFFT = np.fft.fft(tablep[0])/_window_normal(length, 0)

    for i in range(tablep.shape[0]):              
        vec_inv0 = np.fft.fft(tablep[i])[0]/_window_normal(length, i)[0]
        vec[i] = vec_inv0

    recovered_tsFFT = vec
    recovered_ts = np.fft.ifft(recovered_tsFFT)*2

    return recovered_ts, recovered_tsFFT