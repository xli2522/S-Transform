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

import numpy            as np
import scipy.signal

from typing import Union, Optional

def sTransform( ts              : np.ndarray        , 
                sample_rate     : Union[int, float] , 
                frange          : Union[list[int, int]]       = [0, 500]  , 
                frate           : Union[int, float] = 1         , 
                alpha           : int               = 1         ,
                downsample      : Optional[int]     = None      
                ) -> np.ndarray:
    
    '''Compute the S Transform of a time series data array

    Parameters
    ----------
    ts (np.ndarray)         : time series data array
    sample_rate (int)       : sample rate of the time series
    frange (list[int, int]) : frequency range to compute the S Transform
    frate (int)             : frequency resolution of the S Transform
    alpha (int)             : normalization factor for the Gaussian window
    downsample (int)        : downsample factor for the S Transform

    Returns
    -------
    amp (np.ndarray)        : S Transform spectrogram array

    NOTE:
    * The S Transform is computed using the inverse FFT
    '''

    length          : int           = len(ts)       # length of the input ts
    Nfreq           : list[int, int]= [int(frange[0]*length/sample_rate), 
                                        int(frange[1]*length/sample_rate)]
    tsVal           : np.ndarray    = np.copy(ts)   # copy of the input ts

    # number of freq bins
    number_freq     : int           = abs(int(Nfreq[1] - Nfreq[0]))  

    _scaled_frate   : int           = int(np.ceil(frate*length/sample_rate)) 

    # FFT of the original time series
    tsFFT           : np.ndarray    = np.fft.fft(tsVal)

    # time domain downsampling
    if downsample   ==  None:
        tsFFT_cut   : np.ndarray    = np.concatenate((tsFFT, tsFFT))
        downsampled_length          : int       = len(tsFFT)
        normalize_cut               : int       = 1

    elif isinstance(downsample, int):
        # positive half
        tsFFT_positive              : np.ndarray= tsFFT[:Nfreq[1]]             
        # negative half   
        tsFFT_negative              : np.ndarray= tsFFT[-Nfreq[1]:len(tsFFT)]     
        if downsample < 2*number_freq:
            # perform the max allowed lower and higher frequency cut-off
            pass
        else:
            # 0 padding to make up for the high and low-passed freq elements
            tsFFT_positive          : np.ndarray= np.concatenate((
                tsFFT_positive, 
                np.zeros(downsample//2-number_freq)
                ))  

            tsFFT_negative          : np.ndarray= np.concatenate((
                np.zeros(downsample//2-number_freq), 
                tsFFT_negative
                ))

        # connect high and low-passed coefficients
        tsFFT_cut   : np.ndarray    = np.concatenate(
            (tsFFT_positive, tsFFT_negative))
        downsampled_length          : int       = len(tsFFT_cut)

        # normalization factor
        normalize_cut = 1 - downsampled_length/length

    # prepare the stacked vector for the S Transform convolution operation
    # vec             : np.ndarray    = np.hstack((tsFFT_cut, tsFFT_cut))      
    vec             : np.ndarray    = tsFFT_cut
    
    # spectrogram array container
    amp             : np.ndarray    = np.zeros((
        int(number_freq/_scaled_frate)+1,
        downsampled_length), 
        dtype='c8')

    # convolution operation
    # for each frequency bin, perform the inverse FFT
    for i in range(_scaled_frate, number_freq+1, _scaled_frate):                       
        amp[int(i/_scaled_frate)] = np.fft.ifft(
                    vec[Nfreq[0]+i:Nfreq[0]+i+downsampled_length]
                    *_window_normal(downsampled_length, alpha+Nfreq[0]+i, 
                                    factor=1)*normalize_cut)
    
    return amp

def _window_normal( length       : int, 
                    freq         : int, 
                    factor       : Union[int, float] = 1
                    ) -> np.ndarray:
    '''Splitted Gaussian window function for S Transform convolution

    Parameters
    ----------
    length (int)         : length of the Gaussian window
    freq (int)           : frequency at which this window is to be applied to
    factor (int, float)  : normalizing factor of the Gaussian; default set to 1

    Returns
    -------
    win (np.ndarray)     : split gaussian window

    NOTE:
    win is not your typical Gaussian => splitted for S Transform convolution
    '''
    gauss       : np.ndarray    = scipy.signal.gaussian(
        length,std=freq/(2*np.pi))*factor
    
    return np.hstack((gauss,gauss))[length//2:length//2+length]

def recoverS(   table       : np.ndarray, 
                lowFreq     : Optional[int] = None
                ) -> np.ndarray:
    '''[Deprecated: uses inverseS() instead]

    Original Function Information:
    Quick 'Perfect' Recovery of Time-Series from S Transform Spectrogram 
    Generated using sTransform
        
    Parameters
    ----------
    table (np.ndarray)      : spectrogram table
    lowFreq (int, optional) : starting frequency

    Returns
    -------
    ts_recovered (np.ndarray) : recovered time series

    NOTE:
    * only when [0] frequency row encodes full time-series information
    * lowFreq is not yet supported
    '''
    Warning('This function is deprecated. Useing inverseS() instead.')

    return inverseS(table, lowFreq)

def inverseS(   table       : np.ndarray, 
                lowFreq     : Optional[int] = None
                ) -> tuple[np.ndarray, np.ndarray]:
    '''The True Inverse S Transform (without optimization)
    Parameters
    ----------
    table               : spectrogram table
    lowFreq             : starting frequency
    
    Returns
    -------
    ts_recovered        : recovered time series
    recovered_tsFFT     : the recovered FFT of the time series (left side only)

    NOTE:
    * ts_recovered is not yet optimized
    * lowFreq is not yet supported
    '''
    
    tablep          : np.ndarray    = np.copy(table)
    length          : int           = tablep.shape[1]
    recovered_tsFFT : np.ndarray    = np.zeros(length, dtype='c8')

    for i in range(tablep.shape[0]):
        recovered_tsFFT[i] = np.fft.fft(tablep[i])[0]
    
    recovered_ts    : np.ndarray    = np.fft.ifft(recovered_tsFFT)

    return recovered_ts, recovered_tsFFT