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

import  numpy                as np
import  scipy
import  matplotlib.pyplot    as plt
from    s import *

def test():

    # Generate a quadratic chirp signal
    dt = 0.001; t = 3
    rate = int(1/dt)
    ts = np.linspace(0, t, int(t/dt))
    
    data = scipy.signal.chirp(ts, 10, t, 120, method='quadratic')

    # Compute S Transform Spectrogram
    spectrogram = sTransform(data, sample_rate=rate, frate=rate/len(data),
                                        downsample=None, frange=[0,500])
    plt.imshow(abs(spectrogram), origin='lower', aspect='auto')
    plt.title('Original Spectrogram')
    plt.colorbar()
    plt.show()


    # Quick Inverse of ts from S Transform
    inverse_ts, inverse_tsFFT = inverseS(spectrogram)
    fig, axs = plt.subplots(2,1)
    axs[0].plot(data)
    axs[1].plot(inverse_ts.real)
    axs[0].set_title('Original Signal')
    axs[1].set_title('(inverseS) Freq-passed, down-sampled Signal')
    plt.show()

    plt.plot(inverse_ts)
    plt.plot(inverse_ts-data)
    plt.title('Time Series Reconstruction Error')
    plt.legend(['Recovered ts', 'Error'])
    plt.show()

    # Compute S Transform Spectrogram on the recovered time series
    # IMPORTANT: the inverse transform itself is exact;
    # however, information could be lost in the forward ST due to downsampling
    # in both time and frequency
    inverseSpectrogram = sTransform(inverse_ts, 
                            sample_rate=len(inverse_ts)/len(data)*rate, 
                                                        frange=[0,500])
    plt.imshow(abs(inverseSpectrogram), origin='lower', aspect='auto')
    plt.title('Recovered Spectrogram (inverseS)')
    plt.colorbar()
    plt.show()

    return 

if __name__ == '__main__':
    test()