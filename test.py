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
import scipy
import matplotlib.pyplot as plt
from s import *

def test():

    # Generate a quadratic chirp signal
    dt = 0.0001
    rate = int(1/dt)
    ts = np.linspace(0, 5, int(1/dt))
    
    data = scipy.signal.chirp(ts, 10, 5, 300, method='quadratic')

    # Compute S Transform Spectrogram
    spectrogram = sTransform(data, sample_rate=rate)
    plt.imshow(abs(spectrogram), origin='lower', aspect='auto')
    plt.title('Original Spectrogram')
    plt.show()

    # Quick Recovery of ts from S Transform spectrogram
    inverse_ts = recoverS(spectrogram)
    plt.plot(inverse_ts-data)
    plt.title('Time Series Reconstruction Error')
    plt.show()

    # Compute S Transform Spectrogram on the recovered time series
    inverseSpectrogram = sTransform(inverse_ts, sample_rate=rate)
    plt.imshow(abs(inverseSpectrogram), origin='lower', aspect='auto')
    plt.title('Recovered Specctrogram')
    plt.show()

    return 

if __name__ == '__main__':
    test()