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
import matplotlib.pyplot as plt
from scipy.signal import chirp
from s import s_transform, inverse_s_transform

def test_s_transform():
    """
    Test the S-Transform and its inverse on a quadratic chirp signal.
    """
    # Time vector and sampling rate
    dt = 0.001
    t_end = 3.0
    fs = int(1.0 / dt)
    t = np.linspace(0, t_end, int(t_end * fs), endpoint=False)

    # Generate quadratic chirp signal
    x = chirp(t, f0=10.0, t1=t_end, f1=120.0, method='quadratic')

    # Compute S-Transform spectrogram
    st = s_transform(
        x,
        sample_rate=fs,
        freq_range=(0.0, 500.0),
        freq_step=fs / x.size,
        downsample=None
    )
    plt.figure()
    plt.imshow(np.abs(st), origin='lower', aspect='auto')
    plt.title('Original Spectrogram')
    plt.colorbar()
    plt.show()

    # Inverse S-Transform
    x_rec, fft_rec = inverse_s_transform(st)

    # Magnitude compensation (real signal, one-sided FFT)
    x_rec_comp = x_rec * 2.0

    # Plot original vs. recovered
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(x)
    axes[0].set_title('Original Signal')
    axes[1].plot(x_rec_comp.real)
    axes[1].set_title('Recovered Signal (magnitude-compensated)')
    plt.tight_layout()
    plt.show()

    # Reconstruction error
    plt.figure()
    plt.plot(x_rec_comp.real, label='Recovered Signal')
    plt.plot(x - x_rec_comp.real, label='Error')
    plt.title('Time Series Reconstruction Error')
    plt.legend()
    plt.show()

    # Spectrogram of recovered signal
    st_rec = s_transform(
        x_rec.real,
        sample_rate=fs,
        freq_range=(0.0, 500.0),
        freq_step=fs / x_rec.size
    )
    plt.figure()
    plt.imshow(np.abs(st_rec), origin='lower', aspect='auto')
    plt.title('Recovered Spectrogram')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test_s_transform()