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

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from s import s_transform, inverse_s_transform


IMAGE_DIR = Path(__file__).with_name("img")

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
    )
    display_mask = st.frequencies <= 300.0
    fig, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(
        np.abs(np.asarray(st)[display_mask]),
        origin='lower',
        aspect='auto',
    )
    axis.set_title('Original Spectrogram (0–300 Hz)')
    axis.set_xlabel('Time samples')
    axis.set_ylabel('Frequency bins')
    fig.colorbar(image, ax=axis, label='Magnitude')
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'original_spectrogram.png', dpi=150)
    plt.show()
    plt.close(fig)

    # Inverse S-Transform
    x_rec, fft_rec = inverse_s_transform(st)

    # The inverse restores the conjugate negative-frequency bins for real input.
    x_rec_comp = x_rec.real
    np.testing.assert_allclose(x_rec_comp, x, atol=1e-10)

    # Plot original vs. recovered
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(x)
    axes[0].set_title('Original Signal')
    axes[1].plot(x_rec_comp)
    axes[1].set_title('Recovered Signal')
    plt.tight_layout()
    plt.show()

    # Reconstruction error
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(x_rec_comp, label='Recovered Signal')
    axis.plot(x - x_rec_comp, label='Error')
    axis.set_title('Time Series Reconstruction Error')
    axis.set_xlabel('Time samples')
    axis.legend()
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'recovered_ts_error.png', dpi=150)
    plt.show()
    plt.close(fig)

    # Spectrogram of recovered signal
    st_rec = s_transform(
        x_rec.real,
        sample_rate=fs,
        freq_range=(0.0, 500.0),
        freq_step=fs / x_rec.size
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(np.abs(st_rec), origin='lower', aspect='auto')
    axis.set_title('Recovered Spectrogram')
    axis.set_xlabel('Time samples')
    axis.set_ylabel('Frequency bins')
    fig.colorbar(image, ax=axis, label='Magnitude')
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / 'recovered_spectrogram.png', dpi=150)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    test_s_transform()
