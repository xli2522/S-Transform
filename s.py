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
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Sequence, Tuple
from scipy import signal, fft

__all__ = ["s_transform", "inverse_s_transform"]

def s_transform(
    ts: ArrayLike,
    sample_rate: float,
    freq_range: Sequence[float] = (0.0, 500.0),
    freq_step: float = 1.0,
    alpha: float = 1.0,
    downsample: Optional[int] = None
) -> NDArray[np.complex128]:
    """
    Compute the S Transform of a time series data array.

    Parameters
    ----------
    ts : array_like
        Time series data array.
    sample_rate : float
        Sample rate of the time series.
    freq_range : sequence of float, optional
        Frequency range to compute the S Transform, default is (0.0, 500.0).
    freq_step : float, optional
        Frequency resolution of the S Transform, default is 1.0.
    alpha : float, optional
        Normalization factor for the Gaussian window, default is 1.0.
    downsample : int, optional
        Downsample factor for the S Transform, default is None.

    Returns
    -------
    ndarray[complex]
        S Transform spectrogram array.

    Notes
    -----
    The S Transform is computed using the inverse FFT.
    """
    ts_arr = np.asarray(ts, dtype=float)
    n_samples = ts_arr.size
    fmin, fmax = freq_range
    n0 = int(fmin * n_samples / sample_rate)
    n1 = int(fmax * n_samples / sample_rate)
    n_bins = abs(n1 - n0)
    step = int(np.ceil(freq_step * n_samples / sample_rate))

    ts_fft = fft.fft(ts_arr)

    if downsample is None:
        ts_fft_cut = np.concatenate((ts_fft, ts_fft))
        n_cut = ts_fft.size
        norm = 1.0
    else:
        ts_fft_pos = ts_fft[:n1]
        ts_fft_neg = ts_fft[-n1:]
        if downsample < 2 * n_bins:
            # max allowed lower/higher freq cut-off
            pass
        else:
            pad_len = downsample // 2 - n_bins
            ts_fft_pos = np.concatenate((ts_fft_pos, np.zeros(pad_len)))
            ts_fft_neg = np.concatenate((np.zeros(pad_len), ts_fft_neg))
        ts_fft_cut = np.concatenate((ts_fft_pos, ts_fft_neg))
        n_cut = ts_fft_cut.size
        norm = 1.0 - n_cut / n_samples

    ST = np.zeros((int(n_bins / step) + 1, n_cut), dtype=np.complex128)
    vec = ts_fft_cut

    for i in range(step, n_bins + 1, step):
        window = _window_normal(n_cut, alpha + n0 + i)
        segment = vec[n0 + i : n0 + i + n_cut]
        ST[int(i / step)] = fft.ifft(segment * window * norm)

    return ST


def _window_normal(
    length: int,
    freq: float,
    factor: float = 1.0
) -> NDArray[np.float64]:
    """
    Generate a split Gaussian window for S Transform convolution.

    Parameters
    ----------
    length : int
        Length of the Gaussian window.
    freq : float
        Frequency at which this window is applied.
    factor : float, optional
        Normalizing factor for the Gaussian, default is 1.0.

    Returns
    -------
    ndarray[float]
        Split Gaussian window.
    """
    gauss = signal.windows.gaussian(length, std=freq / (2 * np.pi)) * factor
    return np.hstack((gauss, gauss))[length // 2 : length // 2 + length]

def inverse_s_transform(
    table: ArrayLike,
    low_freq: Optional[int] = None
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Perform the True Inverse S Transform.

    Parameters
    ----------
    table : array_like
        Spectrogram table generated using `s_transform`.
    low_freq : int, optional
        Starting frequency (not yet supported).

    Returns
    -------
    tuple of
        - ndarray[complex]: Recovered time series.
        - ndarray[complex]: Recovered FFT (one-sided).
    """
    T = np.asarray(table, dtype=np.complex128)
    n_freq, n_time = T.shape
    fft_rec = np.zeros(n_time, dtype=np.complex128)

    for idx in range(n_freq):
        fft_rec[idx] = fft.fft(T[idx])[0]

    ts_rec = fft.ifft(fft_rec)
    return ts_rec, fft_rec
