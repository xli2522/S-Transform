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

"""Discrete Stockwell (S) transform utilities.

The forward transform returns positive-frequency rows. When the input is real
and the complete positive-frequency range is requested, the inverse can
reconstruct the original signal by restoring conjugate negative-frequency
bins.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft, signal

__all__ = ["STransformArray", "s_transform", "inverse_s_transform"]


class STransformArray(np.ndarray):
    """S-transform data with metadata required for interpretation/inversion.

    The object behaves like a regular two-dimensional complex NumPy array, so
    existing code using ``np.abs(st)`` continues to work. The metadata avoids
    making callers reverse-engineer frequency-bin spacing from the array shape.
    """

    frequency_indices: NDArray[np.int64]
    sample_rate: float
    input_length: int
    real_input: bool

    def __new__(
        cls,
        data: ArrayLike,
        *,
        frequency_indices: ArrayLike,
        sample_rate: float,
        input_length: int,
        real_input: bool,
    ) -> "STransformArray":
        obj = np.asarray(data, dtype=np.complex128).view(cls)
        obj.frequency_indices = np.asarray(frequency_indices, dtype=np.int64)
        obj.sample_rate = float(sample_rate)
        obj.input_length = int(input_length)
        obj.real_input = bool(real_input)
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None:
            return
        self.frequency_indices = getattr(obj, "frequency_indices", None)
        self.sample_rate = getattr(obj, "sample_rate", None)
        self.input_length = getattr(obj, "input_length", None)
        self.real_input = getattr(obj, "real_input", None)

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """Actual frequency represented by each transform row, in hertz."""
        return self.frequency_indices * self.sample_rate / self.input_length

    @property
    def times(self) -> NDArray[np.float64]:
        """Time coordinate represented by each transform column, in seconds."""
        return np.arange(self.shape[1], dtype=float) * self.input_length / (
            self.shape[1] * self.sample_rate
        )


def _validate_input(
    ts: ArrayLike,
    sample_rate: float,
    freq_range: Optional[Sequence[float]],
    freq_step: Optional[float],
    alpha: float,
    downsample: Optional[int],
) -> Tuple[NDArray[np.complex128], float, float, float, float, Optional[int]]:
    """Validate arguments and return normalized numeric values."""
    ts_arr = np.asarray(ts)
    if ts_arr.ndim != 1:
        raise ValueError("ts must be a one-dimensional time series")
    if ts_arr.size == 0:
        raise ValueError("ts must contain at least one sample")

    try:
        sample_rate = float(sample_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("sample_rate must be a positive real number") from exc
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive finite number")

    if np.iscomplexobj(ts_arr):
        ts_arr = np.asarray(ts_arr, dtype=np.complex128)
    else:
        ts_arr = np.asarray(ts_arr, dtype=np.float64).astype(np.complex128)

    nyquist = sample_rate / 2.0
    if freq_range is None:
        fmin, fmax = 0.0, nyquist
    else:
        if len(freq_range) != 2:
            raise ValueError("freq_range must contain exactly two values")
        try:
            fmin, fmax = (float(freq_range[0]), float(freq_range[1]))
        except (TypeError, ValueError) as exc:
            raise TypeError("freq_range values must be real numbers") from exc

    if not np.isfinite(fmin) or not np.isfinite(fmax):
        raise ValueError("freq_range values must be finite")
    if fmin < 0 or fmax > nyquist or fmin >= fmax:
        raise ValueError(
            f"freq_range must satisfy 0 <= fmin < fmax <= Nyquist ({nyquist:g} Hz)"
        )

    if freq_step is None:
        freq_step = sample_rate / ts_arr.size
    try:
        freq_step = float(freq_step)
    except (TypeError, ValueError) as exc:
        raise TypeError("freq_step must be a positive real number") from exc
    if not np.isfinite(freq_step) or freq_step <= 0:
        raise ValueError("freq_step must be a positive finite number")

    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError("alpha must be a positive real number") from exc
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number")

    if downsample is not None:
        if isinstance(downsample, bool) or not isinstance(
            downsample, (int, np.integer)
        ):
            raise TypeError("downsample must be a positive integer or None")
        downsample = int(downsample)
        if downsample <= 0:
            raise ValueError("downsample must be a positive integer or None")

    return ts_arr, sample_rate, fmin, fmax, freq_step, downsample


def _window_normal(
    length: int,
    freq: float,
    factor: float = 1.0,
) -> NDArray[np.float64]:
    """Return the periodized Gaussian frequency-domain window.

    ``freq`` is an FFT-bin frequency, not hertz. The signed-bin layout puts
    zero lag at index zero and negative lags at the end of the array, which is
    the layout expected by the FFT convolution formula.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if not np.isfinite(freq) or freq <= 0:
        raise ValueError("freq must be a positive finite FFT-bin frequency")

    bins = np.arange(length, dtype=float)
    signed_bins = np.where(bins <= length // 2, bins, bins - length)
    return factor * np.exp(-2.0 * np.pi**2 * (signed_bins / freq) ** 2)


def s_transform(
    ts: ArrayLike,
    sample_rate: float,
    freq_range: Optional[Sequence[float]] = None,
    freq_step: Optional[float] = None,
    alpha: float = 1.0,
    downsample: Optional[int] = None,
) -> STransformArray:
    """Compute the positive-frequency discrete Stockwell transform.

    Parameters
    ----------
    ts : array_like, shape (n_samples,)
        Real or complex time-series samples.
    sample_rate : float
        Sample rate in hertz.
    freq_range : sequence of float, optional
        Positive-frequency range in hertz. The default is ``(0, Nyquist)``.
    freq_step : float, optional
        Requested frequency spacing in hertz. The actual spacing is rounded
        up to an integer number of FFT bins. The default is one FFT bin.
    alpha : float, optional
        Positive Gaussian width multiplier. ``alpha=1`` is the standard
        Stockwell window; larger values produce narrower windows.
    downsample : int, optional
        If supplied, resample the time axis to this many columns after the
        transform. This is time-axis resampling and is not lossless when it
        reduces the number of columns.

    Returns
    -------
    STransformArray
        Complex array of shape ``(n_frequencies, n_time)``. The returned array
        exposes ``frequency_indices``, ``frequencies``, ``times``,
        ``sample_rate``, ``input_length`` and ``real_input`` metadata.
    """
    ts_arr, sample_rate, fmin, fmax, freq_step, downsample = _validate_input(
        ts, sample_rate, freq_range, freq_step, alpha, downsample
    )
    n_samples = ts_arr.size
    frequency_resolution = sample_rate / n_samples
    max_bin = n_samples // 2

    # Use bins wholly inside the requested range. A requested spacing below
    # one FFT bin is represented by the finest available spacing.
    start_bin = max(0, int(np.ceil(fmin / frequency_resolution - 1e-12)))
    stop_bin = min(max_bin, int(np.floor(fmax / frequency_resolution + 1e-12)))
    if start_bin > stop_bin:
        raise ValueError("freq_range does not contain an available FFT bin")
    step_bins = max(1, int(np.ceil(freq_step / frequency_resolution - 1e-12)))
    frequency_indices = np.arange(start_bin, stop_bin + 1, step_bins, dtype=np.int64)

    spectrum = fft.fft(ts_arr)
    rows = []
    for frequency_index in frequency_indices:
        if frequency_index == 0:
            # The zero-frequency S-transform row is the signal mean.
            row = np.full(n_samples, spectrum[0] / n_samples, dtype=np.complex128)
        else:
            shifted_spectrum = spectrum[
                (frequency_index + np.arange(n_samples, dtype=np.int64)) % n_samples
            ]
            window = _window_normal(
                n_samples,
                frequency_index / alpha,
            )
            row = fft.ifft(shifted_spectrum * window)
        rows.append(row)

    transform = np.asarray(rows, dtype=np.complex128)
    if downsample is not None and downsample != n_samples:
        transform = signal.resample(transform, downsample, axis=1)

    return STransformArray(
        transform,
        frequency_indices=frequency_indices,
        sample_rate=sample_rate,
        input_length=n_samples,
        real_input=not np.iscomplexobj(ts),
    )


def inverse_s_transform(
    table: ArrayLike,
    low_freq: Optional[int] = None,
    *,
    frequency_indices: Optional[ArrayLike] = None,
    original_length: Optional[int] = None,
    real_output: Optional[bool] = None,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Reconstruct a signal from an S-transform table.

    When ``table`` is the result of :func:`s_transform`, its frequency-bin
    metadata is used automatically. For a plain ndarray, rows are assumed to
    represent contiguous FFT bins beginning at ``low_freq`` (default zero).
    Exact reconstruction requires all positive-frequency bins through Nyquist,
    no time downsampling, and a real input. Partial frequency ranges and
    downsampled tables produce the corresponding band-limited approximation.

    Parameters
    ----------
    table : array_like, shape (n_frequencies, n_time)
        S-transform table.
    low_freq : int, optional
        Backward-compatible starting FFT bin for plain ndarray input.
    frequency_indices : array_like, optional
        FFT-bin index for each row when ``table`` is a plain ndarray.
    original_length : int, optional
        Length of the original signal. Required for a downsampled plain
        ndarray and inferred from ``STransformArray`` metadata otherwise.
    real_output : bool, optional
        Mirror positive-frequency coefficients when true. Inferred from the
        forward-transform metadata when available.
    """
    if low_freq is not None and frequency_indices is not None:
        raise ValueError("provide either low_freq or frequency_indices, not both")

    metadata = table if isinstance(table, STransformArray) else None
    table_arr = np.asarray(table, dtype=np.complex128)
    if table_arr.ndim != 2:
        raise ValueError("table must be a two-dimensional array")
    n_frequencies, n_time = table_arr.shape

    if metadata is not None:
        metadata_indices = metadata.frequency_indices
        if frequency_indices is not None and not np.array_equal(
            np.asarray(frequency_indices), metadata_indices
        ):
            raise ValueError("frequency_indices do not match table metadata")
        frequency_indices_arr = metadata_indices
        inferred_length = metadata.input_length
        inferred_real = metadata.real_input
    else:
        if frequency_indices is None:
            start = 0 if low_freq is None else int(low_freq)
            frequency_indices_arr = np.arange(start, start + n_frequencies)
        else:
            frequency_indices_arr = np.asarray(frequency_indices, dtype=np.int64)
        inferred_length = n_time
        inferred_real = False

    frequency_indices_arr = np.asarray(frequency_indices_arr, dtype=np.int64)
    if frequency_indices_arr.ndim != 1 or frequency_indices_arr.size != n_frequencies:
        raise ValueError("frequency_indices must contain one FFT bin per table row")

    if original_length is None:
        original_length = inferred_length
    if isinstance(original_length, bool) or not isinstance(
        original_length, (int, np.integer)
    ):
        raise TypeError("original_length must be a positive integer")
    original_length = int(original_length)
    if original_length <= 0:
        raise ValueError("original_length must be a positive integer")

    if np.any(frequency_indices_arr < 0) or np.any(
        frequency_indices_arr >= original_length
    ):
        raise ValueError("frequency_indices contain bins outside original_length")

    if real_output is None:
        real_output = inferred_real
    real_output = bool(real_output)

    # A downsampled table no longer has the original row length. Resampling
    # rows back before taking their zero-frequency FFT coefficient preserves
    # the transform coefficient as closely as the time-axis resampling allows.
    if n_time != original_length:
        table_arr = signal.resample(table_arr, original_length, axis=1)

    recovered_spectrum = np.zeros(original_length, dtype=np.complex128)
    for row_index, frequency_index in enumerate(frequency_indices_arr):
        coefficient = fft.fft(table_arr[row_index])[0]
        recovered_spectrum[frequency_index] = coefficient

        if real_output and frequency_index != 0:
            is_nyquist = (
                original_length % 2 == 0
                and frequency_index == original_length // 2
            )
            if not is_nyquist:
                recovered_spectrum[-frequency_index] = np.conj(coefficient)

    recovered_signal = fft.ifft(recovered_spectrum)
    return recovered_signal, recovered_spectrum
