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

The forward transform returns rows for the requested FFT frequencies. By
default this is the canonical nonnegative range. When the input is real and
the complete positive-frequency range is requested, the inverse can
reconstruct the original signal by restoring conjugate negative-frequency
bins.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft, signal

__all__ = ["STransformArray", "s_transform", "inverse_s_transform"]


class InverseKwargs(TypedDict):
    """Keyword arguments packaged for inversion of a plain ndarray copy."""

    frequency_indices: NDArray[np.int64]
    original_length: int
    real_output: bool


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
        data_arr = np.asarray(data, dtype=np.complex128)
        if data_arr.ndim != 2:
            raise ValueError("data must be a two-dimensional transform")
        if 0 in data_arr.shape:
            raise ValueError("data must have non-empty frequency and time axes")

        frequency_indices_arr = np.asarray(frequency_indices, dtype=np.int64)
        if frequency_indices_arr.ndim != 1:
            raise ValueError("frequency_indices must be one-dimensional")
        if frequency_indices_arr.size != data_arr.shape[0]:
            raise ValueError(
                "frequency_indices must contain one FFT bin per transform row"
            )

        obj = data_arr.view(cls)
        obj.frequency_indices = frequency_indices_arr
        obj.sample_rate = float(sample_rate)
        obj.input_length = int(input_length)
        obj.real_input = bool(real_input)
        obj._metadata_valid = True
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None:
            return
        # A view may slice, reorder, or transpose rows and columns. There is
        # no generally safe way to update the metadata for all such
        # operations, so derived arrays must opt back in with explicit
        # metadata when they are passed to the inverse.
        self.frequency_indices = None
        self.sample_rate = None
        self.input_length = None
        self.real_input = None
        self._metadata_valid = False

    def _require_metadata(self) -> None:
        if not getattr(self, "_metadata_valid", False):
            raise ValueError(
                "transform metadata is unavailable after an array operation; "
                "use the original transform or pass explicit metadata"
            )

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """Actual frequency represented by each transform row, in hertz."""
        self._require_metadata()
        return self.signed_frequency_indices * self.sample_rate / self.input_length

    @property
    def signed_frequency_indices(self) -> NDArray[np.int64]:
        """Signed FFT-bin indices represented by each transform row."""
        self._require_metadata()
        return np.where(
            self.frequency_indices <= self.input_length // 2,
            self.frequency_indices,
            self.frequency_indices - self.input_length,
        )

    @property
    def times(self) -> NDArray[np.float64]:
        """Time coordinate represented by each transform column, in seconds."""
        self._require_metadata()
        return np.arange(self.shape[1], dtype=float) * self.input_length / (
            self.shape[1] * self.sample_rate
        )

    @property
    def duration(self) -> float:
        """Duration of the original signal, in seconds."""
        self._require_metadata()
        return self.input_length / self.sample_rate

    @property
    def time_step(self) -> float:
        """Spacing between adjacent transform columns, in seconds."""
        self._require_metadata()
        return self.duration / self.shape[1]

    @property
    def frequency_resolution(self) -> float:
        """Spacing between adjacent raw FFT bins, in hertz."""
        self._require_metadata()
        return self.sample_rate / self.input_length

    @property
    def frequency_range(self) -> Tuple[float, float]:
        """Minimum and maximum represented frequencies, in hertz."""
        self._require_metadata()
        nyquist_index = self.input_length // 2
        minimum = maximum = int(self.frequency_indices[0])
        if minimum > nyquist_index:
            minimum -= self.input_length
            maximum = minimum
        for raw_index in self.frequency_indices[1:]:
            signed_index = int(raw_index)
            if signed_index > nyquist_index:
                signed_index -= self.input_length
            minimum = min(minimum, signed_index)
            maximum = max(maximum, signed_index)
        scale = self.frequency_resolution
        return minimum * scale, maximum * scale

    @property
    def time_range(self) -> Tuple[float, float]:
        """First and last represented time coordinates, in seconds."""
        self._require_metadata()
        return 0.0, (self.shape[1] - 1) * self.time_step

    @property
    def is_downsampled(self) -> bool:
        """Whether the transform has fewer time columns than the input."""
        self._require_metadata()
        return self.shape[1] < self.input_length

    @property
    def is_upsampled(self) -> bool:
        """Whether the transform has more time columns than the input."""
        self._require_metadata()
        return self.shape[1] > self.input_length

    @property
    def inverse_kwargs(self) -> InverseKwargs:
        """Metadata arguments for inverting a plain ndarray copy."""
        self._require_metadata()
        return {
            "frequency_indices": self.frequency_indices.copy(),
            "original_length": self.input_length,
            "real_output": self.real_input,
        }


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
        try:
            if len(freq_range) != 2:
                raise ValueError("freq_range must contain exactly two values")
        except TypeError as exc:
            raise TypeError("freq_range must be a two-item sequence") from exc
        try:
            fmin, fmax = (float(freq_range[0]), float(freq_range[1]))
        except (TypeError, ValueError) as exc:
            raise TypeError("freq_range values must be real numbers") from exc

    if not np.isfinite(fmin) or not np.isfinite(fmax):
        raise ValueError("freq_range values must be finite")
    if fmin < -nyquist or fmax > nyquist or fmin >= fmax:
        raise ValueError(
            f"freq_range must satisfy -Nyquist <= fmin < fmax <= Nyquist "
            f"({nyquist:g} Hz)"
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


def _as_integer_scalar(value: object, name: str) -> int:
    """Validate an integer-valued scalar without silently truncating it."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _as_integer_array(values: ArrayLike, name: str) -> NDArray[np.int64]:
    """Validate a one-dimensional integer-valued array."""
    values_arr = np.asarray(values)
    if values_arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(values_arr.dtype, np.number):
        raise TypeError(f"{name} must contain numbers")
    if np.issubdtype(values_arr.dtype, np.complexfloating):
        if np.any(values_arr.imag != 0):
            raise ValueError(f"{name} must contain real integers")
        values_arr = values_arr.real
    if not np.all(np.isfinite(values_arr)):
        raise ValueError(f"{name} must contain finite integers")
    if not np.all(values_arr == np.floor(values_arr)):
        raise ValueError(f"{name} must contain integer values")
    int64_info = np.iinfo(np.int64)
    if np.any(values_arr < int64_info.min) or np.any(values_arr > int64_info.max):
        raise ValueError(f"{name} contains values outside the int64 range")
    return values_arr.astype(np.int64)


def _normalize_frequency_indices(
    values: ArrayLike,
    original_length: int,
    name: str,
) -> NDArray[np.int64]:
    """Convert raw or signed FFT-bin indices to canonical modulo-N bins."""
    indices = _as_integer_array(values, name)
    minimum_signed_bin = -(original_length // 2)
    if np.any(indices < minimum_signed_bin) or np.any(indices >= original_length):
        raise ValueError(
            f"{name} must contain bins in the signed/raw range "
            f"[{minimum_signed_bin}, {original_length - 1}]"
        )
    return np.mod(indices, original_length)


def s_transform(
    ts: ArrayLike,
    sample_rate: float,
    freq_range: Optional[Sequence[float]] = None,
    freq_step: Optional[float] = None,
    alpha: float = 1.0,
    downsample: Optional[int] = None,
    max_output_elements: Optional[int] = 20_000_000,
) -> STransformArray:
    """Compute the discrete Stockwell transform over a signed frequency range.

    Parameters
    ----------
    ts : array_like, shape (n_samples,)
        Real or complex time-series samples.
    sample_rate : float
        Sample rate in hertz.
    freq_range : sequence of float, optional
        Signed frequency range in hertz. The default is ``(0, Nyquist)``;
        negative values select negative-frequency FFT bins.
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
    max_output_elements : int or None, optional
        Safety limit for the number of complex output elements. Set to
        ``None`` to opt into an unbounded allocation for large transforms.

    Returns
    -------
    STransformArray
        Complex array of shape ``(n_frequencies, n_time)``. The returned array
        exposes ``frequency_indices``, ``signed_frequency_indices``,
        ``frequencies``, ``times``,
        ``sample_rate``, ``input_length`` and ``real_input`` metadata.
    """
    ts_arr, sample_rate, fmin, fmax, freq_step, downsample = _validate_input(
        ts, sample_rate, freq_range, freq_step, alpha, downsample
    )
    n_samples = ts_arr.size
    frequency_resolution = sample_rate / n_samples
    max_bin = n_samples // 2

    # Use canonical signed bins. For even lengths, the Nyquist bin has only
    # one representation and is assigned the positive sign.
    min_signed_bin = -((n_samples - 1) // 2)
    max_signed_bin = n_samples // 2
    start_bin = max(
        min_signed_bin,
        int(np.ceil(fmin / frequency_resolution - 1e-12)),
    )
    stop_bin = min(
        max_signed_bin,
        int(np.floor(fmax / frequency_resolution + 1e-12)),
    )
    if start_bin > stop_bin:
        raise ValueError("freq_range does not contain an available FFT bin")
    step_bins = max(1, int(np.ceil(freq_step / frequency_resolution - 1e-12)))
    signed_frequency_indices = np.arange(
        start_bin,
        stop_bin + 1,
        step_bins,
        dtype=np.int64,
    )
    frequency_indices = np.mod(signed_frequency_indices, n_samples)

    if max_output_elements is not None:
        max_output_elements = _as_integer_scalar(
            max_output_elements, "max_output_elements"
        )
        if max_output_elements <= 0:
            raise ValueError("max_output_elements must be positive or None")
    output_length = n_samples if downsample is None else downsample
    output_elements = frequency_indices.size * output_length
    if (
        max_output_elements is not None
        and output_elements > max_output_elements
    ):
        raise MemoryError(
            f"requested transform has {output_elements:,} complex elements, "
            f"which exceeds max_output_elements={max_output_elements:,}; "
            "reduce the frequency range/downsample or pass "
            "max_output_elements=None"
        )

    spectrum = fft.fft(ts_arr)
    transform = np.empty(
        (frequency_indices.size, output_length),
        dtype=np.complex128,
    )
    for row_index, frequency_index in enumerate(frequency_indices):
        if frequency_index == 0:
            # The zero-frequency S-transform row is the signal mean.
            row = np.full(n_samples, spectrum[0] / n_samples, dtype=np.complex128)
        else:
            signed_frequency_index = (
                frequency_index
                if frequency_index <= max_bin
                else frequency_index - n_samples
            )
            shifted_spectrum = spectrum[
                (frequency_index + np.arange(n_samples, dtype=np.int64)) % n_samples
            ]
            window = _window_normal(
                n_samples,
                abs(signed_frequency_index) / alpha,
            )
            row = fft.ifft(shifted_spectrum * window)
        if output_length == n_samples:
            transform[row_index] = row
        else:
            transform[row_index] = signal.resample(row, output_length)

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
    For real input, exact reconstruction requires all nonnegative-frequency
    bins through Nyquist and no time downsampling. For complex input, request
    the complete canonical signed-frequency range instead. Partial frequency
    ranges and downsampled tables produce the corresponding band-limited
    approximation.

    Parameters
    ----------
    table : array_like, shape (n_frequencies, n_time)
        S-transform table.
    low_freq : int, optional
        Starting signed/raw FFT bin for plain ndarray input. Negative values
        select negative-frequency bins.
    frequency_indices : array_like, optional
        FFT-bin index for each row when ``table`` is a plain ndarray. Raw
        modulo-``N`` indices and signed indices are both accepted.
    original_length : int, optional
        Length of the original signal. Required for a plain ndarray and
        inferred from ``STransformArray`` metadata otherwise.
    real_output : bool, optional
        Mirror positive-frequency coefficients when true. Inferred from the
        forward-transform metadata when available.
    """
    if low_freq is not None and frequency_indices is not None:
        raise ValueError("provide either low_freq or frequency_indices, not both")

    metadata = (
        table
        if isinstance(table, STransformArray)
        and getattr(table, "_metadata_valid", False)
        else None
    )
    if isinstance(table, STransformArray) and metadata is None:
        if frequency_indices is None:
            raise ValueError(
                "transform metadata is unavailable; pass frequency_indices "
                "and original_length explicitly"
            )
    table_arr = np.asarray(table, dtype=np.complex128)
    if table_arr.ndim != 2:
        raise ValueError("table must be a two-dimensional array")
    n_frequencies, n_time = table_arr.shape

    if metadata is not None:
        inferred_length = _as_integer_scalar(metadata.input_length, "input_length")
    else:
        inferred_length = None
        if original_length is None:
            raise ValueError(
                "original_length is required when inverting a plain ndarray"
            )

    if original_length is None:
        original_length = inferred_length
    original_length = _as_integer_scalar(original_length, "original_length")
    if original_length <= 0:
        raise ValueError("original_length must be a positive integer")
    if metadata is not None and original_length != inferred_length:
        raise ValueError("original_length does not match the transform metadata")

    if metadata is not None:
        metadata_indices = _normalize_frequency_indices(
            metadata.frequency_indices,
            original_length,
            "frequency_indices",
        )
        if frequency_indices is not None and not np.array_equal(
            _normalize_frequency_indices(
                frequency_indices,
                original_length,
                "frequency_indices",
            ),
            metadata_indices,
        ):
            raise ValueError("frequency_indices do not match table metadata")
        frequency_indices_arr = metadata_indices.copy()
        inferred_real = metadata.real_input
    else:
        if frequency_indices is None:
            start = 0 if low_freq is None else _as_integer_scalar(low_freq, "low_freq")
            provided_indices = np.arange(
                start,
                start + n_frequencies,
                dtype=np.int64,
            )
        else:
            provided_indices = frequency_indices
        frequency_indices_arr = _normalize_frequency_indices(
            provided_indices,
            original_length,
            "frequency_indices",
        )
        inferred_real = False

    if frequency_indices_arr.ndim != 1 or frequency_indices_arr.size != n_frequencies:
        raise ValueError("frequency_indices must contain one FFT bin per table row")

    if np.any(frequency_indices_arr < 0) or np.any(
        frequency_indices_arr >= original_length
    ):
        raise ValueError("frequency_indices contain bins outside original_length")
    if np.unique(frequency_indices_arr).size != frequency_indices_arr.size:
        raise ValueError("frequency_indices must not contain duplicates")

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
