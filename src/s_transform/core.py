"""Discrete Stockwell (S) transform utilities.

The public API returns :class:`STransform`, a small metadata wrapper around a
two-dimensional complex NumPy array.  The wrapper keeps the frequency-bin
layout and original signal information needed to interpret or invert a
transform without relying on ndarray subclass behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft, signal  # type: ignore[import-untyped]

__all__ = ["STransform", "s_transform", "inverse_s_transform"]

_SCRATCH_MEMORY_BUDGET_BYTES = 32 * 1024 * 1024
_BOUNDARY_PADDING_SIGMAS = 6.0

BoundaryMode = Literal["periodic", "zeros"]


class InverseKwargs(TypedDict):
    """Keyword arguments packaged for inversion of a plain ndarray copy."""

    frequency_indices: NDArray[np.int64]
    original_length: int
    real_output: bool
    boundary: BoundaryMode
    inverse_compatible: bool


def _as_integer_scalar(value: object, name: str) -> int:
    """Validate an integer-valued scalar without silently truncating it."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _as_bool_scalar(value: object, name: str) -> bool:
    """Validate a boolean scalar without coercing arbitrary values."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _normalize_boundary(value: object) -> BoundaryMode:
    """Validate a supported S-transform boundary condition."""
    if not isinstance(value, str):
        raise ValueError("boundary must be 'periodic' or 'zeros'")
    if value == "periodic":
        return "periodic"
    if value == "zeros":
        return "zeros"
    raise ValueError("boundary must be 'periodic' or 'zeros'")


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


def _validate_transform_metadata(
    data: ArrayLike,
    frequency_indices: ArrayLike,
    sample_rate: float,
    input_length: int,
    real_input: bool,
    transform_length: int | None,
    boundary: BoundaryMode,
    boundary_padding: int,
    inverse_compatible: bool,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.int64],
    float,
    int,
    bool,
    int,
    BoundaryMode,
    int,
    bool,
]:
    """Validate and normalize data and metadata for an ``STransform``."""
    data_arr = np.asarray(data, dtype=np.complex128)
    if data_arr.ndim != 2:
        raise ValueError("data must be a two-dimensional transform")
    if 0 in data_arr.shape:
        raise ValueError("data must have non-empty frequency and time axes")

    try:
        sample_rate_value = float(sample_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("sample_rate must be a positive real number") from exc
    if not np.isfinite(sample_rate_value) or sample_rate_value <= 0:
        raise ValueError("sample_rate must be a positive finite number")

    input_length_value = _as_integer_scalar(input_length, "input_length")
    if input_length_value <= 0:
        raise ValueError("input_length must be a positive integer")

    transform_length_value = (
        input_length_value
        if transform_length is None
        else _as_integer_scalar(transform_length, "transform_length")
    )
    if transform_length_value <= 0:
        raise ValueError("transform_length must be a positive integer")
    boundary_value = _normalize_boundary(boundary)
    boundary_padding_value = _as_integer_scalar(
        boundary_padding,
        "boundary_padding",
    )
    if boundary_padding_value < 0:
        raise ValueError("boundary_padding must be non-negative")
    inverse_compatible_value = _as_bool_scalar(
        inverse_compatible,
        "inverse_compatible",
    )

    if boundary_value == "periodic":
        if transform_length_value != input_length_value:
            raise ValueError(
                "periodic transforms must have transform_length equal to input_length"
            )
        if boundary_padding_value != 0 or not inverse_compatible_value:
            raise ValueError(
                "periodic transforms cannot have boundary padding or be marked "
                "inverse-incompatible"
            )
    else:
        if transform_length_value != input_length_value + 2 * boundary_padding_value:
            raise ValueError(
                "zero-boundary transform_length must include symmetric boundary_padding"
            )
        if inverse_compatible_value:
            raise ValueError(
                "zero-boundary transforms cropped to the input interval are not "
                "compatible with inverse_s_transform"
            )

    frequency_indices_arr = _normalize_frequency_indices(
        frequency_indices,
        transform_length_value,
        "frequency_indices",
    )
    if frequency_indices_arr.size != data_arr.shape[0]:
        raise ValueError(
            "frequency_indices must contain one FFT bin per transform row"
        )
    if np.unique(frequency_indices_arr).size != frequency_indices_arr.size:
        raise ValueError("frequency_indices must not contain duplicates")

    real_input_value = _as_bool_scalar(real_input, "real_input")
    return (
        data_arr,
        frequency_indices_arr,
        sample_rate_value,
        input_length_value,
        real_input_value,
        transform_length_value,
        boundary_value,
        boundary_padding_value,
        inverse_compatible_value,
    )


@dataclass(frozen=True, eq=False)
class STransform:
    """S-transform data with validated metadata.

    ``data`` is a complex array with shape ``(n_frequencies, n_time)``.
    Metadata is stored separately from the numerical array so ordinary copies,
    slicing, and pickling do not silently alter or discard the transform's
    interpretation. ``input_length`` describes the returned time interval;
    ``transform_length`` describes the FFT period used for frequency-bin
    metadata and can be longer for zero-boundary transforms.
    """

    data: NDArray[np.complex128]
    frequency_indices: NDArray[np.int64]
    sample_rate: float
    input_length: int
    real_input: bool
    transform_length: int | None = None
    boundary: BoundaryMode = "periodic"
    boundary_padding: int = 0
    inverse_compatible: bool = True

    def __hash__(self) -> int:
        """Keep the wrapper unhashable because it owns mutable arrays."""
        raise TypeError(f"unhashable type: {type(self).__name__!r}")

    @property
    def _fft_length(self) -> int:
        """FFT length, normalized to ``input_length`` after construction."""
        assert self.transform_length is not None
        return self.transform_length

    def __post_init__(self) -> None:
        (
            data,
            frequency_indices,
            sample_rate,
            input_length,
            real_input,
            transform_length,
            boundary,
            boundary_padding,
            inverse_compatible,
        ) = _validate_transform_metadata(
            self.data,
            self.frequency_indices,
            self.sample_rate,
            self.input_length,
            self.real_input,
            self.transform_length,
            self.boundary,
            self.boundary_padding,
            self.inverse_compatible,
        )
        frequency_indices = frequency_indices.copy()
        frequency_indices.setflags(write=False)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "frequency_indices", frequency_indices)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "input_length", input_length)
        object.__setattr__(self, "real_input", real_input)
        object.__setattr__(self, "transform_length", transform_length)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "boundary_padding", boundary_padding)
        object.__setattr__(self, "inverse_compatible", inverse_compatible)

    def __eq__(self, other: object) -> bool:
        """Compare numerical data and metadata without ndarray ambiguity."""
        if not isinstance(other, STransform):
            return False
        return bool(
            np.array_equal(self.data, other.data)
            and np.array_equal(self.frequency_indices, other.frequency_indices)
            and self.sample_rate == other.sample_rate
            and self.input_length == other.input_length
            and self.real_input == other.real_input
            and self.transform_length == other.transform_length
            and self.boundary == other.boundary
            and self.boundary_padding == other.boundary_padding
            and self.inverse_compatible == other.inverse_compatible
        )

    def __array__(
        self,
        dtype: np.dtype | None = None,
        copy: bool | None = None,
    ) -> NDArray[np.complex128]:
        """Return the numerical transform for NumPy conversion."""
        array = np.asarray(self.data, dtype=dtype)
        if copy is True:
            return array.copy()
        return array

    @property
    def signed_frequency_indices(self) -> NDArray[np.int64]:
        """Signed FFT-bin indices represented by each transform row."""
        return np.where(
            self.frequency_indices <= self._fft_length // 2,
            self.frequency_indices,
            self.frequency_indices - self._fft_length,
        )

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """Actual frequency represented by each transform row, in hertz."""
        return np.asarray(
            self.signed_frequency_indices * self.frequency_resolution,
            dtype=np.float64,
        )

    @property
    def times(self) -> NDArray[np.float64]:
        """Time coordinate represented by each transform column, in seconds."""
        return np.arange(self.data.shape[1], dtype=float) * self.time_step

    @property
    def duration(self) -> float:
        """Duration of the original signal, in seconds."""
        return self.input_length / self.sample_rate

    @property
    def time_step(self) -> float:
        """Spacing between adjacent transform columns, in seconds."""
        return self.duration / self.data.shape[1]

    @property
    def frequency_resolution(self) -> float:
        """Spacing between adjacent raw FFT bins, in hertz."""
        return self.sample_rate / self._fft_length

    @property
    def frequency_range(self) -> tuple[float, float]:
        """Minimum and maximum represented frequencies, in hertz."""
        nyquist_index = self._fft_length // 2
        minimum = maximum = int(self.frequency_indices[0])
        if minimum > nyquist_index:
            minimum -= self._fft_length
            maximum = minimum
        for raw_index in self.frequency_indices[1:]:
            signed_index = int(raw_index)
            if signed_index > nyquist_index:
                signed_index -= self._fft_length
            minimum = min(minimum, signed_index)
            maximum = max(maximum, signed_index)
        scale = self.frequency_resolution
        return minimum * scale, maximum * scale

    @property
    def time_range(self) -> tuple[float, float]:
        """First and last represented time coordinates, in seconds."""
        return 0.0, (self.data.shape[1] - 1) * self.time_step

    @property
    def is_downsampled(self) -> bool:
        """Whether the transform has fewer time columns than the input."""
        return self.data.shape[1] < self.input_length

    @property
    def is_upsampled(self) -> bool:
        """Whether the transform has more time columns than the input."""
        return self.data.shape[1] > self.input_length

    @property
    def inverse_kwargs(self) -> InverseKwargs:
        """Metadata arguments for inverting a plain ndarray copy."""
        return {
            "frequency_indices": self.frequency_indices.copy(),
            "original_length": self._fft_length,
            "real_output": self.real_input,
            "boundary": self.boundary,
            "inverse_compatible": self.inverse_compatible,
        }


def _validate_input(
    ts: ArrayLike,
    sample_rate: float,
    frequency_range: Sequence[float] | None,
    frequency_step: float | None,
    alpha: float,
    output_length: int | None,
) -> tuple[NDArray[np.complex128], float, float, float, float, int | None]:
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
    if frequency_range is None:
        fmin, fmax = 0.0, nyquist
    else:
        try:
            if len(frequency_range) != 2:
                raise ValueError("frequency_range must contain exactly two values")
        except TypeError as exc:
            raise TypeError("frequency_range must be a two-item sequence") from exc
        try:
            fmin, fmax = (
                float(frequency_range[0]),
                float(frequency_range[1]),
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("frequency_range values must be real numbers") from exc

    if not np.isfinite(fmin) or not np.isfinite(fmax):
        raise ValueError("frequency_range values must be finite")
    if fmin < -nyquist or fmax > nyquist or fmin >= fmax:
        raise ValueError(
            f"frequency_range must satisfy -Nyquist <= fmin < fmax <= Nyquist "
            f"({nyquist:g} Hz)"
        )

    if frequency_step is None:
        frequency_step = sample_rate / ts_arr.size
    try:
        frequency_step = float(frequency_step)
    except (TypeError, ValueError) as exc:
        raise TypeError("frequency_step must be a positive real number") from exc
    if not np.isfinite(frequency_step) or frequency_step <= 0:
        raise ValueError("frequency_step must be a positive finite number")

    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError("alpha must be a positive real number") from exc
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number")

    if output_length is not None:
        if isinstance(output_length, (bool, np.bool_)) or not isinstance(
            output_length, (int, np.integer)
        ):
            raise TypeError("output_length must be a positive integer or None")
        output_length = int(output_length)
        if output_length <= 0:
            raise ValueError("output_length must be a positive integer or None")

    return ts_arr, sample_rate, fmin, fmax, frequency_step, output_length


def _select_frequency_indices(
    n_samples: int,
    sample_rate: float,
    fmin: float,
    fmax: float,
    frequency_step: float,
) -> NDArray[np.int64]:
    """Select canonical FFT bins in the requested signed-frequency range."""
    frequency_resolution = sample_rate / n_samples
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
        raise ValueError("frequency_range does not contain an available FFT bin")
    step_bins = max(
        1,
        int(np.ceil(frequency_step / frequency_resolution - 1e-12)),
    )
    signed_frequency_indices = np.arange(
        start_bin,
        stop_bin + 1,
        step_bins,
        dtype=np.int64,
    )
    return np.mod(signed_frequency_indices, n_samples)


def _window_normal(
    length: int,
    freq: float,
    factor: float = 1.0,
) -> NDArray[np.float64]:
    """Return the periodized Gaussian frequency-domain window."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not np.isfinite(freq) or freq <= 0:
        raise ValueError("freq must be a positive finite FFT-bin frequency")

    bins = np.arange(length, dtype=float)
    signed_bins = np.where(bins <= length // 2, bins, bins - length)
    return factor * np.exp(-2.0 * np.pi**2 * (signed_bins / freq) ** 2)


def _frequency_chunk_size(
    n_samples: int,
    output_columns: int,
    n_frequency_rows: int,
) -> int:
    """Choose a batch size targeting a bounded FFT/resampling workspace.

    The per-row estimate includes shifted complex data, the real window, and
    temporary FFT/resampling storage. The one-row minimum may exceed the target.
    """
    estimated_bytes_per_row = 64 * (n_samples + output_columns)
    return max(
        1,
        min(
            n_frequency_rows,
            _SCRATCH_MEMORY_BUDGET_BYTES // estimated_bytes_per_row,
        ),
    )


def s_transform(
    ts: ArrayLike,
    sample_rate: float,
    frequency_range: Sequence[float] | None = None,
    frequency_step: float | None = None,
    alpha: float = 1.0,
    output_length: int | None = None,
    max_output_elements: int | None = 20_000_000,
    boundary: BoundaryMode = "periodic",
    boundary_padding: int | None = None,
) -> STransform:
    """Compute the discrete Stockwell transform.

    Parameters
    ----------
    ts : array_like, shape (n_samples,)
        Real or complex time-series samples.
    sample_rate : float
        Sample rate in hertz.
    frequency_range : sequence of float, optional
        Signed frequency range in hertz. The default is ``(0, Nyquist)``.
    frequency_step : float, optional
        Requested frequency spacing in hertz. The actual spacing is rounded
        up to an integer number of FFT bins. The default is one FFT bin for
        the unpadded input; zero-boundary padding retains that requested spacing.
    alpha : float, optional
        Positive Gaussian width multiplier. For a nonzero frequency ``f``,
        the corresponding temporal Gaussian has standard deviation
        ``alpha / abs(f)``. ``alpha=1`` is the standard Stockwell window;
        larger values broaden the window in time and narrow it in frequency,
        while smaller values do the reverse.
    output_length : int, optional
        Number of columns in the returned transform. The default is the input
        length. Values smaller than the input downsample the time axis; larger
        values upsample it. Periodic boundaries use Fourier resampling;
        zero-extended boundaries use polyphase resampling with zero extension.
    max_output_elements : int or None, optional
        Safety limit for the number of complex output elements. Set to
        ``None`` to opt into an unbounded allocation for large transforms.
    boundary : {"periodic", "zeros"}, optional
        Boundary condition for the finite record. ``"periodic"`` preserves
        the standard discrete S-transform and its wrap-around behavior.
        ``"zeros"`` pads the record with zeros, computes on that extended
        period, then returns only columns corresponding to the original input.
        Zero-boundary results are not invertible with
        :func:`inverse_s_transform`.
    boundary_padding : int, optional
        Symmetric zero-padding in samples when ``boundary="zeros"``. By
        default, padding is six temporal Gaussian standard deviations at the
        lowest nonzero frequency selected from the unpadded input grid. A
        manually supplied value is treated as a minimum and rounded up as
        needed to keep the padded FFT frequency bins aligned with the input
        frequency grid. Padding trades memory and computation for more or less
        separation from the periodic seam. It must be positive when the
        selected frequency rows include a nonzero frequency. This option is
        not allowed with ``boundary="periodic"``.

    Returns
    -------
    STransform
        Validated complex transform data and physical-axis metadata.
    """
    ts_arr, sample_rate, fmin, fmax, frequency_step, output_length = (
        _validate_input(
            ts,
            sample_rate,
            frequency_range,
            frequency_step,
            alpha,
            output_length,
        )
    )
    input_length = ts_arr.size
    boundary = _normalize_boundary(boundary)
    if boundary_padding is not None:
        boundary_padding = _as_integer_scalar(
            boundary_padding,
            "boundary_padding",
        )
        if boundary_padding < 0:
            raise ValueError("boundary_padding must be non-negative")

    original_grid_indices = _select_frequency_indices(
        input_length,
        sample_rate,
        fmin,
        fmax,
        frequency_step,
    )
    original_grid_signed_indices = np.where(
        original_grid_indices <= input_length // 2,
        original_grid_indices,
        original_grid_indices - input_length,
    )

    if boundary == "periodic":
        if boundary_padding not in (None, 0):
            raise ValueError(
                "boundary_padding can only be used with boundary='zeros'"
            )
        padding = 0
        transform_multiple = 1
        transform_length = input_length
    else:
        nonzero_original_bins = np.abs(
            original_grid_signed_indices[original_grid_signed_indices != 0]
        )
        if boundary_padding is None:
            if nonzero_original_bins.size:
                padding_ratio = (
                    _BOUNDARY_PADDING_SIGMAS
                    * alpha
                    / nonzero_original_bins.min()
                )
                if (
                    not np.isfinite(padding_ratio)
                    or padding_ratio > np.iinfo(np.int64).max / input_length
                ):
                    raise ValueError("automatic boundary padding is too large")
                requested_padding = int(np.ceil(padding_ratio * input_length))
            else:
                requested_padding = 0
        else:
            requested_padding = boundary_padding
        if nonzero_original_bins.size and requested_padding == 0:
            raise ValueError(
                "boundary_padding must be positive when nonzero frequencies "
                "are selected"
            )
        if requested_padding == 0:
            transform_multiple = 1
        else:
            transform_multiple = max(
                2,
                (2 * requested_padding + 2 * input_length - 1) // input_length,
            )
            # Keep the padded and original lengths at matching parity so
            # original FFT-bin frequencies map exactly onto padded FFT bins.
            if input_length % 2 and transform_multiple % 2 == 0:
                transform_multiple += 1
        transform_length = input_length * transform_multiple
        padding = (transform_multiple - 1) * input_length // 2

    if transform_length > np.iinfo(np.int64).max:
        raise ValueError("boundary padding produces an unsupported transform length")
    frequency_indices = np.mod(
        original_grid_signed_indices * transform_multiple,
        transform_length,
    )
    max_bin = transform_length // 2

    if max_output_elements is not None:
        max_output_elements = _as_integer_scalar(
            max_output_elements,
            "max_output_elements",
        )
        if max_output_elements <= 0:
            raise ValueError("max_output_elements must be positive or None")
    output_columns = input_length if output_length is None else output_length
    output_elements = frequency_indices.size * output_columns
    if (
        max_output_elements is not None
        and output_elements > max_output_elements
    ):
        raise MemoryError(
            f"requested transform has {output_elements:,} complex elements, "
            f"which exceeds max_output_elements={max_output_elements:,}; "
            "reduce the frequency range/output_length or pass "
            "max_output_elements=None"
        )

    working_signal = (
        ts_arr
        if boundary == "periodic"
        else np.pad(ts_arr, (padding, padding), mode="constant")
    )
    spectrum = fft.fft(working_signal)
    transform = np.empty(
        (frequency_indices.size, output_columns),
        dtype=np.complex128,
    )
    dc_rows = np.flatnonzero(frequency_indices == 0)
    if dc_rows.size:
        dc_value = (
            spectrum[0] / transform_length
            if boundary == "periodic"
            else np.mean(ts_arr)
        )
        if boundary == "periodic" and output_columns != input_length:
            dc_row = signal.resample(
                np.full(input_length, dc_value, dtype=np.complex128),
                output_columns,
            )
            transform[dc_rows] = dc_row
        else:
            transform[dc_rows] = dc_value

    nonzero_rows = np.flatnonzero(frequency_indices != 0)
    if nonzero_rows.size:
        chunk_size = _frequency_chunk_size(
            transform_length,
            output_columns,
            nonzero_rows.size,
        )
        sample_bins = np.arange(transform_length, dtype=np.int64)
        signed_sample_bins = np.where(
            sample_bins <= max_bin,
            sample_bins,
            sample_bins - transform_length,
        )

        for chunk_start in range(0, nonzero_rows.size, chunk_size):
            row_positions = nonzero_rows[chunk_start : chunk_start + chunk_size]
            chunk_frequency_indices = frequency_indices[row_positions]
            signed_frequency_bins = np.where(
                chunk_frequency_indices <= max_bin,
                chunk_frequency_indices,
                chunk_frequency_indices - transform_length,
            )

            shifted_spectra = np.empty(
                (row_positions.size, transform_length),
                dtype=np.complex128,
            )
            for chunk_row, frequency_index in enumerate(chunk_frequency_indices):
                shifted_spectra[chunk_row] = np.roll(spectrum, -int(frequency_index))

            windows = np.empty(shifted_spectra.shape, dtype=np.float64)
            frequency_scales = np.abs(signed_frequency_bins).astype(np.float64) / alpha
            np.divide(
                signed_sample_bins[np.newaxis, :],
                frequency_scales[:, np.newaxis],
                out=windows,
            )
            np.square(windows, out=windows)
            windows *= -2.0 * np.pi**2
            np.exp(windows, out=windows)
            shifted_spectra *= windows
            del windows

            rows = fft.ifft(shifted_spectra, axis=-1, overwrite_x=True)
            if boundary == "zeros":
                rows = rows[:, padding : padding + input_length]
                if output_columns != input_length:
                    divisor = int(np.gcd(input_length, output_columns))
                    rows = signal.resample_poly(
                        rows,
                        output_columns // divisor,
                        input_length // divisor,
                        axis=-1,
                        padtype="constant",
                    )
            elif output_columns != input_length:
                rows = signal.resample(rows, output_columns, axis=-1)
            transform[row_positions] = rows

    return STransform(
        data=transform,
        frequency_indices=frequency_indices,
        sample_rate=sample_rate,
        input_length=input_length,
        real_input=not np.iscomplexobj(ts),
        transform_length=transform_length,
        boundary=boundary,
        boundary_padding=padding,
        inverse_compatible=boundary == "periodic",
    )


def inverse_s_transform(
    table: ArrayLike | STransform,
    low_frequency_index: int | None = None,
    *,
    frequency_indices: ArrayLike | None = None,
    original_length: int | None = None,
    real_output: bool | None = None,
    boundary: BoundaryMode | None = None,
    inverse_compatible: bool | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Reconstruct a signal from an S-transform table.

    When ``table`` is an :class:`STransform`, its frequency-bin metadata is
    used automatically. For a plain ndarray, rows are assumed to represent
    contiguous FFT bins beginning at ``low_frequency_index`` (default zero).
    For real input, exact reconstruction requires all nonnegative-frequency
    bins through Nyquist and no time resampling. For complex input, request
    the complete canonical signed-frequency range instead. Partial frequency
    ranges and resampled tables produce the corresponding approximation.

    Parameters
    ----------
    table : array_like, shape (n_frequencies, n_time)
        S-transform table or an ``STransform`` wrapper.
    low_frequency_index : int, optional
        Starting signed/raw FFT bin for plain ndarray input. Negative values
        select negative-frequency bins.
    frequency_indices : array_like, optional
        FFT-bin index for each row when ``table`` is a plain ndarray. Raw
        modulo-``N`` indices and signed indices are both accepted.
    original_length : int, optional
        Length of the original signal. Required for a plain ndarray and
        inferred from ``STransform`` metadata otherwise.
    real_output : bool, optional
        Mirror positive-frequency coefficients when true. Inferred from the
        forward-transform metadata when available.
    boundary : {"periodic", "zeros"}, optional
        Boundary condition used to create a plain ndarray table. Only
        ``"periodic"`` tables are supported by this inverse implementation.
        Inferred from ``STransform`` metadata when available.
    inverse_compatible : bool, optional
        Whether the table uses periodic time-domain rows supported by this
        inverse implementation. Inferred from ``STransform`` metadata when
        available. Zero-boundary transforms cropped to the input interval are
        not compatible with this inverse.
    """
    if low_frequency_index is not None and frequency_indices is not None:
        raise ValueError(
            "provide either low_frequency_index or frequency_indices, not both"
        )

    metadata = table if isinstance(table, STransform) else None
    if metadata is not None and low_frequency_index is not None:
        raise ValueError(
            "low_frequency_index cannot be supplied with an STransform"
        )

    if metadata is not None:
        if boundary is not None and _normalize_boundary(boundary) != metadata.boundary:
            raise ValueError("boundary does not match the transform metadata")
        if inverse_compatible is not None and _as_bool_scalar(
            inverse_compatible,
            "inverse_compatible",
        ) != metadata.inverse_compatible:
            raise ValueError("inverse_compatible does not match the transform metadata")
        if not metadata.inverse_compatible:
            raise ValueError(
                "this boundary-extended transform is cropped to the input interval "
                "and cannot be inverted by inverse_s_transform"
            )
    else:
        boundary_value = (
            "periodic" if boundary is None else _normalize_boundary(boundary)
        )
        inverse_compatible_value = (
            True
            if inverse_compatible is None
            else _as_bool_scalar(inverse_compatible, "inverse_compatible")
        )
        if boundary_value != "periodic" or not inverse_compatible_value:
            raise ValueError(
                "inverse_s_transform only supports inverse-compatible periodic tables"
            )

    table_arr = np.asarray(
        metadata.data if metadata is not None else table,
        dtype=np.complex128,
    )
    if table_arr.ndim != 2:
        raise ValueError("table must be a two-dimensional array")
    if 0 in table_arr.shape:
        raise ValueError("table must have non-empty frequency and time axes")
    n_frequencies, n_time = table_arr.shape

    if metadata is not None:
        inferred_length = metadata._fft_length
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
        metadata_indices = metadata.frequency_indices
        if frequency_indices is not None and not np.array_equal(
            _normalize_frequency_indices(
                frequency_indices,
                original_length,
                "frequency_indices",
            ),
            metadata_indices,
        ):
            raise ValueError("frequency_indices do not match the transform metadata")
        frequency_indices_arr = metadata_indices.copy()
        inferred_real = metadata.real_input
    else:
        provided_indices: ArrayLike
        if frequency_indices is None:
            start = (
                0
                if low_frequency_index is None
                else _as_integer_scalar(
                    low_frequency_index,
                    "low_frequency_index",
                )
            )
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

    if frequency_indices_arr.size != n_frequencies:
        raise ValueError("frequency_indices must contain one FFT bin per table row")
    if np.unique(frequency_indices_arr).size != frequency_indices_arr.size:
        raise ValueError("frequency_indices must not contain duplicates")

    if real_output is None:
        real_output = inferred_real
    else:
        real_output = _as_bool_scalar(real_output, "real_output")

    # A resampled table no longer has the original row length. Resampling rows
    # back before taking their zero-frequency FFT coefficient preserves the
    # transform coefficient as closely as the time-axis resampling allows.
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
