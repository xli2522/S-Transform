import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import fft, signal

from s_transform import STransform, inverse_s_transform, s_transform
from s_transform import core as transform_core


def _rowwise_reference(ts, transform, alpha=1.0):
    """Reproduce the pre-batching frequency-row implementation for tests."""
    samples = np.asarray(ts, dtype=np.complex128)
    n_samples = samples.size
    max_bin = n_samples // 2
    signed_bins = np.arange(n_samples, dtype=np.int64)
    signed_bins = np.where(
        signed_bins <= max_bin,
        signed_bins,
        signed_bins - n_samples,
    )
    spectrum = fft.fft(samples)
    output = np.empty_like(transform.data)

    for row_index, frequency_index in enumerate(transform.frequency_indices):
        if frequency_index == 0:
            row = np.full(n_samples, spectrum[0] / n_samples, dtype=np.complex128)
        else:
            signed_frequency_index = (
                frequency_index
                if frequency_index <= max_bin
                else frequency_index - n_samples
            )
            shifted = spectrum[
                (frequency_index + np.arange(n_samples, dtype=np.int64))
                % n_samples
            ]
            window = np.exp(
                -2.0
                * np.pi**2
                * (
                    signed_bins
                    / (abs(signed_frequency_index) / alpha)
                )
                ** 2
            )
            row = fft.ifft(shifted * window)
        if output.shape[1] != n_samples:
            row = signal.resample(row, output.shape[1])
        output[row_index] = row
    return output


def test_default_transform_contains_dc_through_nyquist():
    x = np.ones(16)
    transform = s_transform(x, sample_rate=16)

    assert isinstance(transform, STransform)
    assert transform.data.shape == (9, 16)
    np.testing.assert_array_equal(transform.frequency_indices, np.arange(9))
    np.testing.assert_allclose(transform.frequencies, np.arange(9))
    np.testing.assert_allclose(transform.data[0], 1.0)


@pytest.mark.parametrize(
    (
        "n_samples",
        "is_complex",
        "frequency_range",
        "frequency_step",
        "alpha",
        "output_length",
    ),
    [
        (8, False, None, None, 1.0, None),
        (9, False, None, 2.0, 0.7, 5),
        (8, True, (-4.0, 4.0), None, 1.2, 12),
        (9, True, (-3.1, 2.9), 1.5, 1.4, None),
    ],
)
def test_batched_transform_matches_rowwise_reference(
    n_samples,
    is_complex,
    frequency_range,
    frequency_step,
    alpha,
    output_length,
):
    rng = np.random.default_rng(n_samples)
    samples = rng.standard_normal(n_samples)
    if is_complex:
        samples = samples + 1j * rng.standard_normal(n_samples)

    transform = s_transform(
        samples,
        sample_rate=n_samples,
        frequency_range=frequency_range,
        frequency_step=frequency_step,
        alpha=alpha,
        output_length=output_length,
    )

    np.testing.assert_allclose(
        transform.data,
        _rowwise_reference(samples, transform, alpha),
        rtol=1e-12,
        atol=1e-10,
    )


@pytest.mark.parametrize("rows_per_chunk", [1, 3])
def test_batched_transform_handles_chunk_boundaries(monkeypatch, rows_per_chunk):
    n_samples = 9
    nonzero_rows = n_samples - 1
    bytes_per_row = 64 * (n_samples + n_samples)
    monkeypatch.setattr(
        transform_core,
        "_SCRATCH_MEMORY_BUDGET_BYTES",
        rows_per_chunk * bytes_per_row,
    )
    chunk_size = transform_core._frequency_chunk_size(
        n_samples,
        n_samples,
        nonzero_rows,
    )
    assert chunk_size == rows_per_chunk
    assert (nonzero_rows + chunk_size - 1) // chunk_size >= 2
    assert chunk_size * bytes_per_row <= (
        transform_core._SCRATCH_MEMORY_BUDGET_BYTES
    )
    if rows_per_chunk > 1:
        assert nonzero_rows % chunk_size != 0

    observed_ifft_chunk_sizes = []
    original_ifft = transform_core.fft.ifft

    def record_ifft_chunk_size(values, *args, **kwargs):
        observed_ifft_chunk_sizes.append(values.shape[0])
        return original_ifft(values, *args, **kwargs)

    monkeypatch.setattr(transform_core.fft, "ifft", record_ifft_chunk_size)

    rng = np.random.default_rng(109)
    samples = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    transform = s_transform(
        samples,
        sample_rate=n_samples,
        frequency_range=(-n_samples / 2, n_samples / 2),
    )

    expected_chunk_sizes = [
        min(rows_per_chunk, nonzero_rows - start)
        for start in range(0, nonzero_rows, rows_per_chunk)
    ]
    assert observed_ifft_chunk_sizes == expected_chunk_sizes

    np.testing.assert_allclose(
        transform.data,
        _rowwise_reference(samples, transform),
        rtol=1e-12,
        atol=1e-10,
    )


def test_frequency_chunk_sizing_keeps_one_row_below_budget(monkeypatch):
    monkeypatch.setattr(transform_core, "_SCRATCH_MEMORY_BUDGET_BYTES", 1)

    assert transform_core._frequency_chunk_size(1024, 2048, 10) == 1


def test_full_real_transform_round_trips_without_manual_factor():
    n_samples = 64
    sample_rate = 64.0
    time = np.arange(n_samples) / sample_rate
    x = 0.25 + np.sin(2 * np.pi * 5 * time) + 0.5 * np.cos(2 * np.pi * 11 * time)

    transform = s_transform(x, sample_rate=sample_rate)
    recovered, recovered_spectrum = inverse_s_transform(transform)

    np.testing.assert_allclose(recovered.real, x, atol=1e-10)
    np.testing.assert_allclose(recovered.imag, 0.0, atol=1e-10)
    np.testing.assert_allclose(recovered_spectrum[5], -32j, atol=1e-10)


@pytest.mark.parametrize("n_samples", [3, 4, 5, 8, 9, 16])
def test_real_round_trip_for_odd_and_even_lengths(n_samples):
    sample_indices = np.arange(n_samples)
    x = (
        0.1
        + np.cos(2 * np.pi * sample_indices / n_samples)
        + 0.2 * np.sin(2 * np.pi * sample_indices / n_samples)
    )

    transform = s_transform(x, sample_rate=n_samples)
    recovered, _ = inverse_s_transform(transform)

    np.testing.assert_allclose(recovered.real, x, atol=1e-10)
    np.testing.assert_allclose(recovered.imag, 0.0, atol=1e-10)


def test_frequency_step_preserves_actual_fft_bin_indices():
    x = np.sin(2 * np.pi * 4 * np.arange(32) / 32)
    transform = s_transform(
        x,
        sample_rate=32,
        frequency_range=(0, 16),
        frequency_step=4,
    )

    np.testing.assert_array_equal(transform.frequency_indices, np.arange(0, 17, 4))
    np.testing.assert_allclose(transform.frequencies, np.arange(0, 17, 4))
    np.testing.assert_allclose(np.fft.fft(transform.data[1])[0], -16j, atol=1e-10)


def test_transform_exposes_physical_axis_metadata():
    transform = s_transform(
        np.ones(16),
        sample_rate=32,
        frequency_range=(-16, 16),
        output_length=5,
    )

    assert transform.duration == 0.5
    assert transform.time_step == 0.1
    assert transform.frequency_resolution == 2.0
    assert transform.frequency_range == (-14.0, 16.0)
    assert transform.time_range == (0.0, 0.4)
    assert transform.is_downsampled

    upsampled = s_transform(
        np.ones(16),
        sample_rate=32,
        output_length=24,
    )
    assert not upsampled.is_downsampled
    assert upsampled.is_upsampled
    assert not transform.is_upsampled

    np.testing.assert_array_equal(
        transform.inverse_kwargs["frequency_indices"],
        transform.frequency_indices,
    )
    assert transform.inverse_kwargs["original_length"] == 16
    assert transform.inverse_kwargs["real_output"] is True


def test_frequency_range_is_order_independent():
    transform = STransform(
        np.zeros((3, 8), dtype=complex),
        frequency_indices=[3, 1, 2],
        sample_rate=8,
        input_length=8,
        real_input=False,
    )

    assert transform.frequency_range == (1.0, 3.0)


@pytest.mark.parametrize("shape", [(0, 8), (3, 0)])
def test_transform_rejects_empty_axes(shape):
    with pytest.raises(ValueError, match="non-empty"):
        STransform(
            np.zeros(shape, dtype=complex),
            frequency_indices=np.arange(shape[0]),
            sample_rate=8,
            input_length=8,
            real_input=False,
        )


@pytest.mark.parametrize("n_samples", [3, 4, 5, 8, 9, 16])
def test_complete_signed_complex_round_trip_for_odd_and_even_lengths(n_samples):
    sample_indices = np.arange(n_samples)
    x = np.exp(-2j * np.pi * (n_samples // 3) * sample_indices / n_samples)

    transform = s_transform(
        x,
        sample_rate=n_samples,
        frequency_range=(-n_samples / 2, n_samples / 2),
    )
    recovered, recovered_spectrum = inverse_s_transform(transform)

    expected_signed_indices = np.arange(
        -((n_samples - 1) // 2),
        n_samples // 2 + 1,
    )
    np.testing.assert_array_equal(
        transform.signed_frequency_indices,
        expected_signed_indices,
    )
    np.testing.assert_allclose(recovered, x, atol=1e-10)
    np.testing.assert_allclose(
        recovered_spectrum[np.argmax(np.abs(recovered_spectrum))],
        n_samples,
        atol=1e-10,
    )


def test_inverse_accepts_signed_indices_for_plain_arrays():
    n_samples = 16
    sample_indices = np.arange(n_samples)
    x = np.exp(2j * np.pi * 3 * sample_indices / n_samples)
    transform = s_transform(
        x,
        sample_rate=n_samples,
        frequency_range=(-n_samples / 2, n_samples / 2),
    )

    recovered, _ = inverse_s_transform(
        np.asarray(transform),
        frequency_indices=transform.signed_frequency_indices,
        original_length=n_samples,
    )

    np.testing.assert_allclose(recovered, x, atol=1e-10)


def test_low_frequency_index_selects_contiguous_plain_array_bins():
    n_samples = 16
    sample_indices = np.arange(n_samples)
    x = np.exp(2j * np.pi * 3 * sample_indices / n_samples)
    transform = s_transform(
        x,
        sample_rate=n_samples,
        frequency_range=(3, 5),
    )

    recovered, _ = inverse_s_transform(
        np.asarray(transform),
        low_frequency_index=3,
        original_length=n_samples,
    )

    np.testing.assert_allclose(recovered, x, atol=1e-10)


def test_odd_output_length_is_supported_and_metadata_survives_inverse():
    x = np.sin(2 * np.pi * np.arange(32) / 32)
    transform = s_transform(x, sample_rate=32, output_length=15)

    assert transform.data.shape == (17, 15)
    assert transform.input_length == 32
    recovered, _ = inverse_s_transform(transform)
    assert recovered.shape == x.shape


@pytest.mark.parametrize("output_length", [15, 24])
def test_output_length_supports_downsampling_and_upsampling(output_length):
    sample_indices = np.arange(32)
    x = np.sin(2 * np.pi * 3 * sample_indices / 32)
    transform = s_transform(x, sample_rate=32, output_length=output_length)

    recovered, _ = inverse_s_transform(transform)

    assert transform.data.shape[1] == output_length
    np.testing.assert_allclose(recovered, x, atol=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": 16, "frequency_step": 0},
        {"sample_rate": 16, "frequency_range": (-9, 8)},
        {"sample_rate": 16, "frequency_range": (0, 9)},
        {"sample_rate": 16, "output_length": 0},
    ],
)
def test_invalid_parameters_raise_value_error(kwargs):
    with pytest.raises((TypeError, ValueError)):
        s_transform(np.ones(16), **kwargs)


def test_scalar_frequency_range_has_a_friendly_error():
    with pytest.raises(TypeError, match="two-item sequence"):
        s_transform(np.ones(16), sample_rate=16, frequency_range=4)


def test_multidimensional_input_is_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        s_transform(np.ones((2, 8)), sample_rate=16)


def test_plain_ndarray_can_be_inverted_with_explicit_metadata():
    x = np.sin(2 * np.pi * np.arange(16) / 16)
    transform = s_transform(x, sample_rate=16)
    plain_table = np.asarray(transform)

    recovered, _ = inverse_s_transform(
        plain_table,
        frequency_indices=transform.frequency_indices,
        original_length=16,
        real_output=True,
    )

    np.testing.assert_allclose(recovered, x, atol=1e-10)
    np.testing.assert_allclose(recovered.imag, 0.0, atol=1e-10)


def test_plain_ndarray_defaults_to_contiguous_bins_from_zero():
    x = np.sin(2 * np.pi * np.arange(16) / 16)
    transform = s_transform(x, sample_rate=16)

    recovered, _ = inverse_s_transform(
        np.asarray(transform),
        original_length=16,
        real_output=True,
    )

    np.testing.assert_allclose(recovered, x, atol=1e-10)


def test_plain_ndarray_requires_original_length():
    with pytest.raises(ValueError, match="original_length"):
        inverse_s_transform(np.zeros((2, 8), dtype=complex))


def test_inverse_rejects_fractional_or_duplicate_frequency_bins():
    table = np.zeros((2, 8), dtype=complex)

    with pytest.raises(ValueError, match="integer"):
        inverse_s_transform(
            table,
            frequency_indices=[0.5, 1],
            original_length=8,
        )
    with pytest.raises(ValueError, match="duplicates"):
        inverse_s_transform(
            table,
            frequency_indices=[1, 1],
            original_length=8,
        )
    with pytest.raises(ValueError, match="duplicates"):
        inverse_s_transform(
            table,
            frequency_indices=[-1, 7],
            original_length=8,
        )


def test_inverse_rejects_conflicting_metadata():
    transform = s_transform(np.ones(16), sample_rate=16)

    with pytest.raises(ValueError, match="metadata"):
        inverse_s_transform(transform, original_length=32)
    with pytest.raises(ValueError, match="metadata"):
        inverse_s_transform(
            transform,
            frequency_indices=np.arange(1, transform.data.shape[0] + 1),
        )
    with pytest.raises(ValueError, match="either"):
        inverse_s_transform(
            np.asarray(transform),
            low_frequency_index=0,
            frequency_indices=transform.frequency_indices,
            original_length=16,
        )


def test_derived_data_requires_explicit_metadata():
    transform = s_transform(np.ones(16), sample_rate=16)
    derived = transform.data[:, :8]

    with pytest.raises(ValueError, match="original_length"):
        inverse_s_transform(derived)

    recovered, _ = inverse_s_transform(
        derived,
        frequency_indices=transform.frequency_indices,
        original_length=16,
        real_output=True,
    )
    assert recovered.shape == (16,)


def test_transform_metadata_round_trips_through_pickle_and_copy():
    transform = s_transform(
        np.exp(-2j * np.pi * 3 * np.arange(16) / 16),
        sample_rate=16,
        frequency_range=(-8, 8),
        output_length=24,
    )

    restored = pickle.loads(pickle.dumps(transform))
    shallow_copy = copy.copy(transform)
    deep_copy = copy.deepcopy(transform)

    assert restored == transform
    assert shallow_copy == transform
    assert deep_copy == transform
    np.testing.assert_array_equal(restored.data, transform.data)
    np.testing.assert_array_equal(
        restored.frequency_indices,
        transform.frequency_indices,
    )
    assert restored.sample_rate == transform.sample_rate
    assert restored.input_length == transform.input_length
    assert restored.real_input == transform.real_input
    np.testing.assert_array_equal(shallow_copy.data, transform.data)
    np.testing.assert_array_equal(
        shallow_copy.frequency_indices,
        transform.frequency_indices,
    )
    np.testing.assert_array_equal(deep_copy.data, transform.data)
    np.testing.assert_array_equal(
        deep_copy.frequency_indices,
        transform.frequency_indices,
    )
    np.testing.assert_allclose(
        inverse_s_transform(restored)[0],
        inverse_s_transform(transform)[0],
    )
    np.testing.assert_allclose(
        inverse_s_transform(shallow_copy)[0],
        inverse_s_transform(transform)[0],
    )
    np.testing.assert_allclose(
        inverse_s_transform(deep_copy)[0],
        inverse_s_transform(transform)[0],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency_indices": [0.5], "sample_rate": 8},
        {"frequency_indices": [9], "sample_rate": 8},
        {"frequency_indices": [1, 1], "sample_rate": 8},
        {"sample_rate": 0},
        {"input_length": 0},
        {"real_input": 1},
    ],
)
def test_transform_rejects_invalid_metadata(kwargs):
    metadata = {
        "frequency_indices": [0],
        "sample_rate": 8,
        "input_length": 8,
        "real_input": False,
    }
    metadata.update(kwargs)
    row_count = len(metadata["frequency_indices"])
    with pytest.raises((TypeError, ValueError)):
        STransform(
            np.zeros((row_count, 8), dtype=complex),
            **metadata,
        )


def test_transform_rejects_non_unique_frequency_bins_after_signed_normalization():
    with pytest.raises(ValueError, match="duplicates"):
        STransform(
            np.zeros((2, 8), dtype=complex),
            frequency_indices=[-1, 7],
            sample_rate=8,
            input_length=8,
            real_input=False,
        )


def test_inverse_rejects_empty_table():
    with pytest.raises(ValueError, match="non-empty"):
        inverse_s_transform(np.zeros((0, 8), dtype=complex), original_length=8)
    with pytest.raises(ValueError, match="non-empty"):
        inverse_s_transform(np.zeros((2, 0), dtype=complex), original_length=8)


def test_transform_array_conversion_is_explicit_and_copyable():
    transform = s_transform(np.ones(8), sample_rate=8)

    np.testing.assert_array_equal(np.asarray(transform), transform.data)
    copied = np.asarray(transform).copy()
    copied[0, 0] = 99
    assert transform.data[0, 0] != 99
    with pytest.raises(FrozenInstanceError):
        transform.sample_rate = 16


def test_transform_output_limit_is_enforced():
    with pytest.raises(MemoryError, match="max_output_elements"):
        s_transform(np.ones(16), sample_rate=16, max_output_elements=8)
