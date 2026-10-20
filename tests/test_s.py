import numpy as np
import pytest

from s import STransformArray, inverse_s_transform, s_transform


def test_default_transform_contains_dc_through_nyquist():
    x = np.ones(16)
    transform = s_transform(x, sample_rate=16)

    assert isinstance(transform, STransformArray)
    np.testing.assert_array_equal(transform.frequency_indices, np.arange(9))
    np.testing.assert_allclose(transform.frequencies, np.arange(9))
    np.testing.assert_allclose(transform[0], 1.0)


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


def test_frequency_step_preserves_actual_fft_bin_indices():
    x = np.sin(2 * np.pi * 4 * np.arange(32) / 32)
    transform = s_transform(
        x,
        sample_rate=32,
        freq_range=(0, 16),
        freq_step=4,
    )

    np.testing.assert_array_equal(transform.frequency_indices, np.arange(0, 17, 4))
    np.testing.assert_allclose(transform.frequencies, np.arange(0, 17, 4))
    np.testing.assert_allclose(np.fft.fft(transform[1])[0], -16j, atol=1e-10)


def test_odd_downsample_is_supported_and_metadata_survives_inverse():
    x = np.sin(2 * np.pi * np.arange(32) / 32)
    transform = s_transform(x, sample_rate=32, downsample=15)

    assert transform.shape == (17, 15)
    assert transform.input_length == 32
    recovered, _ = inverse_s_transform(transform)
    assert recovered.shape == x.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": 16, "freq_step": 0},
        {"sample_rate": 16, "freq_range": (-1, 8)},
        {"sample_rate": 16, "freq_range": (0, 9)},
        {"sample_rate": 16, "downsample": 0},
    ],
)
def test_invalid_parameters_raise_value_error(kwargs):
    with pytest.raises((TypeError, ValueError)):
        s_transform(np.ones(16), **kwargs)


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


def test_inverse_rejects_conflicting_original_length():
    transform = s_transform(np.ones(16), sample_rate=16)

    with pytest.raises(ValueError, match="metadata"):
        inverse_s_transform(transform, original_length=32)


def test_derived_transform_requires_explicit_metadata():
    transform = s_transform(np.ones(16), sample_rate=16)
    derived = transform[:, :8]

    with pytest.raises(ValueError, match="metadata"):
        inverse_s_transform(derived)

    recovered, _ = inverse_s_transform(
        derived,
        frequency_indices=transform.frequency_indices,
        original_length=16,
        real_output=True,
    )
    assert recovered.shape == (16,)


def test_transform_output_limit_is_enforced():
    with pytest.raises(MemoryError, match="max_output_elements"):
        s_transform(np.ones(16), sample_rate=16, max_output_elements=8)
