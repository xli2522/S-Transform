import numpy as np
from scipy.signal import chirp

from s_transform import inverse_s_transform, s_transform


def test_readme_core_example_does_not_require_matplotlib():
    sample_rate = 1_000.0
    duration = 0.25
    time = np.arange(0.0, duration, 1.0 / sample_rate)
    signal = chirp(time, f0=10.0, t1=duration, f1=120.0, method="quadratic")

    transform = s_transform(signal, sample_rate=sample_rate)
    recovered, _ = inverse_s_transform(transform)

    assert transform.data.shape == (126, 250)
    np.testing.assert_allclose(recovered.real, signal, atol=1e-10)
    np.testing.assert_allclose(recovered.imag, 0.0, atol=1e-10)
