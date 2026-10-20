# S-Transform for Python

This repository provides a NumPy/SciPy implementation of the discrete
Stockwell (S) transform for one-dimensional signals.

## Quick start

Install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

The transform returns an ndarray-compatible object with the frequency-bin and
time-axis metadata needed to interpret or invert it:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

from s import inverse_s_transform, s_transform

sample_rate = 1_000.0
duration = 3.0
time = np.arange(0.0, duration, 1.0 / sample_rate)
x = chirp(time, f0=10.0, t1=duration, f1=120.0, method="quadratic")

st = s_transform(x, sample_rate=sample_rate)

plt.imshow(
    20 * np.log10(np.maximum(np.abs(st), 1e-12)),
    origin="lower",
    aspect="auto",
    extent=(st.times[0], st.times[-1], st.frequencies[0], st.frequencies[-1]),
)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("S-transform magnitude (dB)")
plt.colorbar(label="Magnitude (dB)")
plt.show()
```

The default transform uses every FFT bin from zero through Nyquist. The
actual frequency represented by each row is available as `st.frequencies`.
If `freq_step` is supplied, it is rounded up to an integer number of FFT bins;
the selected bins are available as `st.frequency_indices`.

## Inversion

For a real-valued signal, the full positive-frequency transform can be
reconstructed without a manual magnitude correction:

```python
x_recovered, spectrum_recovered = inverse_s_transform(st)
np.testing.assert_allclose(x_recovered.real, x, atol=1e-10)
```

Exact reconstruction requires the full positive-frequency range, one row per
FFT bin, and no time-axis downsampling. A restricted frequency range or a
downsampled transform intentionally produces a band-limited approximation.

For a plain ndarray copied from a transform, pass the metadata explicitly:

```python
x_recovered, spectrum_recovered = inverse_s_transform(
    np.asarray(st),
    frequency_indices=st.frequency_indices,
    original_length=st.input_length,
    real_output=st.real_input,
)
```

Any sliced, transposed, or otherwise derived `STransformArray` loses its
automatic metadata because its rows and columns may no longer correspond to
the original axes. Pass `frequency_indices` and `original_length` explicitly
when inverting such an array. Plain ndarrays always require
`original_length`.

## Downsampling

`downsample=M` resamples the transform's time axis to `M` columns after the
transform is computed. It accepts both even and odd positive lengths. Reducing
the number of columns is not lossless. To prevent accidental multi-gigabyte
allocations, `s_transform` limits the output to 20 million complex elements by
default; pass `max_output_elements=None` only when a larger transform is
intentional.

## References

- R. G. Stockwell, L. Mansinha, and R. P. Lowe, “Localization of the Complex
  Spectrum: The S Transform,” *IEEE Transactions on Signal Processing*, 1996.
- R. G. Stockwell, *S-transform analysis of gravity wave activity from a small
  scale network of airglow imagers*, PhD thesis, University of Western Ontario,
  1999.

![Original spectrogram](img/original_spectrogram.png)

![Recovered signal and error](img/recovered_ts_error.png)
