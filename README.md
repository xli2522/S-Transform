# S-Transform for Python
Supported in: [![PyPI version](https://badge.fury.io/py/TFchirp.svg)](https://badge.fury.io/py/TFchirp)

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

plt.pcolormesh(
    st.times,
    st.frequencies,
    np.abs(st),
    shading="auto",
)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("S-transform magnitude")
plt.colorbar(label="Magnitude")
plt.show()
```

The default transform uses every FFT bin from zero through Nyquist. The
actual frequency represented by each row is available as `st.frequencies`.
If `freq_step` is supplied, it is rounded up to an integer number of FFT bins;
the selected modulo-`N` FFT bins are available as `st.frequency_indices`, while
`st.signed_frequency_indices` exposes their signed mathematical indices when a
signed range is requested.

For plotting and inspection, `st.times`, `st.frequencies`, `st.duration`,
`st.time_step`, `st.frequency_resolution`, `st.frequency_range`, and
`st.time_range` expose the physical coordinates without reconstructing them
from array shapes. `st.is_downsampled` indicates whether the time axis has
fewer columns than the input.

For complex signals with negative-frequency content, request the signed range
explicitly. The canonical FFT ordering uses the positive Nyquist bin for even
lengths and represents all other negative bins with signed frequencies:

```python
complex_signal = x + 0.25j * np.sin(2 * np.pi * 80.0 * time)
complex_st = s_transform(
    complex_signal,
    sample_rate=sample_rate,
    freq_range=(-sample_rate / 2, sample_rate / 2),
)
complex_recovered, _ = inverse_s_transform(complex_st)
np.testing.assert_allclose(complex_recovered, complex_signal, atol=1e-10)
```

## Inversion

For a real-valued signal, the full nonnegative-frequency transform can be
reconstructed without a manual magnitude correction:

```python
x_recovered, spectrum_recovered = inverse_s_transform(st)
np.testing.assert_allclose(x_recovered.real, x, atol=1e-10)
```

For real signals, exact reconstruction requires the full nonnegative-frequency
range, one row per FFT bin, and no time-axis downsampling. For complex signals,
request the complete signed-frequency range shown above. A restricted
frequency range or a downsampled transform intentionally produces a
band-limited approximation.

For a plain ndarray copied from a transform, pass the metadata explicitly:

```python
x_recovered, spectrum_recovered = inverse_s_transform(
    np.asarray(st),
    **st.inverse_kwargs,
)
```

Any sliced, transposed, or otherwise derived `STransformArray` loses its
automatic metadata because its rows and columns may no longer correspond to
the original axes. Pass `frequency_indices` and `original_length` explicitly
when inverting such an array. Plain ndarrays always require
`original_length`; inverse metadata accepts either raw modulo-`N` bins or
negative signed bins and rejects aliases that refer to the same FFT bin.

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

## Reference material

## One of the original copies of Dr. Stockwell's PhD thesis.
![Thesis](https://github.com/xli2522/S-Transform/blob/main/img/stockwell_thesis_small.jpg?raw=true)
