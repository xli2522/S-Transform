Supported in: [![PyPI version](https://badge.fury.io/py/TFchirp.svg)](https://badge.fury.io/py/TFchirp)

[References] 
- R. G. Stockwell, L. Mansinha, and R. P. Lowe, Localization of the Complex Spectrum: The S Transform. IEEE 
Transactions on Signal Processing, Vol. 44, No. 4, Apr. 1996
- Stockwell, RG (1999). S-transform analysis of gravity wave activity from a small scale network of airglow imagers. 
PhD thesis, University of Western Ontario, London, Ontario, Canada.

## S (Stockwell) Transform for Chirp Signals

Step 1: Quadratic chirp signal

Generate a quadratic chirp signal from 10 Hz to 120 Hz in 1 second with 10,000 sampling points.

```Python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

# Time vector and sampling rate
dt = 0.001
t_end = 3.0
fs = int(1.0 / dt)
t = np.linspace(0, t_end, int(t_end * fs), endpoint=False)

# Generate quadratic chirp signal
x = chirp(t, f0=10.0, t1=t_end, f1=120.0, method='quadratic')
```

Step 2: S Transform Spectrogram

```Python
from s import s_transform, inverse_s_transform

# Compute S-Transform spectrogram
st = s_transform(
    x,
    sample_rate=fs,
    freq_range=(0.0, 500.0),
    freq_step=fs / x.size,
    downsample=None
)
plt.figure()
plt.imshow(np.abs(st), origin='lower', aspect='auto')
plt.title('Original Spectrogram')
plt.colorbar()
plt.show()
```

![Original Spectrogram](https://github.com/xli2522/S-Transform/blob/main/img/original_spectrogram.png?raw=true)

Step 3: The inverse S transform

```python
# Inverse S-Transform
x_rec, fft_rec = inverse_s_transform(st)

# Magnitude compensation (real signal, one-sided FFT)
x_rec_comp = x_rec * 2.0

# Plot original vs. recovered
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes[0].plot(x)
axes[0].set_title('Original Signal')
axes[1].plot(x_rec_comp.real)
axes[1].set_title('Recovered Signal (magnitude-compensated)')
plt.tight_layout()
plt.show()

# Reconstruction error
plt.figure()
plt.plot(x_rec_comp.real, label='Recovered Signal')
plt.plot(x - x_rec_comp.real, label='Error')
plt.title('Time Series Reconstruction Error')
plt.legend()
plt.show()
```

![Recovered ts and Error](https://github.com/xli2522/S-Transform/blob/main/img/recovered_ts_error.png?raw=true)

Step 4: Recovered inverse S transform spectrogram

```python
# Spectrogram of recovered signal
st_rec = s_transform(
    x_rec.real,
    sample_rate=fs,
    freq_range=(0.0, 500.0),
    freq_step=fs / x_rec.size
)
plt.figure()
plt.imshow(np.abs(st_rec), origin='lower', aspect='auto')
plt.title('Recovered Spectrogram')
plt.colorbar()
plt.show()
```

![Recovered Spectrogram](https://github.com/xli2522/S-Transform/blob/main/img/recovered_spectrogram.png?raw=true)

## One of the original copies of Dr. Stockwell's PhD thesis. 
![Thesis](https://github.com/xli2522/S-Transform/blob/main/img/stockwell_thesis_small.jpg?raw=true)