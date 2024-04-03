Supported in: [![PyPI version](https://badge.fury.io/py/TFchirp.svg)](https://badge.fury.io/py/TFchirp)

[Reference] R. G. Stockwell, L. Mansinha, and R. P. Lowe, Localization of the Complex Spectrum: The S Transform. IEEE 
Transactions on Signal Processing, Vol. 44, No. 4, Apr. 1996

## S (Stockwell) Transform for Chirp Signals

Step 1: Quadratic chirp signal

Generate a quadratic chirp signal from 10 Hz to 120 Hz in 1 second with 10,000 sampling points.

```Python
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Generate a quadratic chirp signal
dt = 0.001; t = 3
rate = int(1/dt)
ts = np.linspace(0, t, int(1/dt))
data = scipy.signal.chirp(ts, 10, 1, 120, method='quadratic')
```

Step 2: S Transform Spectrogram

```Python
from s import *

# Compute S Transform Spectrogram
spectrogram = sTransform(data, sample_rate=rate, frate=rate/len(data),
                                    downsample=None, frange=[0,500])
plt.imshow(abs(spectrogram), origin='lower', aspect='auto')
plt.title('Original Spectrogram')
plt.colorbar()
plt.show()
```

![Original Spectrogram](https://github.com/xli2522/S-Transform/blob/main/img/original_spectrogram.png?raw=true)

Step 3: The inverse S transform

```python
# Quick Inverse of ts from S Transform
inverse_ts, inverse_tsFFT = inverseS(spectrogram)

# Magnitude Compensation: 
# with the assumption that ts is real and only positive freqs are kept
inverse_ts_comp = inverse_ts*2
# Plot the original signal and the recovered, magnitude compensated signal
fig, axs = plt.subplots(2,1)
axs[0].plot(data)
axs[1].plot(inverse_ts_comp.real)
axs[0].set_title('Original Signal')
axs[1].set_title('(inverseS) Freq-passed, down-sampled Signal')
plt.show()
plt.plot(inverse_ts_comp)
plt.plot(inverse_ts_comp-data)
plt.title('Time Series Reconstruction Error')
plt.legend(['Recovered ts', 'Error'])
plt.show()
```

![Recovered ts and Error](https://github.com/xli2522/S-Transform/blob/main/img/recovered_ts_error.png?raw=true)

Step 4: Recovered inverse S transform spectrogram

```python
# Compute S Transform Spectrogram on the recovered time series
inverseSpectrogram = sTransform(inverse_ts, 
                        sample_rate=len(inverse_ts)/len(data)*rate, 
                                                    frange=[0,500])
plt.imshow(abs(inverseSpectrogram), origin='lower', aspect='auto')
plt.title('Recovered Spectrogram (inverseS)')
plt.colorbar()
plt.show()
```

![Recovered Spectrogram](https://github.com/xli2522/S-Transform/blob/main/img/recovered_spectrogram.png?raw=true)

![Downsampled Timeseries](https://github.com/xli2522/S-Transform/blob/main/img/original_downsampled_signals.png?raw=true)