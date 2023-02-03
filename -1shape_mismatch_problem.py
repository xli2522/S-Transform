from s import *
import matplotlib.pyplot as plt

t = 10; rate = 100          # time 10s; sample rate 100 Hz
data = np.linspace(0, t, rate*t)         # time from 0 - 100s, 
table = sTransform(data, sample_rate=100, frange=[0,50])
spec = np.abs(table)

plt.imshow(spec)
plt.show()