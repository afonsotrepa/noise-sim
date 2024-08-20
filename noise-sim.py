import numpy as np
import matplotlib.pyplot as plt

fig, (time, freq) = plt.subplots(1, 2)
time.set_title("Time")
freq.set_title("Frequency")

arr = np.random.rand(int(1e6)) - 0.5

time.plot(arr.cumsum(), "b", label="Original")

fft = np.fft.rfft(arr)
freq.plot(np.absolute(fft), "b", label="Original")

f_t = int(1e5) #cutoff frequency
fft[:f_t] = [0]*f_t #apply high-pass filter

freq.plot(np.absolute(fft), "r", label="High pass filtered")

lp_arr = np.fft.irfft(fft)
lp_arr *= 1+ (f_t/len(arr)) #adjust amplitude to keep same total noise energy

time.plot(lp_arr.cumsum(), "r", label="High pass filtered")


time.legend(loc="upper center")
freq.legend(loc="upper center")
plt.show()
