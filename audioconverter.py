import pandas as pd
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import os

path = r"\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\PID_measurements\PIDtest_lid_20250303_122340.csv"

data = pd.read_csv(path, header=11)
print(data)
# t = data[' Time (s)']
sig = data[' Input B (V)']
original_signal = sig
fs = 44100
sig = sig - np.mean(sig)
sigmax = np.max(np.abs(sig))
if sigmax > 0:
    sig = sig / sigmax

sig = (sig * 32767).astype(np.int16)

write(os.path.join(r'\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\PID_measurements','PID_s1_3.wav'), fs, sig)

f, t, Sxx = signal.spectrogram(original_signal, fs, scaling='spectrum', noverlap=512, nperseg=1024, nfft=1024)
Sxx = 10 * np.log10(Sxx)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.show()
