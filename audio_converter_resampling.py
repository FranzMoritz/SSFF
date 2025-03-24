import pandas as pd
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import os


def downsample(data, original_fs, target_fs):
    gcd = np.gcd(int(original_fs), int(target_fs))
    # ratio = original_fs / target_fs
    # filtered_data = LPF(data, target_fs / 2, original_fs)
    up = target_fs // gcd
    down = original_fs // gcd
    # downsampled_data = signal.decimate(filtered_data, int(ratio))
    downsampled_data = signal.resample_poly(data, up, down)
    return downsampled_data

path = r"\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\combi_comp\PIDtest_combicomp_20250305_113925.csv"

data = pd.read_csv(path, header=11)
print(data)
# t = data[' Time (s)']
sig = data[' Input B (V)']
original_signal = sig
fs = 407000
new_fs = 44100
num_of_samples = int(len(sig) * new_fs / fs)
sig = signal.resample(sig, num_of_samples)
# sig = downsample(sig, fs, new_fs)
sig = sig - np.mean(sig)
sigmax = np.max(np.abs(sig))
if sigmax > 0:
    sig = sig / sigmax

sig = (sig * 32767).astype(np.int16)

write(os.path.join(r'\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\combi_comp','PID_s1_combi_comp_2khz_3.wav'), new_fs, sig)

f, t, Sxx = signal.spectrogram(original_signal, fs, scaling='spectrum', noverlap=512, nperseg=1024, nfft=1024)
Sxx = 10 * np.log10(Sxx)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.show()

f, t, Sxx = signal.spectrogram(sig, new_fs, scaling='spectrum', noverlap=512, nperseg=1024, nfft=1024)
Sxx = 10 * np.log10(Sxx)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.show()

