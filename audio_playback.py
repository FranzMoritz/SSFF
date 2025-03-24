import pandas as pd
import numpy as np
from scipy.io.wavfile import write


path = r"\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\linn_measurements\PIDtest_lid_20250224_135736.csv"

data = pd.read_csv(path, header=11)
print(data)
# t = data[' Time (s)']
signal = data[' Input B (V)']


fs = 44100
signal = signal - np.min(signal)
signal = signal / np.max(signal)
signal = (signal * 65535 - 32768).astype(np.int16)

write('PIDoutput_lid_s1_diffcable.wav', fs, signal)