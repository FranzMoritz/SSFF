import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = r"\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\combi_comp\combicomp_20250305_115552_Traces.csv"
data = pd.read_csv(path, header=10)
print(data)
frequencies = data['% Frequency (Hz)']
magnitude_db = data[' Channel 1 Magnitude (dB)']
phase_deg = data[' Channel 1 Phase (deg)']

mag_linear = 10**(magnitude_db / 20)
phase_rad = np.deg2rad(phase_deg)
real_part = mag_linear * np.cos(phase_rad)
imag_part = mag_linear * np.sin(phase_rad)

plt.figure()
plt.plot(real_part, imag_part, label='nyquist')
# plt.plot(real_part, -imag_part, linestyle='--', color='gray')
plt.axhline(0, color='k')
plt.axvline(0, color='k')
plt.plot(-1, 0, 'rx', label='critical point')
unit_circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', label='unit circle')
plt.gca().add_artist(unit_circle)
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
plt.legend()
plt.grid(True)
plt.show()