import os
import json
import numpy as np
from moku.instruments import FrequencyResponseAnalyzer
import time
import matplotlib.pyplot as plt

instrument = FrequencyResponseAnalyzer(ip='10.157.64.203', force_connect=True)

instrument.set_output(1, 0.1)
instrument.set_sweep(start_frequency=500, stop_frequency=100e3, num_points=512,
                     averaging_time=1e-3, averaging_cycles=2, settling_cycles=2, settling_time=1e-3)
instrument.start_sweep()

frame = instrument.get_data()
plt.plot(frame['ch1']['frequency'], frame['ch1']['magnitude'])
plt.show()

instrument.relinquish_ownership()
data = frame['ch1']
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
