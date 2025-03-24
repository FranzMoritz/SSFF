import os
import json
import numpy as np
from moku.instruments import MultiInstrument
from moku.instruments import Oscilloscope, WaveformGenerator
import time
import matplotlib.pyplot as plt


save_path = r'\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\digifit_data\digifit_330k_v2\\'
test_description = 'digifit, OPA1612, 330kOhm Rf, open'

voltage_list = np.arange(start=0.05, stop=1.05, step=0.05)  # Vpp
frequency_list = [1e3, 2e3, 6e3, 10e3, 12e3, 15e3, 18e3, 20e3]
n = 1
instrument = MultiInstrument(ip='10.157.64.203', force_connect=True, platform_id=4)
start_time = time.time()
try:
    wg = instrument.set_instrument(1, WaveformGenerator)
    osc = instrument.set_instrument(2, Oscilloscope)
    osc.disable_input(3)
    osc.disable_input(4)

    connections = [dict(source="Input1", destination="Slot1InA"),
                   dict(source="Slot1OutA", destination="Slot2InA"),
                   dict(source="Slot1OutA", destination="Slot2InB"),
                   dict(source="Slot2OutA", destination="Output1")]
    print(instrument.set_connections(connections=connections))

    wg.generate_waveform(1, "Sine", frequency=1e3, amplitude=0.1, offset=0.0)
    osc.set_timebase(-5e-3, 5e-3)
    data = osc.get_data()
    print(data)
    for key, value in data.items():
        print(key, value)

    plt.ion()
    plt.show()
    plt.grid(True)
    plt.ylim([-1, 1])
    plt.xlim([data['time'][0], data['time'][-1]])

    line1, = plt.plot([])
    line2, = plt.plot([])
    # Configure labels for axes
    ax = plt.gca()
        # Update the plot
    line1.set_ydata(data['ch1'])
    line2.set_ydata(data['ch2'])
    line1.set_xdata(data['time'])
    line2.set_xdata(data['time'])
    plt.pause(0.001)

    combined_output = {
        'voltage': voltage,
        'settings': settings,
        'data': frame['ch1']
    }
    file_name = f'{n:03d}_data_{voltage:02.2f}V'
    with open(os.path.join(save_path, file_name) + '.json', 'w', encoding='utf-8') as f:
        json.dump(combined_output, f, ensure_ascii=False, indent=4)
    n += 1
    print(f'Total Elapsed time: {time.time() - start_time}')
except Exception as e:
    print(f'Exc {e}')

