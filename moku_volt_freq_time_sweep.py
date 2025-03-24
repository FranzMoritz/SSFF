import os
import json
import numpy as np
from moku.instruments import Oscilloscope
import time
import matplotlib.pyplot as plt

save_path = r'\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\time_measurement\run5\\'
test_description = 'digifit, OPA1612, 330kOhm Rf, closed'

os.makedirs(save_path, exist_ok=True)

instrument = Oscilloscope(ip='10.157.64.203', force_connect=True)
instrument.set_defaults()  # mandatory due to bug

# voltage_list = np.arange(start=0.05, stop=5.05, step=0.5)  # Vpp
voltage_list = np.arange(start=0.1, stop=1.1, step=0.1)
freqs1 = np.arange(start=100, stop=20e3, step=100)
freqs2 = np.arange(start=20e3, stop=101e3, step=1000)
frequency_list = np.concatenate((freqs1, freqs2))
# frequency_list = np.logspace(2, 5, 30)
# voltage_list = np.arange(start=0.1, stop=1.1, step=0.1)
# frequency_list = np.logspace(0, 5, 360)

n = 1
# instrument.disable_output(2)
# instrument.disable_output(3)
# instrument.disable_output(4)
instrument.disable_input(3)
instrument.disable_input(4)
instrument.set_source(1, 'Input1')
instrument.set_source(2, 'Output1')
instrument.set_frontend(1, impedance='1MOhm', coupling='DC', range='40Vpp')
instrument.set_input_attenuation(1, 1)
instrument.set_acquisition_mode(mode="Precision")
instrument.set_output_termination(1, "HiZ")
# instrument.set_frontend(2, impedance='50Ohm', coupling='DC', range='40Vpp')
instrument.set_timebase(0, 20e-3)
settings = {
    'impedance': '1MOhm',
    'coupling': 'DC',
    'range': '40Vpp',
    'sample_rate': instrument.get_samplerate(),
    'commentary': test_description
}
start_time = time.time()

plt.ion()
plt.show()
plt.grid(True)
# plt.ylim([-1, 1])
plt.xlim([0, 20e-3])
line1, = plt.plot([])
line2, = plt.plot([])

# Configure labels for axes
ax = plt.gca()

try:
    for voltage in voltage_list:
        for frequency in frequency_list:
            print(f'Run {n}/{len(voltage_list) * len(frequency_list)}')
            print(f'Set amplitude {voltage}')
            instrument.set_trigger(level=0, source='Input1')
            instrument.generate_waveform(1, type='Sine', amplitude=voltage, frequency=frequency, offset=0)
            # time.sleep(1)
            resp = instrument.get_data(wait_complete=True, wait_reacquire=True)
            for key, value in resp.items():
                print(key, value)
            combined_output = {
                'voltage': voltage,
                'frequency': frequency,
                'settings': settings,
                'data': resp
            }
            file_name = f'{n:03d}_data_{voltage:02.2f}V_{frequency:02.0f}Hz'
            with open(os.path.join(save_path, file_name) + '.json', 'w', encoding='utf-8') as f:
                json.dump(combined_output, f, ensure_ascii=False, indent=4)

            # plotting
            line1.set_ydata(resp['ch1'])
            line2.set_ydata(resp['ch2'])
            line1.set_xdata(resp['time'])
            line2.set_xdata(resp['time'])
            plt.pause(0.001)
            n += 1
            print(f'Total Elapsed time: {time.time() - start_time}')
except Exception as e:
    print(f'Exception: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    instrument.relinquish_ownership()

plt.show()