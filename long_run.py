import os
import json
import numpy as np
from moku.instruments import Oscilloscope
import time
import matplotlib.pyplot as plt

save_path = r'\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\time_measurement\test4\\'
test_description = 'digifit, OPA1612, 330kOhm Rf, open'

os.makedirs(save_path, exist_ok=True)

instrument = Oscilloscope(ip='10.157.64.203', force_connect=True)
instrument.set_defaults()


# voltage_list = np.arange(start=0.05, stop=5.05, step=0.5)  # Vpp
# voltage_list = np.arange(start=0.1, stop=1.1, step=0.1)
# frequency_list = np.logspace(0, 5, 360)
voltage_list = np.arange(start=0.1, stop=5.1, step=0.5)
frequency_list = np.logspace(0, 5, 20)
# freqs = np.logspace(0, 5, int(n))

settings = {
    'start_frequency': 500,
    'stop_frequency': 100e3,
    'sweep_length': 1024,
    'log_scale': True,
    'averaging_time': 1e-3,
    'settling_time': 1e-3,
    'averaging_cycles': 5,
    'settling_cycles': 5,
    'impedance': '1MOhm',
    'coupling': 'AC',
    'range': '40Vpp',
    'commentary': test_description
}
n = 1
# instrument.disable_output(2)
# instrument.disable_output(3)
# instrument.disable_output(4)
instrument.disable_input(3)
instrument.disable_input(4)
instrument.set_source(1, 'Input1')
instrument.set_source(2, 'Output1')
instrument.set_frontend(1, impedance='50Ohm', coupling='DC', range='40Vpp')
instrument.set_input_attenuation(1, 1)
# instrument.set_frontend(2, impedance='50Ohm', coupling='DC', range='40Vpp')
instrument.set_timebase(0, 10)
start_time = time.time()

plt.ion()
plt.show()
plt.grid(True)
# plt.ylim([-1, 1])
plt.xlim([0, 10])
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
            time.sleep(0.4)
            instrument.generate_waveform(1, type='Sine', amplitude=voltage*2, frequency=frequency, offset=0)
            resp = instrument.get_data(wait_complete=True, wait_reacquire=True)
            print(resp)
            for key, value in resp.items():
                print(key, value)
            # time.sleep(resp['estimated_sweep_time'])

            # frame = instrument.get_data(wait_complete=True)
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