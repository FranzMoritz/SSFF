import os
import json
import numpy as np
from moku.instruments import FrequencyResponseAnalyzer
import time
import matplotlib.pyplot as plt

save_path = r'\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\digifit_data\digifit_330k_v2\\'
test_description = 'digifit, OPA1612, 330kOhm Rf, open'

os.makedirs(save_path, exist_ok=True)

instrument = FrequencyResponseAnalyzer(ip='10.157.64.203', force_connect=True)

voltage_list = np.arange(start=0.05, stop=1.05, step=0.05)  # Vpp
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
instrument.disable_output(2)
instrument.disable_output(3)
instrument.disable_output(4)
instrument.set_frontend(1, impedance=settings['impedance'],
                        coupling=settings['coupling'],
                        range=settings['range'])
instrument.measurement_mode(mode='In')
# instrument.measurement_mode(mode='InOut')
start_time = time.time()
try:
    for voltage in voltage_list:
        print(f'Run {n}/{len(voltage_list)}')
        instrument.set_output(channel=1, amplitude=voltage,
                              offset=0, enable_offset=False, enable_amplitude=True)
        instrument.set_sweep(start_frequency=settings['start_frequency'],
                             stop_frequency=settings['stop_frequency'],
                             num_points=settings['sweep_length'],
                             averaging_time=settings['averaging_time'],
                             settling_time=settings['settling_time'],
                             averaging_cycles=settings['averaging_cycles'],
                             settling_cycles=settings['settling_cycles'])
        instrument.start_sweep()
        resp = instrument.get_sweep()
        print(resp)
        # time.sleep(resp['estimated_sweep_time'])

        frame = instrument.get_data(wait_complete=True)
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
        instrument.stop_sweep()
        instrument.set_output(channel=1, amplitude=0.05,
                              offset=0, enable_offset=False, enable_amplitude=False)

        # plotting

except Exception as e:
    print(f'Exception: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    instrument.relinquish_ownership()
