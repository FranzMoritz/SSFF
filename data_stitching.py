import json
import pandas as pd
import os

path = r'\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\time_measurement\run4'

file_list = sorted([f for f in os.listdir(path) if f.endswith('.json')],
                   key=lambda x: int(x.split('_')[0]))

data_frames = []
time_offset = 0
for filename in file_list:
    filepath = os.path.join(path, filename)
    print(filename)
    with open(filepath, 'r') as file:
        data = json.load(file)
        time = data['data'].get('time', [])
        time = [t + time_offset for t in time]
        df = pd.DataFrame({
            'x': data['data'].get('ch2', []),
            'y': data['data'].get('ch1', []),
            't': time
        })
        if time:
            time_offset = time[-1] + (time[1] - time[0] if len(time) > 1 else 1)
        data_frames.append(df)
result = pd.concat(data_frames, ignore_index=True)
result.to_csv('run4test2.csv', index=False)