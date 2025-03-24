import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from moku.nn import LinnModel, save_linn
import pandas as pd
import os

folder_path = r'\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\linn_measurements'

# Initialize an empty list to store individual DataFrames
dataframes = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, header=11)
        # Append the DataFrame to the list
        dataframes.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

data_length = 100000
block_size = 100

def prepare_batches_non_overlapping(x_data, y_data):
    batches = []
    labels = []
    for i in range(data_length):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        if end_idx > len(x_data):
            break  # Ensure we don't exceed array bounds
        batch_element = x_data[start_idx:end_idx]
        label = y_data[end_idx - 1]
        batches.append(batch_element)
        labels.append(label)

    return np.array(batches), np.array(labels)

def prepare_batches_overlapping(x_data, y_data):
    batches = []
    labels = []
    for i in range(len(x_data) - block_size - 1):
        start_idx = i
        end_idx = start_idx + block_size
        if end_idx > len(x_data):
            break  # Ensure we don't exceed array bounds
        batch_element = x_data[start_idx:end_idx]
        label = y_data[end_idx - 1]
        batches.append(batch_element)
        labels.append(label)
    return np.array(batches), np.array(labels)

x_data_full = combined_df[' Input B (V)'].to_numpy()
y_data_full = combined_df[' Input A (V)'].to_numpy()

input_data, output_data = prepare_batches_overlapping(x_data_full, y_data_full)
input_data.shape = [-1, block_size]
output_data = output_data.reshape(-1, 1)

quant_mod = LinnModel()
quant_mod.set_training_data(training_inputs=input_data, training_outputs=output_data)
model_definition = [(16, 'tanh'), (16, 'tanh'), (16, 'tanh'), (1, 'linear')]

# build the model
quant_mod.construct_model(model_definition)
early_stopping_config = {'patience': 40, 'restore': True}

# set the training data
history = quant_mod.fit_model(epochs=200, es_config=early_stopping_config, validation_split=0.15)


# nn_out = quant_mod.predict(input_validation)
# plt.plot(nn_out, '--', label='predicted')
# plt.plot(output_validation, label='data')
# plt.grid(True)
# plt.legend()
# plt.show()

save_linn(quant_mod, input_channels=1, output_channels=1, file_name='demo7.linn')

plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'])
plt.xlabel('Epochs')
plt.show()

nn_out = quant_mod.predict(input_data)
plt.plot(nn_out, '--', label='predicted')
plt.plot(output_data, label='data')
plt.grid(True)
plt.legend()
plt.show()