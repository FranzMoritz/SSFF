import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from moku.nn import LinnModel, save_linn

# get data
input_data = pd.read_csv(r"\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\time_measurement\DigifitTimeSweep_20250113_145110.csv",
                         header=11)
print(input_data)
x_full = input_data[' Input B (V)'].to_numpy()
# print(type(x_full))
# print()
y_full = input_data[' Input A (V)'].to_numpy()

in1 = np.array(x_full)
out = np.array(y_full)

inputs = in1.reshape(-1,1)
out = out.reshape(-1,1)

input_data = pd.read_csv(r"\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\time_measurement\DigifitTimeSweep_20250113_145120.csv",
                         header=11)
x_full_valid = input_data[' Input B (V)'].to_numpy()
y_full_valid = input_data[' Input A (V)'].to_numpy()

input_validation = x_full_valid.reshape(-1,1)
output_validation = y_full_valid.reshape(-1,1)

fig, axs = plt.subplots(2)
axs[0].plot(inputs, label='training')
axs[0].plot(input_validation, label='validation')
axs[0].legend()
axs[1].plot(out, label='training')
axs[1].plot(output_validation, label='validation')
axs[1].legend()
plt.show()

# create model
linn_model = LinnModel()
linn_model.set_training_data(training_inputs=inputs, training_outputs=out, scale=False)

model_definition = [(64, 'relu'), (64, 'relu'), (64, 'relu'), (64, 'relu'), (1, 'linear')]
linn_model.construct_model(model_definition, show_summary=True)
es_config = {'patience': 100, 'restore': True}
history = linn_model.fit_model(epochs=500, es_config=es_config, validation_data=(input_validation, output_validation))

plt.figure()
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.title('Loss functions')
plt.show()

nn_out = linn_model.predict(inputs)
print(nn_out)
print(nn_out.shape)
fig, axs = plt.subplots(2)
axs[0].plot(out, label='desired')
axs[0].plot(nn_out, '--', label='Model output')
axs[0].legend()
axs[1].plot(nn_out - out, label='model-desired')
axs[1].legend()
plt.show()

save_linn(linn_model.model, input_channels=1, output_channels=1, file_name='demo_model4.linn')
