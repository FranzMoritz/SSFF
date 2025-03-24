import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from moku.nn import LinnModel, save_linn

# get data
input_data = pd.read_csv(r"\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\linn_measurements\DigifitTimeSweep_20250219_115307.csv",
                         header=11)
print(input_data)
x_full = input_data[' Input B (V)'].to_numpy()
# print(type(x_full))
# print()
y_full = input_data[' Input A (V)'].to_numpy()

# x_full = x_full[30000:180000]
# y_full = y_full[30000:180000]

in1 = np.array(x_full)
out = np.array(y_full)
print(in1)
print(out)
# t = np.linspace(0, 2*np.pi, 1000)
# in1 = np.sin(t)
# out = np.sin(t) * 3


inputs = in1.reshape(-1,1)
out = out.reshape(-1,1)

fig, axs = plt.subplots(2)
axs[0].plot(inputs, label='inputs')
axs[0].legend()
axs[1].plot(out, label='out')
axs[1].legend()
plt.show()

# inputs.shape = [len(in1), 1]
# out.shape = [len(out), 1]

print(in1.shape)
print(inputs.shape)
print(out.shape)
# create model
linn_model = LinnModel()
linn_model.set_training_data(training_inputs=inputs, training_outputs=out, scale=False)

model_definition = [(64, 'relu'), (64, 'relu'), (64, 'relu'), (64, 'relu'), (1, 'linear')]
linn_model.construct_model(model_definition, show_summary=True)
history = linn_model.fit_model(epochs=500, validation_split=0.15, es_config={'patience': 10})

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
