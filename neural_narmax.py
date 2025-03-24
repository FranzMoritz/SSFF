from torch import nn
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.neural_network import NARXNN
import pandas as pd
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
from sysidentpy.utils.narmax_tools import regressor_code
from sysidentpy.utils.save_load import save_model, load_model
import torch
torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

input_data = pd.read_csv(r"\\nas.ads.mwn.de\ge35haf\TUM-PC\Dokumente\MasterThesis\Code\moku_measurements\time_measurement\DigifitTimeSweep_20250113_145058.csv",
                         header=11)
print(input_data)
x_full = input_data[' Input B (V)'].to_numpy()
# print(type(x_full))
# print()
y_full = input_data[' Input A (V)'].to_numpy()


# x_full = x_full[0:10000]
# y_full = y_full[0:10000]

split_horizontally_idx = int(x_full.shape[0]* 0.8) # integer for line selection (horizontal selection)

x_train = x_full[:split_horizontally_idx].reshape(-1, 1) # indexing/selection of the 80%
x_valid = x_full[split_horizontally_idx:].reshape(-1, 1) # indexing/selection of the remaining 20%
y_train = y_full[:split_horizontally_idx].reshape(-1, 1) # indexing/selection of the 80%
y_valid = y_full[split_horizontally_idx:].reshape(-1, 1) # indexing/selection of the remaining 20%


basis_function = Polynomial(degree=1)

narx_net = NARXNN(
    ylag=5,
    xlag=5,
    basis_function=basis_function,
    model_type="NARMAX",
    loss_func="mse_loss",
    optimizer="Adam",
    epochs=2000,
    verbose=False,
    device=device,
    optim_params={
        "betas": (0.9, 0.999),
        "eps": 1e-05,
    },  # optional parameters of the optimizer
)

regressors = regressor_code(
    X=x_train,
    xlag=5,
    ylag=5,
    model_type="NARMAX",
    model_representation="neural_network",
    basis_function=basis_function,
)
n_features = regressors.shape[0]
class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(n_features, 30)
        self.lin2 = nn.Linear(30, 30)
        self.lin3 = nn.Linear(30, 1)
        self.tanh = nn.Tanh()

    def forward(self, xb):
        z = self.lin(xb)
        z = self.tanh(z)
        z = self.lin2(z)
        z = self.tanh(z)
        z = self.lin3(z)
        return z

narx_net.net = NARX()
narx_net.fit(X=x_train, y=y_train, X_test=x_valid, y_test=y_valid)
yhat = narx_net.predict(X=x_valid, y=y_valid, steps_ahead=1)
save_model(model=narx_net, file_name="neuralSS2.syspy")
print("MSE: ", mean_squared_error(y_valid, yhat))
plot_results(y=y_valid, yhat=yhat, n=100000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
