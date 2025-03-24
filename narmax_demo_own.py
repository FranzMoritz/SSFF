import numpy as np
import pandas as pd
from Tools.demo.sortvisu import steps
from sysidentpy.model_structure_selection import FROLS, ER
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares, RecursiveLeastSquares
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_results
from sysidentpy.utils.save_load import save_model, load_model


# Generating 1 input 1 output sample data from a benchmark system
# x_train, x_valid, y_train, y_valid = get_siso_data(
#     n=1000, colored_noise=False, sigma=0.0001, train_percentage=90
# )
# print(x_train)
# print(type(x_train))
# print(x_train.shape)

input_data = pd.read_csv('run4test2.csv')
x_full = input_data['x'].to_numpy()
# print(type(x_full))
# print()
y_full = input_data['y'].to_numpy()

x_full = x_full[0:5000]
y_full = y_full[0:5000]

split_horizontally_idx = int(x_full.shape[0]* 0.8) # integer for line selection (horizontal selection)

x_train = x_full[:split_horizontally_idx].reshape(-1, 1) # indexing/selection of the 80%
x_valid = x_full[split_horizontally_idx:].reshape(-1, 1) # indexing/selection of the remaining 20%
y_train = y_full[:split_horizontally_idx].reshape(-1, 1) # indexing/selection of the 80%
y_valid = y_full[split_horizontally_idx:].reshape(-1, 1) # indexing/selection of the remaining 20%

# x_train = np.asarray(x_train)
# x_valid = np.asarray(x_valid)
# y_train = np.asarray(y_train)
# y_valid = np.asarray(y_valid)


basis_function = Polynomial(degree=1)
# estimator = LeastSquares()
estimator = RecursiveLeastSquares()
# model = FROLS(
#     order_selection=True,
#     n_info_values=5,
#     ylag=list(range(1, 8)),
#     xlag=2,
#     info_criteria="bic",
#     estimator=estimator,
#     basis_function=basis_function
# )

model = ER(
    ylag=6,
    xlag=6,
    n_perm=2,
    k=2,
    skip_forward=True,
    estimator=estimator,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)

yhat = model.predict(X=x_valid, y=y_valid, steps_ahead=1)

# Gathering results
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
# save_model(model_variable, file_name.syspy, path (optional))
save_model(model=model, file_name="model_name.syspy")

#
# # load_model(file_name.syspy, path (optional))
# loaded_model = load_model(file_name="model_name.syspy")
#
# # Predicting output with loaded_model
# yhat_loaded = loaded_model.predict(X=x_valid, y=y_valid, steps_ahead=1)
#
# r_loaded = pd.DataFrame(
#     results(
#         loaded_model.final_model,
#         loaded_model.theta,
#         loaded_model.err,
#         loaded_model.n_terms,
#         err_precision=8,
#         dtype="sci",
#     ),
#     columns=["Regressors", "Parameters", "ERR"],
# )
#
# # Printing both: original model and model loaded from file
# print("\n Original model \n", r)
# print("\n Model Loaded from file \n", r_loaded)
#
# # Checking predictions from both: original model and model loaded from file
# if (yhat == yhat_loaded).all():
#     print("\n Predictions are the same!")

# Ploting results
plot_results(y=y_valid, yhat=yhat, n=20000)
