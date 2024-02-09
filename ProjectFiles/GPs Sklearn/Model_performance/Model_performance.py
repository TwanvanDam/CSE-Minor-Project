'''
GP regression on our dataset: (inputs, output) = (features_of_rocks, yield_points)

Input: excel file with dataset
Output: plot of the trained model with parameter information about the model

Joey Cheung, Dec 2023
'''

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# First flag is for training the GP model without test-train split
model = True
# Second flag is for training the GP model with test-train split
split = not(model)

def plot_stuff():
    if model:
        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse)

        # See how good model performs: compare y_train and y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4.5))
        ax1.scatter(y_train, y_pred, color="blue", label="Predicitions")
        ax1.plot([0, 120], [0, 120], color="black", label="Truth values", linestyle="dashed")
        ax1.set_xlabel("y_train")
        ax1.set_ylabel("y_pred")
        ax1.set_title(f"y_train vs y_pred")
        ax1.legend()

    if split:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # See how good model performs: compare y_test and y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4.5))
        ax1.scatter(y_test, y_pred, color="blue", label="Predicitions")
        ax1.plot([0, 120], [0, 120], color="black", label="Truth values", linestyle="dashed")
        ax1.set_xlabel("y_test")
        ax1.set_ylabel("y_pred")
        ax1.set_title(f"y_test ({test_size*100}%) vs y_pred")
        ax1.legend()

    # Set both x- and y-axis limits to [0, 10] instead of default [0, 1]
    ax2.axis([0, 10, 0, 10])

    # This will make sure to print only 2 digits after the dot (and scientific notation) 
    np.set_printoptions(formatter={'float': lambda x: "{:.2e}".format(x)})

    ax2.text(0.5, 10, 'Info', weight='bold', fontsize=15,
            bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10})
    ax2.text(0.5, 9, "GP model:", weight="bold")
    ax2.text(   0.5, 5,
            f"The linear-valued hyperparameters\n" 
            f"Initial: [Scaling factor, length_scales, noise] are\n"
            f"{initial_normal_hyperparameters}\n\n"
            f"Optimum: [Scaling factor, length_scales, noise] are\n"
            f"{normal_hyperparameters}\n\n"
            f"log marginal likelihood: {np.round(gpr.log_marginal_likelihood(gpr.kernel_.theta), decimals=3)}\n"
            f"n_restarts: {n_restarts}\n"
            f"{these_features} = {feature_names}",
            fontsize=9
            )

    ax2.text(0.5, 3, "y_train:", weight="bold")
    ax2.text(   0.5, 1,
            f"Min yield strength is {np.min(y_train)}\n"
            f"Max yield strength is {np.max(y_train)}\n"
            f"Mean is {np.mean(y_train)}\n\n"
            f"RMSE: {rmse}",
            fontsize=9
            )

    plt.show()

# Extract the input and output data
dataframe = pd.read_excel(r"merged_data_offset_final.xlsx")  # Cleaned data
npoints_obs = 584  # 584 datapoints in total

these_features = [1, 2, 3, 4]  # Select first, second, third and fourth feature
nfeatures = len(these_features)  # Max is 35
X, y = dataframe.iloc[:npoints_obs, these_features], dataframe.iloc[:npoints_obs, -1]
if model:
    X_train = X_test = X
    y_train = y
if split:
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)

y_train = np.array(y_train).reshape(-1, 1)

feature_names = dataframe.columns[these_features]

# Normalizing 
Xscaler = StandardScaler()  # Standardization initialization for X_train and X_pred (Note: StandardScaler() is a class )
yscaler = StandardScaler()  # Standardization object for y
Xscaler.fit(X_train)
X_train = Xscaler.transform(X_train)
X_test = Xscaler.transform(X_test)

# random_sigma = 1.0
# random_noise = 1e-1
# length_scale_bounds = [1e-2, 1e5]
# random_length_scales = 1.0 * np.ones(nfeatures)

# Normalizing
y_train = yscaler.fit_transform(y_train)

# Initialize random hyperparameters
# random_nums = np.random.uniform(-1, 2, size=(2, ))
# random_nums = 10**random_nums
# random_sigma = random_nums[0]
# random_noise = random_nums[1]
# length_scales = 1.0 * np.ones(nfeatures)
# lower_power = -5; upper_power = 5
# length_scale_bounds = [1*10**lower_power, 1*10**upper_power]  # == [1e-4, 1e4] for example
# random_length_scales = np.random.uniform(lower_power, upper_power, nfeatures)
# random_length_scales = 10**random_length_scales

# Initialize pre-determined hyperparameters
length_scale_bounds = [1e-2, 1e5]
random_sigma = 1e-1; 
random_noise = 1e-1; 
random_length_scales = 1e0 * np.ones(nfeatures)

# Define how many times you want to train the GP model --> 
# At every restart different random intial values for hyperparameters are used
n_restarts = 1

# Define the kernel function
kernel = random_sigma * RBF(length_scale=random_length_scales, length_scale_bounds=length_scale_bounds) + WhiteKernel(
noise_level=random_noise, noise_level_bounds=(1e-10, 1e2)
)
# kernel = DotProduct() + WhiteKernel()  # Other kernel choice
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42).fit(X_train, y_train)

# Predict on test locations
y_pred = gpr.predict(X_test)
y_pred = y_pred.reshape(-1, 1)

# Denormalize output variables only
y_train = yscaler.inverse_transform(y_train)
y_pred = yscaler.inverse_transform(y_pred)

initial_normal_hyperparameters = np.exp(gpr.kernel.theta)

# Print the initial and optimal kernel functions
print(f"Initial: {kernel}\nOptimum: {gpr.kernel_}\n")

logp = np.round(gpr.log_marginal_likelihood(gpr.kernel_.theta), decimals=3)
print(f"Log-Marginal-Likelihood: {logp}")

log_hyperparameters = gpr.kernel_.theta  # Numpy array
normal_hyperparameters = np.exp(log_hyperparameters)  # length scales are normal_hyperparameters[1:-1]

# Plot the model results: either on train locations or test locations for 1st and second flag respectively
plot_stuff()
