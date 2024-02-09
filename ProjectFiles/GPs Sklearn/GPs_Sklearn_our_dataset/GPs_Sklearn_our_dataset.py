'''
GP regression on our dataset.
Extension of the file "Model performance".
Here the trained model parameters (initial and optimal hyperparameters with logp and RMSE) of each restart 
are written to a .txt file called "data.txt". (logp is the log marginal likelihood).
The model is evaluated on the train locations.
Note: this file takes very long to execute (with 100 repetitions); about 20 minutes

Input: excel file with dataset
Output: data.txt

Joey Cheung, Dec 2023
'''

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Extract the input and output data
dataframe = pd.read_excel(r"merged_data_offset_final.xlsx")  # Cleaned data
npoints_obs = 584  # 584 datapoints in total

these_features = [1, 2, 3, 4]  # Select first, second, third and fourth feature
nfeatures = len(these_features)  # Max is 35
X, y = dataframe.iloc[:npoints_obs, these_features], dataframe.iloc[:npoints_obs, -1]
X_train = X_test = X
y_train = y
y_train = np.array(y_train).reshape(-1, 1)

feature_names = dataframe.columns[these_features]

# Normalizing 
Xscaler = StandardScaler()  # Standardization initialization for X_train and X_pred (Note: StandardScaler() is a class )
yscaler = StandardScaler()  # Standardization object for y
Xscaler.fit(X_train)
X_train = Xscaler.transform(X_train)
X_test = Xscaler.transform(X_test)

# How many times the GP model should be trained
repetitions = 100

# overwrite = int(input("File will be overwritten! Continue? (0/1): "))
# if overwrite:  # Uncomment this and comment the "if True" statement if you want warning before overwriting the .txt file

if True:
    # Write data to .txt file
    with open("data.txt",'w') as f:
        f.write("Features" + str(these_features) + ", " + str(nfeatures) + " features\n")
        f.write("Initial linear-valued hyperparameters | Optimal hyperparameters | Logp | RMSE\n")
            
    for i in range(repetitions):
        # Normalizing
        y_train = yscaler.fit_transform(y_train)

        # Initialize random hyperparameters
        random_nums = np.random.uniform(-1, 2, size=(2, ))
        random_nums = 10**random_nums
        random_sigma = random_nums[0]; random_noise = random_nums[1]
        length_scales = 1.0 * np.ones(nfeatures)
        lower_power = -5; upper_power = 5
        length_scale_bounds = [1*10**lower_power, 1*10**upper_power]  # == [1e-4, 1e4] for example
        random_length_scales = np.random.uniform(lower_power, upper_power, nfeatures)
        random_length_scales = 10**random_length_scales

        # This should stay 1
        n_restarts = 1

        # Define the kernel function
        kernel = random_sigma * RBF(length_scale=random_length_scales, length_scale_bounds=length_scale_bounds) + WhiteKernel(
        noise_level=random_noise, noise_level_bounds=(1e-10, 1e2)
        )
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=None).fit(X_train, y_train)

        # Predict on test locations, aka the train locations
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

        # Obtain RMSE
        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse)

        # Append the parameters of this trained model to the .txt file
        with open("data.txt", "a") as f:
            np.savetxt(f, initial_normal_hyperparameters, fmt="%.3e", newline=" ")
            f.write("| ")
            np.savetxt(f, normal_hyperparameters, fmt="%.3e", newline=" ")
            f.write("| ")
            f.write(str(np.round(logp, decimals=3)))
            f.write(" | ")
            f.write(str(np.round(rmse, decimals=3)))
            f.write("\n")

    print("Done!")

else:
    print("File not overwritten!")