# GP regression on our dataset: (features_of_rocks, yield_points)
# Joey Cheung, Dec 2023

from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Extract the input and output data
npoints_obs = 100
dataframe1 = pd.read_excel(r"merged_data1.xlsx")
X_train = dataframe1.iloc[:npoints_obs, [1,-2]]
X_train = X_train.to_numpy()
y_train = dataframe1.iloc[:npoints_obs, -1]
y_train = y_train.to_numpy()
y_train = y_train.reshape(-1, 1)

# Create random prediction locations
npoints_pred = 50 # Number of prediction points in 1D 
                   # (so number of prediction points in 2D will be the value squared)
rng = np.random.default_rng(456)
X_pred1 = rng.uniform(min(X_train[:, 0]), max(X_train[:, 0]), (npoints_pred,))
X_pred2 = rng.uniform(min(X_train[:, 1]), max(X_train[:, 1]), (npoints_pred,))

# Sorting is needed for plotting the prediction surface
X_pred1 = np.sort(X_pred1)
X_pred2 = np.sort(X_pred2)

# Reshape the 2D arrays into 1D arrays 
# --> needed for GP predict function
XX_pred1, XX_pred2 = np.meshgrid(X_pred1, X_pred2)
XX_pred1_L = XX_pred1.reshape(-1, 1)
# print(XX_pred1_L.shape)
XX_pred2_L = XX_pred2.reshape(-1, 1)
XX_pred_L = np.hstack([XX_pred1_L, XX_pred2_L])
X_pred = XX_pred_L

# Normalizing 
Xscaler = StandardScaler()  # Standardization initialization for X_train and X_pred (Note: StandardScaler() is a class )
yscaler = StandardScaler()  # Standardization object for y
Xscaler.fit(X_train)
X_train = Xscaler.transform(X_train)
X_pred = Xscaler.transform(X_pred)
y_train = yscaler.fit_transform(y_train)

# Define the kernel function
kernel = 1.0 * RBF(length_scale=[1e+0,1e+0], length_scale_bounds=(1e-5, 1e3)) + WhiteKernel(
    noise_level=1e-1, noise_level_bounds=(1e-10, 1e1)
)
# kernel = DotProduct() + WhiteKernel()  # Other kernel choice
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train)
y_pred = gpr.predict(X_pred)

# Denormalize
X_train = Xscaler.inverse_transform(X_train)
X_train1 = X_train[:, 0]
X_train2 = X_train[:, 1]

X_pred = Xscaler.inverse_transform(X_pred)
X_pred1 = X_pred[:, 0]
X_pred2 = X_pred[:, 1]

y = yscaler.inverse_transform(y_train)
y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1))

# 3D scatterplot
fig2 = plt.figure()

axnew = fig2.add_subplot(projection='3d')
axnew.scatter(X_train1, X_train2, y, color="green", label="Training data/observations")
# axnew.scatter(X_pred1, X_pred2, y_pred, color="orange", label="Predictions on random locs")

# Plot the prediction surface
y_pred_2d = y_pred.reshape(npoints_pred, npoints_pred)
surf = axnew.plot_surface(XX_pred1, XX_pred2, y_pred_2d,
                          color="orange", linewidth=0,
                          label="Prediction surface")


axnew.set_xlabel('x')
axnew.set_ylabel('y')
axnew.set_zlabel('z')
axnew.legend()

axnew.set_title(
    (
        f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
        f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}"
    ),
    fontsize=8,
    )
plt.tight_layout()
plt.show()