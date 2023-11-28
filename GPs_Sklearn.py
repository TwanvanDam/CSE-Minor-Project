# GP regression on dummy dataset: California housing. Later the real dataset (features_of_rocks, yield_points) will be used
# Joey Cheung, Nov 2023

from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def create_pred_locations(input_data, points, zeros=False):
    min1 = np.min(input_data)
    max1 = np.max(input_data)

    if not zeros:
        X_predi = np.linspace(min1,max1,points)
        X_predi = X_predi.reshape(-1,1)
    if zeros:
        X_predi = np.zeros(points)
        X_predi = X_predi.reshape(-1,1)

    return X_predi

def plot_GPs():
    global y_pred

    # Scatterplots with the regression line
    fig, ax = plt.subplots(ncols=3, figsize=(8,5))
    ax[0].scatter(X_train1,y)
    ax[0].plot(X_pred1_nonzeros,y_pred[:observations], color="green")
    ax[0].set_xlabel("x-axis (first feature axis)")
    ax[0].set_ylabel("Target y")

    ax[1].scatter(X_train3,y)
    ax[1].plot(X_pred3_nonzeros,y_pred[2*observations:3*observations], color="green")
    ax[1].set_xlabel("y-axis (second feature axis)")
    ax[1].set_ylabel("Target y")

    y_pred_at_train_locations = gpr.predict(X_train)
    ax[2].scatter(y,y_pred_at_train_locations)
    ax[2].set_xlabel("y")
    ax[2].set_ylabel("y_pred")

    # Miss 3D scatterplot toevoegen? --> y_pred[observations:2*observations]
    # fig2 = plt.figure()
    # axnew = fig2.add_subplot(projection='3d')
    # axnew.scatter(X_pred1_nonzeros, X_pred3_nonzeros, y_pred[observations:2*observations], label="Predicited data")
    # axnew.scatter(X_train1, X_train3, y, label="Training data")
    # axnew.set_xlabel('x')
    # axnew.set_ylabel('y')
    # axnew.set_zlabel('z')
    # axnew.legend()

    plt.tight_layout()
    plt.show()

def plot_lr():
    lr = LinearRegression()
    lr.fit(X_train,y)
    y_pred_lr = lr.predict(X_pred)

    fig, ax = plt.subplots(ncols=3, figsize=(8,5))
    ax[0].scatter(X_train1,y)
    ax[0].plot(X_pred1_nonzeros,y_pred_lr[:observations], color="green")

    ax[1].scatter(X_train3,y)
    ax[1].plot(X_pred3_nonzeros,y_pred_lr[2*observations:3*observations], color="green")

    y_pred_at_train_locations = gpr.predict(X_train)
    ax[2].scatter(y,y_pred_at_train_locations)
    plt.show()

# =========================================================================
# Start main code

# Extract the input and output data
Cali = fetch_california_housing(return_X_y=True, as_frame=False)
data = Cali[0]
observations = 100
data1 = data[:observations,0]  # (observations,)  # First feature (e.g. median income in block group -- large impact)
data3 = data[:observations,3]  # (observations,)  # Second feature (e.g. average number of bedrooms per household -- less impact)
y = (Cali[1])[:observations]  # Shape is (observations,)  median house value for California districts
 
# Make the feature matrix
X_train1 = data1.reshape(-1,1)
X_train3 = data3.reshape(-1,1)
X_train = np.hstack([X_train1,X_train3])  # (observations,2)

# Normalizing
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# y = scaler.fit_transform(y.reshape(-1,1))
# X_train1 = X_train[:,0]
# X_train3 = X_train[:,1]

# Create prediction locations
X_pred1_zeros = create_pred_locations(X_train1, points=observations, zeros=True)
X_pred1_nonzeros = create_pred_locations(X_train1, points=observations, zeros=False)
X_pred1 = np.vstack([X_pred1_nonzeros, X_pred1_nonzeros, X_pred1_zeros])
# print(X_pred1_nonzeros)
# print(X_train1)

X_pred3_zeros = create_pred_locations(X_train3, points=observations, zeros=True)
X_pred3_nonzeros = create_pred_locations(X_train3, points=observations, zeros=False)
X_pred3 = np.vstack([X_pred3_zeros, X_pred3_nonzeros, X_pred3_nonzeros])
X_pred = np.hstack([X_pred1,X_pred3])  # (3*observations,2)

# Define the kernel function
# Some problem with the lower bound. During optimization, the length scale wants to become very small but that is not good --> overfitting
kernel = 1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-5, 1e3)) + WhiteKernel(
    noise_level=1e-1, noise_level_bounds=(1e-10, 1e1)
)
# kernel = DotProduct() + WhiteKernel()  # Other kernel choice
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y)
y_pred = gpr.predict(X_pred)
# print(y_pred)

# Obtain the hyperparameters
hyperparameters = np.array(kernel.theta)
hyperparameters = 10**hyperparameters
print(f"The hyperparameters sigma_f, l_1 and l_2 are respectively: {hyperparameters}")

# Denormalize
# X_train = scaler.inverse_transform(X_train)
# X_train1 = X_train[:,0]
# X_train3 = X_train[:,1]

# X_pred = scaler.inverse_transform(X_pred)
# X_pred1 = X_pred[:,0]
# X_pred3 = X_pred[:,1]

# y = scaler.inverse_transform(y)
# y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))

plot_GPs()
# plot_lr()

# Doorloop een keer alle GP stappen

# ------------------------------------------------------
# Andere dummy dataset gebruikt

# from sklearn.datasets import make_friedman2
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
# X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
# kernel =  1.0 * RBF(1.0)  # length_scale_bounds="fixed"
# gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X, y)

# X_pred1 = create_pred_locations(X[:,0])
# X_pred2 = create_pred_locations(X[:,1])
# X_pred3 = create_pred_locations(X[:,2])
# X_pred4 = create_pred_locations(X[:,3])
# X_pred = np.hstack([X_pred1,X_pred2,X_pred3,X_pred4])  # Shape is (1000,4)
# y_pred = gpr.predict(X_pred)


# Visualizing the predictions
# fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(8,5))
# ax[0,0].plot(X_pred1,y_pred, color="green")
# ax[0,0].scatter(X[:,0],y)

# ax[1,0].plot(X_pred2,y_pred, color="green")
# ax[1,0].scatter(X[:,1],y)

# ax[0,1].plot(X_pred3,y_pred, color="green")
# ax[0,1].scatter(X[:,2],y)

# ax[1,1].plot(X_pred4,y_pred, color="green")
# ax[1,1].scatter(X[:,3],y)
# plt.show()