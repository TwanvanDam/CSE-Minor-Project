# GP regression on sine
# Joey Cheung, Nov 2023

# Create dataset for sine wave surface, possibly add some noise
# Train the GP model
# Look at the length_scales

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import random

def create_groundtruth(xend=4.0, npoints=5, noise=0.01):
    '''
    Create a sine surface on a rectangluar domain
    Parameters: x is 1D array
    Outputs: the 2D input array, the sine value (a surface)
    '''
    x1 = np.linspace(0,xend,npoints)
    x1 = x1.reshape(-1,1)
    noise_vector = np.sqrt(noise)*np.random.normal(size=(npoints,))  # Normal(0, noise)
    noise_vector = noise_vector.reshape(-1,1)
    y = np.sin(0.5*np.pi*x1)  +  noise_vector  # So a full period is from 0 to 4
    y1 = y

    # Create a new array where all the elements of the original array are repeated
    # E.g. [1,2] becomes [1,1,1,2,2,2]
    y = np.tile(y,(npoints,1))
    y2 = np.full((npoints,1), y[0])

    # Stack x1 multiple times (npoints) upon each other
    X = np.tile(x1,(npoints,1))
    x2 = x1
    x2 = np.repeat(x2,npoints)  
    x2 = x2.reshape(-1,1)
    X_final = np.hstack([X,x2])
    # print(X_final)

    return X_final, y, x1, x1, y1, y2

def create_pred_locations(input_data, points, zeros=False, offset=0.2):
    min1 = np.min(input_data)
    max1 = np.max(input_data)

    if not zeros:
        X_predi = np.linspace(min1,max1,points) + offset
        X_predi = X_predi.reshape(-1,1)
    if zeros:
        X_predi = np.zeros(points)
        X_predi = X_predi.reshape(-1,1)

    return X_predi

def plot_3D():
    fig2 = plt.figure()
    # axnew = fig2.add_subplot(projection='3d')
    axnew = plt.axes(projection='3d')
    axnew.scatter(X_train[:,0], X_train[:,1], y.flatten(), label="The ground truth")
    axnew.set_xlabel('x')
    axnew.set_ylabel('y')
    axnew.set_zlabel('z')
    axnew.legend()
    plt.show()

def reshape_and_round(array, decimals=5):
    '''
    Input: a 1D array 
    Output: a 2D array
    '''
    array = array.reshape(-1,1)
    array = np.round(array,decimals)

    return array
    
def plot_GPs(two_d=True, one_d=False, one_d_first=True, one_d_second=False):
    global y_pred
    global y2

    if two_d:
        print(y_pred[:npoints])
        # Scatterplots with the regression line
        fig, ax = plt.subplots(ncols=3, figsize=(13,5))
        ax[0].scatter(X_train[:npoints,0],y[:npoints,0])
        ax[0].plot(X_pred1_nonzeros,y_pred[:npoints], color="orange")
        ax[0].set_xlabel("x-axis (first feature axis)")
        ax[0].set_ylabel("Target y")

        ax[1].scatter(x2,y2)
        ax[1].plot(X_pred3_nonzeros,y_pred[2*npoints:], color="orange")
        ax[1].set_xlabel("y-axis (second feature axis)")
        ax[1].set_ylabel("Target y")
        ax[1].set_ylim([-y[1],y[1]])

        # y_pred_at_train_locations = gpr.predict(X_train)
        # ax[2].scatter(y,y_pred_at_train_locations)
        # ax[2].set_xlabel("y")
        # ax[2].set_ylabel("y_pred")

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

    elif one_d:
        if one_d_first:
            # Scatterplots with the regression line
            fig, ax = plt.subplots(ncols=2, figsize=(8,5))
            ax[0].scatter(x1,y1)
            ax[0].plot(X_pred1_nonzeros,y_pred, color="orange")
            ax[0].set_xlabel("x-axis (first feature axis)")
            ax[0].set_ylabel("Target y")
            ax[0].set_ylim([-1,1])

            y_pred_at_train_locations = gpr.predict(x1)
            ax[1].scatter(y1,y_pred_at_train_locations)
            ax[1].set_xlabel("y")
            ax[1].set_ylabel("y_pred")

            plt.tight_layout()
            plt.show()

        elif one_d_second:
            y2 = reshape_and_round(y2, decimals=2)
            Error = np.abs(y2.flatten()-y_pred)
            print(f"Error1 is {Error[0]}")
            # print(y_pred)
            # print(y2)

            # Scatterplots with the regression line
            fig, ax = plt.subplots(ncols=2, figsize=(8,5))
            ax[0].scatter(x2,y2)
            ax[0].plot(x2,y_pred, color="orange")
            ax[0].set_xlabel("x-axis (first feature axis)")
            ax[0].set_ylabel("Target y")
            ax[0].set_ylim([-1,1])
            # print(X_pred3_nonzeros)
            # print(y2)
            # print(y_pred)

            y_pred_at_train_locations = gpr.predict(x2)
            Error = np.abs(y2.flatten()-y_pred_at_train_locations)
            print(f"Error2 is {Error[0]}")
            
            y_pred_at_train_locations = reshape_and_round(y_pred_at_train_locations)
            # print(y2)
            # print(y_pred_at_train_locations)
            ax[1].scatter(y2,y_pred_at_train_locations)
            ax[1].set_xlabel("y")
            ax[1].set_ylabel("y_pred")

            plt.tight_layout()
            plt.show()

# =========================================================================
# Start main code

npoints = 10
noise = 0.0001  # noise heeft geen invloed op de error
X_train, y, x1, x2, y1, y2 = create_groundtruth(xend=4.0, npoints=npoints, noise=noise)
print(X_train.shape)
print(y.shape)

# plot_3D()

# Create prediction locations
X_pred1_zeros = create_pred_locations(X_train[:,0], points=npoints, zeros=True)
X_pred1_nonzeros = create_pred_locations(X_train[:,0], points=npoints, zeros=False)
X_pred1 = np.vstack([X_pred1_nonzeros, X_pred1_nonzeros, X_pred1_zeros])
print(X_pred1_nonzeros.shape)
# print(X_train1)

X_pred3_zeros = create_pred_locations(X_train[:,1], points=npoints, zeros=True, offset=0.2)
X_pred3_nonzeros = create_pred_locations(X_train[:,1], points=npoints, zeros=False, offset=0.2)
X_pred3 = np.vstack([X_pred3_zeros, X_pred3_nonzeros, X_pred3_nonzeros])
X_pred = np.hstack([X_pred1,X_pred3])  # (3*observations,2)

one_d_first = False
one_d_second = False
two_d = True

if one_d_first:
    # Define the kernel function
    kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-5, 1e3)) + WhiteKernel(
        noise_level=1e-1, noise_level_bounds=(1e-2, 1e1)
    )

    # You have to pass 2D arrays for fit and predict functions!
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x1,y1)
    y_pred = gpr.predict(X_pred1_nonzeros)

    # Obtain the hyperparameters
    hyperparameters = np.array(kernel.theta)
    hyperparameters = 10**hyperparameters
    print(f"The hyperparameters sigma_f, l1 and noise level are respectively: {hyperparameters}")
    # print(f"The hyperparameter names are {kernel.hyperparameters}")

    plot_GPs(two_d=False,one_d=True, one_d_first=True, one_d_second=False)
    
elif one_d_second:
    # Define the kernel function
    kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-5, 1e10)) + WhiteKernel(
        noise_level=1e-1, noise_level_bounds=(1e-2, 1e1)
    )

    # You have to pass 2D arrays for fit and predict functions!
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x2,y2)
    y_pred = gpr.predict(X_pred3_nonzeros)

    # Obtain the hyperparameters
    hyperparameters = np.array(kernel.theta)
    hyperparameters = 10**hyperparameters
    print(f"The hyperparameters sigma_f, l2 and noise level are respectively: {hyperparameters}")
    # print(f"The hyperparameter names are {kernel.hyperparameters}")

    plot_GPs(two_d=False,one_d=True, one_d_first=False, one_d_second=True)

elif two_d:
    # Define the kernel function
    kernel = 1.0 * RBF(length_scale=[1e0, 1e0], length_scale_bounds=(1e-5, 1e10)) + WhiteKernel(
        noise_level=1e-1, noise_level_bounds=(1e-10, 1e1)
    )

    # You have to pass 2D arrays for fit and predict functions!
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0, n_restarts_optimizer=0).fit(X_train,y)
    y_pred = gpr.predict(X_pred)

    # Obtain the hyperparameters
    hyperparameters = np.array(kernel.theta)
    hyperparameters = 10**hyperparameters
    print(f"The hyperparameters sigma_f, l2 and noise level are respectively: {hyperparameters}\n")
    print(f"The hyperparameter names are {kernel.hyperparameters}\n")
    print(gpr.kernel_)

    plot_GPs(two_d=True,one_d=False, one_d_first=False, one_d_second=False)

# print(X_train); print(y)
# plot_3D()
# print(X_pred)