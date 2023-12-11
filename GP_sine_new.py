import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
# There is the standard random library
# There is the Numpy random library


class ground_truth:
    def __init__(self, noise):
        self.noise = noise

    def truth(self, x):
        '''
        Input: x (2D array)
        '''
        y = np.sin(x)
        return y

    def observe(self, x):
        '''
        Input: x (1D array)

        Note that the observation data is generated in a different way than the ground_truth data
        '''
        if (len(x) != npoints_obs):
            raise Exception("x array dimensions don't correspond to npoints_obs!")
        
        random_numbers_array = np.sqrt(self.noise)*np.random.normal(loc=0, scale=1, size=(npoints_obs,))  # ~ N(0,noise)
        y = np.sin(x) + random_numbers_array
        return y


# ===================== Start main code =====================

noise = 0.1
true_func = ground_truth(noise=noise)
npoints_truth = 30  # Number of ground_truth points in 1D

npoints_obs = 25  # Number of observations in 1D
npoints_pred = 50  # Number of predictions in 1D
begin = 0; end = 10

# Create the ground truth points
x_truth1 = np.linspace(begin, end, npoints_truth)
x_truth2 = np.linspace(begin, end, npoints_truth)
xx_truth1, xx_truth2 = np.meshgrid(x_truth1, x_truth2)
y_truth = true_func.truth(xx_truth1)

# Create random observation points (Adapted Marijns method: see his file)
rng = np.random.default_rng(123)
x_obs = rng.uniform(begin, end, (2, npoints_obs))
x_obs1 = x_obs[0]
x_obs2 = x_obs[1]
y_obs_1d = true_func.observe(x_obs1)

x_obs1 = x_obs1.reshape(-1, 1)
x_obs2 = x_obs2.reshape(-1, 1)
x_obs12 = np.hstack([x_obs1, x_obs2])

# Create random prediction locations
rng = np.random.default_rng(456)
x_pred = rng.uniform(begin, end, (2, npoints_pred))
x_pred1 = x_pred[0]
x_pred2 = x_pred[1]

# Reshape the 2D arrays into 1D arrays 
# --> needed for GP predict function
xx_pred1, xx_pred2 = np.meshgrid(x_pred1, x_pred2)
xx_pred1_L = xx_pred1.reshape(-1, 1)
print(xx_pred1_L.shape)
xx_pred2_L = xx_pred2.reshape(-1, 1)
xx_pred_L = np.hstack([xx_pred1_L, xx_pred2_L])

kernel = 1.0 * RBF(length_scale=[1, 1], length_scale_bounds=(1e-5, 1e10)) + WhiteKernel(
        noise_level=1e-1, noise_level_bounds=(1e-2, 1e1)
    )

gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=4).fit(x_obs12, y_obs_1d)
y_pred, sig = gpr.predict(xx_pred_L, return_std=True)

# Plot the results
fig1 = plt.figure()
axnew = fig1.add_subplot(projection='3d')
# surf = axnew.plot_surface(xx_truth1, xx_truth2, y_truth,
#                           color="blue", linewidth=0,
#                           label="Ground truth", antialiased=False)  # Plot the surface
# axnew.scatter(xx_truth1, xx_truth2, y_truth, color="blue",
#               label="Ground truth")                                 # Scatter plot the ground truth
axnew.scatter(x_obs1, x_obs2, y_obs_1d, color="green", label="Observations")
axnew.scatter(xx_pred1, xx_pred2, y_pred, color="orange", label="Predictions")
plt.title(
    (
        f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
        f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}"
    ),
    fontsize=8,
    )
fig1_manager = plt.get_current_fig_manager()
fig1_manager.window.geometry("+100+100")  # Set the spawn location for the first figure

# Make a second plot without the
fig2 = plt.figure()
axnew = fig2.add_subplot(projection='3d')
axnew = fig2.add_subplot(projection='3d')
surf = axnew.plot_surface(xx_truth1, xx_truth2, y_truth,
                          color="blue", linewidth=0,
                          label="Ground truth", antialiased=False)  # Plot the surface
# axnew.scatter(xx_truth1, xx_truth2, y_truth, color="blue",
#               label="Ground truth")                                 # Scatter plot the ground truth
axnew.scatter(x_obs1, x_obs2, y_obs_1d, color="green", label="Observations")
axnew.scatter(xx_pred1, xx_pred2, y_pred, color="orange", label="Predictions")
fig2_manager = plt.get_current_fig_manager()
fig2_manager.window.geometry("+1000+100")  # Set the spawn location for the second figure

plt.legend()
plt.show()
