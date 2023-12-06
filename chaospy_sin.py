import numpy as np
import chaospy
import matplotlib.pyplot as plt

def sin_data(parameters):
    x, y = parameters
    z = np.sin(x)
    return z

# Set distribution for input variables
x = chaospy.Uniform(0, np.pi)
y = chaospy.Uniform(0, np.pi)
joint = chaospy.J(x, y)

# Sample joint distribution and sample function
samples = joint.sample(100, rule="random")
evaluations = sin_data(samples)

# Visualize distribution and sample points
grid = np.mgrid[0:np.pi:100j, 0:np.pi:100j]
plt.contourf(grid[0], grid[1], joint.pdf(grid), 30)
plt.scatter(*samples)
plt.show()

# Create polynomial expansion
expansion = chaospy.generate_expansion(2, joint)

# Fit expansion to data
sin_approx = chaospy.fit_regression(expansion, samples, evaluations)

