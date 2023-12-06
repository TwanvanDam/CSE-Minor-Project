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

# Sample joint distribution
x_samples, y_samples = joint.sample(100, rule="random")
print(x_samples, y_samples)

# Visualize distribution and sample points
grid = np.mgrid[0:np.pi:100j, 0:np.pi:100j]
plt.contourf(grid[0], grid[1], joint.pdf(grid), 30)
plt.scatter(x_samples, y_samples)
plt.show()

# Create polynomial expansion
expansion = chaospy.generate_expansion(1, joint)
print(expansion.round(2))
