import numpy as np
import chaospy

def sin_data(parameters):
    x, y = parameters
    z = np.sin(x)
    return z

x = chaospy.Uniform(0, np.pi)
y = chaospy.Uniform(0, np.pi)
joint = chaospy.J(x, y)

x_samples, y_samples = joint.sample(100, rule="random")
print(x_samples, y_samples)

