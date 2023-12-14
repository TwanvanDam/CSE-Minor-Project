import numpy as np
import chaospy
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_squared_error

noise_level = 0.1

def read_data():
    x = []
    y = []
    z = []
    with open("data/sin_data.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            x.append(row['x'])
            y.append(row['y'])
            z.append(row['sin(x)'])
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    z = np.array(z, dtype=np.float64)
    samples = np.array([x, y])
    return samples, z

# Read data from file
samples, evaluations = read_data()

# Add noise to data
truth = evaluations
rng = np.random.default_rng(1234)
evaluations = evaluations + noise_level * rng.uniform(-1, 1, size=evaluations.shape)

# Approximate distribution from data using KDE (assuming sample variables are independent)
x = chaospy.GaussianKDE(samples[0])
y = chaospy.GaussianKDE(samples[1])
joint = chaospy.J(x, y)

# Visualize distribution and sample points
grid = np.mgrid[0:np.pi:100j, 0:np.pi:100j]
plt.contourf(grid[0], grid[1], joint.pdf(grid), 30)
plt.scatter(*samples)
plt.show()

# Create polynomial expansion
expansion = chaospy.generate_expansion(2, joint, rule="gram_schmidt")

# Fit expansion to data using least squares
sin_approx = chaospy.fit_regression(expansion, samples, evaluations)

# Calculate approximate evaluations
evaluations_approx = sin_approx(*samples)

# Get sobol indices
print("First order Sobol indices")
print(chaospy.Sens_m(sin_approx, joint))
print("Second order Sobol indices")
print(chaospy.Sens_m2(sin_approx, joint))

# Calculate error 
error = np.sqrt(mean_squared_error(truth, evaluations_approx))
print(f"rmse: {error}")

print("expansion:")
print(sin_approx.round(4))

# Visualize approximation
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*samples, evaluations_approx, label="approximation")
ax.scatter(*samples, evaluations, label="truth + noice")
fig.legend()
plt.show()
