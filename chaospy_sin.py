import numpy as np
import chaospy
import matplotlib.pyplot as plt
import csv

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

# Approximate distribution from data using KDE
joint = chaospy.GaussianKDE(samples)

# Visualize distribution and sample points
grid = np.mgrid[0:np.pi:100j, 0:np.pi:100j]
plt.contourf(grid[0], grid[1], joint.pdf(grid), 30)
plt.scatter(*samples)
plt.show()

# Create polynomial expansion
expansion = chaospy.generate_expansion(2, joint, rule="gram_schmidt")

# Fit expansion to data using least squares
sin_approx = chaospy.fit_regression(expansion, samples, evaluations)

# Visualize approximation
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*samples, sin_approx(*samples), label="approximation")
ax.scatter(*samples, evaluations, label="truth + noice")
fig.legend()
plt.show()
