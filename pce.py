import numpy as np
import chaospy
import csv
from sklearn.metrics import mean_squared_error
import math
from random import sample

def read_data(sample_fieldnames, evaluation_fieldname, path):
    samples = [[] for _ in sample_fieldnames]
    evaluations = []
    with open(path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            evaluations.append(row[evaluation_fieldname])
            for element, fieldname in zip(samples, sample_fieldnames):
                element.append(row[fieldname])

    samples = np.array(samples, dtype=np.float64)
    evaluations = np.array(evaluations, dtype=np.float64)
    return samples, evaluations

samples, evaluations  = read_data(["VolumeDensity", "SurfaceAreaDensity", "Volume", "SurfaceArea"], " Yield Stress ", "Results/merged_data.csv")
split = int(0.2 * samples.shape[1])

validate_idx = sample(range(samples.shape[1]), split)
samples_validate = np.take(samples, validate_idx, axis=1)
evaluations_validate = np.take(evaluations, validate_idx)

samples = np.delete(samples, validate_idx, axis=1)
evaluations = np.delete(evaluations, validate_idx)

# Approximate distribution from data using KDE (assume independent variables)
variables = [chaospy.GaussianKDE(sample) for sample in samples]
joint = chaospy.J(*variables)

# Create polynomial expansion
orders = range(1, 10)
expansions = [chaospy.generate_expansion(order, joint) for order in orders]
print("done creating expansions")

# Fit expansion to data
approxs = [chaospy.fit_regression(expansion, samples, evaluations) for expansion in expansions]

# Evaluate approximations on evaluation data and calculate error
evaluations_val_approxs = [approx(*samples_validate) for approx in approxs]
errors = [mean_squared_error(evaluations_validate, evaluations_val_approx)
         for evaluations_val_approx in evaluations_val_approxs]

# print order and rmse
for order, error in zip(orders, errors):
    print(f"order: {order}, rmse: {math.sqrt(error)}")

# Get lowest error order
best_order_arg = np.argmin(errors)
best_order = orders[best_order_arg]
best_approx = approxs[best_order_arg]
print(f"\nSelected order {best_order} to calculate Sobol indices")

# Calculate Sobol indices
print("First order Sobol indices")
print(chaospy.Sens_m(best_approx, joint))
print("Second order Sobol indices")
print(chaospy.Sens_m2(best_approx, joint))
