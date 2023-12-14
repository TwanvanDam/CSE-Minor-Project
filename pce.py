import numpy as np
import chaospy
import csv
from sklearn.metrics import mean_squared_error
import math

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

samples, evaluations  = read_data(["x", "y"], "sin(x)", "data/sin_data.csv")
split = int(0.2 * samples.shape[1])
samples_validate = samples[:,0:split]
evaluations_validate = evaluations[0:split]
samples = samples[:,split:]
evaluations = evaluations[split:]

# Approximate distribution from data using KDE (assume independent variables)
variables = [chaospy.GaussianKDE(sample) for sample in samples]
joint = chaospy.J(*variables)

# Create polynomial expansion
orders = range(10)
expansions = [chaospy.generate_expansion(order, joint) for order in orders]

# Fit expansion to data
approxs = [chaospy.fit_regression(expansion, samples, evaluations) for expansion in expansions]

# Evaluate approximations on evaluation data and calculate error
evaluations_val_approxs = [approx(*samples_validate) for approx in approxs]
errors = [mean_squared_error(evaluations_validate, evaluations_val_approx)
         for evaluations_val_approx in evaluations_val_approxs]

# print order and rmse
for order, error in zip(orders, errors):
    print(f"order: {order}, rmse: {math.sqrt(error)}")
