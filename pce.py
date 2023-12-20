import numpy as np
import chaospy
import csv
from sklearn.metrics import mean_squared_error
import math
from random import sample
from pathlib import Path

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


def train_pce(x_train, y_train, x_test, y_test, orders, save_dir):
    errors = []

    # Approximate distribution from data using KDE (assume independent variables)
    variables = [chaospy.GaussianKDE(x) for x in x_train]
    joint = chaospy.J(*variables)
    for order in orders:
        # Create expansion
        expansion = chaospy.generate_expansion(order, joint)

        # Fit expansion to data and sample expansion on validation points
        approx = chaospy.fit_regression(expansion, x_train, y_train)
        evaluations_val_approx = approx(*x_test)

        # Calculate error
        error = mean_squared_error(y_test, evaluations_val_approx)
        errors.append(error)

        # Save expansion to file
        chaospy.save(save_dir / f"pce_unfitted_{order}", expansion)
        chaospy.save(save_dir / f"pce_fitted_{order}", approx)
        print(f"saved order {order}")

    return errors, joint

fieldnames = ["VolumeDensity", "SurfaceAreaDensity", "MeanBreadthDensity", "EulerNumberDensity"]
samples, evaluations  = read_data(fieldnames, " Yield Stress ", "Results/merged_data.csv")
split = int(0.3 * samples.shape[1])

validate_idx = sample(range(samples.shape[1]), split)
samples_validate = np.take(samples, validate_idx, axis=1)
evaluations_validate = np.take(evaluations, validate_idx)

samples = np.delete(samples, validate_idx, axis=1)
evaluations = np.delete(evaluations, validate_idx)

dir = Path("./pce_expansions")

# Create polynomial expansion
orders = range(1, 5)

errors, joint = train_pce(samples, evaluations, samples_validate, evaluations_validate, orders, dir)

# print order and rmse
for order, error in zip(orders, errors):
    print(f"order: {order}, rmse: {math.sqrt(error)}")

# Get lowest error order
best_order_arg = np.argmin(errors)
best_order = orders[best_order_arg]
print(dir / f"pce_fitted_{best_order}")
best_approx = chaospy.load(dir / f"pce_fitted_{best_order}", allow_pickle=True)
print(f"\nSelected order {best_order} to calculate Sobol indices")

# Calculate Sobol indices
first_sobol = chaospy.Sens_m(best_approx, joint)
print("First order Sobol indices")
for name, sobol in zip(fieldnames, first_sobol):
          print(f"{name:<20}: {sobol:.8f}")
print("Second order Sobol indices")
print(chaospy.Sens_m2(best_approx, joint))
total_sobol = chaospy.Sens_t(best_approx, joint)
print("Total sobol indices")
for name, sobol in zip(fieldnames, total_sobol):
    print(f"{name:<20}: {sobol:.8f}")
