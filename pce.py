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
        print(f"saved order {order}, rmse: {math.sqrt(error)}")

    return errors, joint

class Kfold:
    def __init__(self, x, y, k):
        self.iter_num = 0
        self.x = x
        self.y = y
        self.len_data = y.shape[0]
        self.k = k
        self.random_idx = np.array(sample(range(self.len_data), self.len_data))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_num < self.k:
            start = int(self.iter_num * self.len_data / self.k)
            end = int((self.iter_num + 1) * self.len_data / self.k)
            print(f"start: {start}, end: {end}")
            idx = self.random_idx[start:end]
            x_test = np.take(self.x, idx, axis=1)
            y_test = np.take(self.y, idx)
            x_train = np.delete(self.x, idx, axis=1)
            y_train = np.delete(self.y, idx)
            self.iter_num += 1
            return x_train, y_train, x_test, y_test
        else:
            raise StopIteration

fieldnames = ["VolumeDensity", "SurfaceAreaDensity", "MeanBreadthDensity", "EulerNumberDensity"]
samples, evaluations  = read_data(fieldnames, " Yield Stress ", "Results/merged_data_clean.csv")

dir = Path("./pce_expansions")

# Create polynomial expansion
orders = range(1, 9)

first_sobols = []
second_sobols = []
total_sobols = []
error_list = []

for samples, evaluations, samples_validate, evaluations_validate in Kfold(samples, evaluations, 10):
    errors, joint = train_pce(samples, evaluations, samples_validate, evaluations_validate, orders, dir)

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

    first_sobols.append(first_sobol)
    second_sobols.append(second_sobols)
    total_sobols.append(total_sobol)
    error_list.append(math.sqrt(errors[best_order_arg]))

errors_mean = np.mean(error_list, axis=0)
errors_std = np.std(error_list, axis=0)

print("average error")
print(f"mean: {errors_mean:.8f}, std: {errors_std:.8f}")

mean_f_sobol = np.mean(first_sobols, axis=0)
std_f_sobol = np.std(first_sobols, axis=0)

print("first order sobol indices for all tests")
for name, mean, std in zip(fieldnames, mean_f_sobol, std_f_sobol):
    print(f"{name:<20} mean: {mean:.8f}, std: {std}")

#second_sobols = np.array(second_sobols)
#mean_s_sobol = np.mean(second_sobols, axis=0)
#for i in range(second_sobols.shape[1]):
#    for j in range(second_sobols.shape[1]):
#        if j >= i:
#            continue
#        print(f"{fieldnames[i]:<20} - {fieldnames[j]:<20} mean: {mean[i,j]:.8f}")

mean_t_sobol = np.mean(total_sobols, axis=0)
std_t_sobol = np.std(total_sobols, axis=0)

print("total sobol indices for all tests")
for name, mean, std in zip(fieldnames, mean_t_sobol, std_t_sobol):
    print(f"{name:<20} mean: {mean:.8f}, std: {std}")
