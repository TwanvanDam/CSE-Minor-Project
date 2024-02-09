import numpy as np
import chaospy
import csv
from sklearn.metrics import mean_squared_error
import math
from random import sample
from pathlib import Path
from multiprocessing import Pool

data_file = "Results/merged_data_offset_final.csv"  # File with the data
fieldnames = ["VolumeDensity", "SurfaceAreaDensity", "MeanBreadthDensity", "EulerNumberDensity"]  # fields to use from the datafile
result_fieldname = "Yield Stress"  # field with the result

orders = range(1, 9)  # orders to use for the expansions
num_folds = 20  # numer of folds to use for k-fold cross validation

num_threads = 10  # number of threads for multiprocessing
dir = Path("./pce_expansions")  # empty directory to save the expansions to

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
    y_preds = []

    # Approximate distribution from data using KDE (assume independent variables)
    variables = [chaospy.GaussianKDE(x) for x in x_train]
    joint = chaospy.J(*variables)
    for order in orders:
        # Create expansion
        expansion = chaospy.generate_expansion(order, joint)

        # Fit expansion to data and sample expansion on validation points
        approx = chaospy.fit_regression(expansion, x_train, y_train)
        evaluations_val_approx = approx(*x_test)
        y_preds.append(evaluations_val_approx)

        # Calculate error
        error = mean_squared_error(y_test, evaluations_val_approx)
        errors.append(error)

        # Save expansion to file
        chaospy.save(save_dir / f"pce_unfitted_{order}", expansion)
        chaospy.save(save_dir / f"pce_fitted_{order}", approx)
        # print(f"saved order {order}, rmse: {math.sqrt(error)}")

    return errors, joint, y_preds, y_test

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

samples, evaluations  = read_data(fieldnames, result_fieldname, data_file)

# Create polynomial expansion
def process_fold(inputs):
    samples, evaluations, samples_validate, evaluations_validate = inputs
    save_dir = dir / str(hash(samples_validate.data.tobytes()) + hash(evaluations_validate.tobytes()))
    save_dir.mkdir()
    errors, joint, y_test_pred, y_test = train_pce(samples, evaluations, samples_validate, evaluations_validate, orders, save_dir)

    # Get lowest error order
    best_order_arg = np.argmin(errors)
    best_order = orders[best_order_arg]
    # print(save_dir / f"pce_fitted_{best_order}")
    best_approx = chaospy.load(save_dir / f"pce_fitted_{best_order}", allow_pickle=True)
    # print(f"\nSelected order {best_order} to calculate Sobol indices")

    # Calculate Sobol indices
    first_sobol = chaospy.Sens_m(best_approx, joint)
    # print("First order Sobol indices")
    # for name, sobol in zip(fieldnames, first_sobol):
    #     print(f"{name:<20}: {sobol:.8f}")
    # print("Second order Sobol indices")
    # second_sobol = chaospy.Sens_m2(best_approx, joint)
    second_sobol = [0]
    # print(second_sobol)
    total_sobol = chaospy.Sens_t(best_approx, joint)
    # print("Total sobol indices")
    # for name, sobol in zip(fieldnames, total_sobol):
    #     print(f"{name:<20}: {sobol:.8f}")
    rmse = math.sqrt(errors[best_order_arg])
    return first_sobol, second_sobol, total_sobol, rmse, y_test_pred[best_order_arg], y_test

with Pool(num_threads) as p:
    k_fold = Kfold(samples, evaluations, num_folds)
    results = p.imap_unordered(process_fold, k_fold)

    first_sobols = []
    second_sobols = []
    total_sobols = []
    error_list = []
    y_preds = np.array([])
    y_truths = np.array([])

    for result in results:
        first_sobol, second_sobol, total_sobol, rmse, y_test_pred, y_test = result
        first_sobols.append(first_sobol)
        second_sobols.append(second_sobol)
        total_sobols.append(total_sobol)
        error_list.append(rmse)
        y_preds = np.append(y_preds, y_test_pred)
        y_truths = np.append(y_truths, y_test)

# Write results to files
with open(dir/"total_sobols.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(fieldnames)
    for row in total_sobols:
        writer.writerow(row)

with open(dir/"first_sobols.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(fieldnames)
    for row in first_sobols:
        writer.writerow(row)

with open(dir/"yield_predictions.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["y_pred", "y_truth"])
    for y_pred, y_truth in zip(y_preds, y_truths):
        writer.writerow([y_pred, y_truth])

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
