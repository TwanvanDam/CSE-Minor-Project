import numpy as np
import csv

def get_data(num_points = 1000):
    # create random sample points
    rng = np.random.default_rng(123)
    samples = rng.uniform(0, np.pi, (2,num_points))

    # evaluate sin function for the sample points
    evaluations = np.sin(samples[0])
    return samples, evaluations

if __name__ == "__main__":
    samples, evaluations = get_data()
    with open("data/sin_data.csv", "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["x","y","sin(x)"])
        for sample, evaluation in zip(samples.T, evaluations):
            csv_writer.writerow([*sample, evaluation])
