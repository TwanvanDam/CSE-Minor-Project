import numpy as np
import chaospy
import csv

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

# Approximate distribution from data using KDE (assume independent variables)
variables = [chaospy.GaussianKDE(sample) for sample in samples]
joint = chaospy.J(*variables)

