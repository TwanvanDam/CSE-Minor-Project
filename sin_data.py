import numpy as np

def get_data():
    # create random sample points
    rng = np.random.default_rng(123)
    samples = rng.uniform(0, np.pi, (2,100))

    # evaluate sin function for the sample points
    evaluations = np.sin(samples[0])
    return samples, evaluations

get_data()
