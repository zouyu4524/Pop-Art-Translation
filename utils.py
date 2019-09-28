import pickle
import numpy as np


# moving average of results
def moving_average(x, window=10):
    n = len(x)
    assert n >= window
    return np.asarray([
        np.mean(x[i:i + window]) for i in range(n - window + 1)
    ])


# calculate median and given percentiles of a sequence
def median_and_percentile(x, axis, lower=10, upper=90):
    assert (lower >= 0 and upper <= 100)
    median = np.median(x, axis)
    low_per = np.percentile(x, lower, axis)
    up_per = np.percentile(x, upper, axis)
    return median, low_per, up_per


def save_results(filename, samples, m, l, u):
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)
        pickle.dump(m, f)
        pickle.dump(l, f)
        pickle.dump(u, f)
