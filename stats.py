#!/usr/bin/env python3

"""Library for Stats """

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def ht_diff_proportions(s1, s2):
    """Test the null hypotheis that two samples are from the same Bernoulli distribution

    Input:
        s1, s2: 2 numpy arrays/pandas series for samples

    Returns:
        pvalue

    Note:
        Assume binomial approximation to normal; may not valid for skewed and small amount of data

    """

    N1 = float(s1.shape[0])
    N2 = float(s2.shape[0])
    X1 = s1.sum()
    X2 = s2.sum()
    p1 = X1 / N1
    p2 = X2 / N2
    SE = np.sqrt(p1 * (1 - p1) / N1 + p2 * (1 - p2) / N2)
    if SE != 0:
        Z = np.abs((p2 - p1)) / SE
        pvalue = stats.norm.cdf(-Z) * 2
    else:
        print("SE is 0!")
        pvalue = None

    return pvalue

def plot_cdf(data, color='b'):
    """Plot CDF of a list of data """

    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, yvals, c=color)
