"""
Plot histograms of historical temperature data

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
import pdb


def plot_histogram(site, data1, data2, label1='Data1', label2='Data2', subset_label=None):
    """
    Plot a normalized histogram of two temperature distributions
    Fit GEV curve to distribution
    :param site: site string
    :param data1: array of data from reference period
    :param data2: array of data from new (warmer climate) period
    :param label1: string label for data1
    :param label2: string label for data2
    :param subset_label: string label for the subset of data (e.g. month/season)
    :return some statistics maybe (TBD)
    """
    # get histogram parameters
    range_min = np.min(np.hstack((data1,data2)))-np.min(np.hstack((data1,data2))) % 10
    range_max = np.max(np.hstack((data1, data2))) + (10 - np.max(np.hstack((data1, data2))) % 10)
    bins = int(range_max - range_min)

    # compute histograms
    hist1, bin_edges = np.histogram(data1, bins=bins, range=(range_min, range_max), density=True)
    hist2, bin_edges = np.histogram(data2, bins=bins, range=(range_min, range_max), density=True)

    # gev fitting
    shape1, loc1, scale1 = gev.fit(data1)
    shape2, loc2, scale2 = gev.fit(data2)

    x_gev = np.linspace(range_min, range_max, bins)
    y1_gev = gev.pdf(x_gev, shape1, loc1, scale1)
    y2_gev = gev.pdf(x_gev, shape2, loc2, scale2)

    fig = plt.figure()
    fig.set_size_inches(6, 4)

    # stats 
    mu1 = np.mean(data1)
    sigma1 = np.std(data1)
    mu2 = np.mean(data2)
    sigma2 = np.std(data2)

    plt.bar(bin_edges[:-1], hist1, width=1, align='edge', color='blue', alpha=0.5, label=label1)
    plt.bar(bin_edges[:-1], hist2, width=1, align='edge', color='red', alpha=0.5, label=label2)
    plt.plot(x_gev, y1_gev, color='blue')
    plt.plot(x_gev, y2_gev, color='red')

    plt.ylabel('PDF')
    plt.xlabel('Temperature')

    plt.legend()
    plt.title('{} {}'.format(site, subset_label))
    plt.savefig('{}_{}.png'.format(site,subset_label), bbox_inches='tight', dpi=200)

    return
