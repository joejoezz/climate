"""
Plot histograms of historical temperature data

"""

from util import get_config
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy.stats import mode
import os
import pdb

# config
config = get_config('../sites.conf')

def get_gev_fit(data):
    """
    Tries GEV fits with several loc parameters, selects the best one
    BUG--doesn't always fit well. Might be related to x-range (needs to be wide enough to catch extremes)
    This also needs to get moved to UTIL eventually 
    """
    md = mode(data)[0][0]
    std = np.std(data)
    # first try with loc=mode
    shape, loc, scale = gev.fit(data, loc=md)
    # if bad try again with mean
    if loc > md+std:
        shape, loc, scale = gev.fit(data, loc=np.mean(data))
    else:
        print('GEV fit with mode')
    # if still bad (ugh), try again with mode - std
    if loc > md+std:
        shape, loc, scale = gev.fit(data, loc=md-std)
    else:
        print('GEV fit with mean')
    if loc > md+std:
        print('GEV fit with c=0')
        shape, loc, scale = gev.fit(data, 0)
    else:
        print('GEV fit with mode minus std deviation')
    return shape, loc, scale


def plot_histogram(site, data1, data2, label1='Data1', label2='Data2', subset_label=None, variable=None):
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
    # print some parameters of data
    print('Ref data: {}'.format(len(data1)))
    print('New data: {}'.format(len(data2)))

    # get histogram parameters
    range_min = np.nanmin(np.hstack((data1, data2)))-np.nanmin(np.hstack((data1, data2))) % 10
    range_max = np.nanmax(np.hstack((data1, data2))) + (10 - np.nanmax(np.hstack((data1, data2))) % 10)
    bins = int(range_max - range_min)

    # compute histograms
    hist1, bin_edges = np.histogram(data1, bins=bins, range=(range_min, range_max), density=True)
    hist2, bin_edges = np.histogram(data2, bins=bins, range=(range_min, range_max), density=True)

    # gev fitting--use function to try a couple times to get a good fit
    shape1, loc1, scale1 = get_gev_fit(data1)
    shape2, loc2, scale2 = get_gev_fit(data2)

    x_gev = np.linspace(range_min, range_max, bins*10+1)
    y1_gev = gev.pdf(x_gev, shape1, loc1, scale1)
    y2_gev = gev.pdf(x_gev, shape2, loc2, scale2)

    # compute POD and FAR of 2.5-sigma event (from reference climate)
    mean1 = gev.mean(shape1, loc=loc1, scale=scale1)
    mean2 = gev.mean(shape2, loc=loc2, scale=scale2)
    std1 = np.sqrt(gev.var(shape1, loc=loc1,scale=scale1))
    std2 = np.sqrt(gev.var(shape2, loc=loc2,scale=scale2))
    # calculate a, b, and c params from Durran 2019
    sig20_thres = np.where((x_gev > mean1 + 2.0 * std1))
    sig25_thres = np.where((x_gev > mean1 + 2.5 * std1))
    sig35_thres = np.where((x_gev > mean1 + 3.5 * std1))
    c_val = np.sum(y1_gev[sig25_thres])
    a_val = np.sum(y2_gev[sig25_thres]) - c_val
    b_val = np.sum(y2_gev[sig20_thres]) - np.sum(y1_gev[sig20_thres]) - a_val
    pod =  a_val/(a_val+b_val)
    far =  c_val/(a_val+c_val)
    print('POD = {}   FAR = {}'.format(pod, far))


    fig = plt.figure()
    fig.set_size_inches(6, 4)

    # stats of gev fit
    #mean1, var1, skew1, kurt1 = gev.stats(shape1, moments='mvsk')

    mu1 = np.mean(data1)
    sigma1 = np.std(data1)
    mu2 = np.mean(data2)
    sigma2 = np.std(data2)


    plt.bar(bin_edges[:-1], hist1, width=1, align='edge', color='blue', alpha=0.5, label=label1)
    plt.bar(bin_edges[:-1], hist2, width=1, align='edge', color='red', alpha=0.5, label=label2)
    plt.plot(x_gev, y1_gev, color='blue')
    plt.plot(x_gev, y2_gev, color='red')
    plt.plot([x_gev[sig20_thres[0][0]], x_gev[sig20_thres[0][0]]], [0,y2_gev[sig20_thres[0][0]]], color='k', lw=1.0)
    plt.plot([x_gev[sig25_thres[0][0]], x_gev[sig25_thres[0][0]]], [0, y2_gev[sig25_thres[0][0]]], color='k', lw=1.0)
    #plt.plot([x_gev[sig35_thres[0][0]], x_gev[sig35_thres[0][0]]], [0, y2_gev[sig35_thres[0][0]]], color='k', lw=1.0)
    plt.plot([mu1, mu1], [0, 1], color='blue', linestyle=':')
    plt.plot([mu2, mu2], [0, 1], color='red', linestyle=':')

    plt.ylabel('PDF')
    plt.xlabel('Temperature')
    plt.ylim(0, np.max((np.max(hist1),np.max(hist2),np.max(y1_gev),np.max(y2_gev)))+0.02)

    plt.legend()
    plt.title('{} {}'.format(site, subset_label))

    plt.savefig('{}{}_{}{}.png'.format(config['PLOT_DIR'], site, subset_label, variable), bbox_inches='tight', dpi=200)
    print('Plotted histogram for {}'.format(site))

    return
