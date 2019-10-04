"""
Plot of POD vs. FAR
Use fixed sigma-threshold, use fixed parameters of distributions and
"""

from util import get_config
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy.stats import mode
import pandas as pd
import os
import pdb

# config
config = get_config('../sites.conf')

def get_gev_fit(data):
    """
    Tries GEV fits with several loc parameters, selects the best one
    MIGHT BE DIFFERENT THAN OTHER SCRIPT--BE CAREFUL
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

def get_pod_far_curve(x_gev, y1_gev, y2_gev, y1_mean, y1_std, sigma, sig_thresh = 2.0):
    """

    :param x_gev: x-values of GEV
    :param y1_gev: GEV curve of reference period
    :param y2_gev: GEV curve of new period
    :param y1_mean: mean of reference distribution
    :param y2_mean = standard deviation of reference distribution
    :param sigma: sigma value to test for (no lower than sig_thres)
    :param sig_thresh: sigma threshold for extreme event definition
    :return: pod and far arrays of length sigma_array
    """
    # calculate a, b, and c params from Durran 2019
    sig_thres = np.where((x_gev > y1_mean + sig_thresh * y1_std))
    sig_new_thres = np.where((x_gev > y1_mean + sigma * y1_std))
    c_val = np.sum(y1_gev[sig_new_thres])
    a_val = np.sum(y2_gev[sig_new_thres]) - c_val
    b_val = np.sum(y2_gev[sig_thres]) - np.sum(y1_gev[sig_thres]) - a_val
    pod = a_val/(a_val+b_val)
    far = c_val/(a_val+c_val)
    pod = pod
    far = far
    return pod, far



def plot_pod_vs_far(site, data1_hi, data1_lo, subset_label=None):
    """
    Compare the POD and FAR from 2-sigma to 4-sigma for high and low
    :param site: site string
    :param data1_hi: array of high temp data from reference period
    :param data1_lo: array of low temp data from reference period
    :param subset_label: string label for the subset of data (e.g. month/season)
    """


    # get histogram parameters
    range_min_hi = np.nanmin(np.hstack((data1_hi)))-np.nanmin(np.hstack((data1_hi))) % 10
    range_max_hi = np.nanmax(np.hstack((data1_hi))) + (10 - np.nanmax(np.hstack((data1_hi))) % 10 +
                                                                 20)
    bins_hi = int(range_max_hi - range_min_hi)
    range_min_lo = np.nanmin(np.hstack((data1_lo)))-np.nanmin(np.hstack((data1_lo))) % 10
    range_max_lo = np.nanmax(np.hstack((data1_lo))) + (10 - np.nanmax(np.hstack((data1_lo))) % 10) + 10
    bins_lo = int(range_max_lo - range_min_lo)

    # gev fitting--use function to try a couple times to get a good fit
    shape1_hi, loc1_hi, scale1_hi = get_gev_fit(data1_hi)

    x_gev_hi = np.linspace(range_min_hi, range_max_hi, bins_hi*10+1)
    y1_gev_hi = gev.pdf(x_gev_hi, shape1_hi, loc1_hi, scale1_hi)

    sigma_array = np.linspace(2, 5, 7) # do 30 for longer one
    pod_hi = np.zeros(len(sigma_array))
    far_hi = np.zeros(len(sigma_array))

    # compute POD and FAR of 2.5-sigma event (from reference climate)
    mean1_hi = gev.mean(shape1_hi, loc=loc1_hi, scale=scale1_hi)
    std1_hi = np.sqrt(gev.var(shape1_hi, loc=loc1_hi,scale=scale1_hi))

    # same for low
    shape1_lo, loc1_lo, scale1_lo = get_gev_fit(data1_lo)

    x_gev_lo = np.linspace(range_min_lo, range_max_lo, bins_lo*10+1)
    y1_gev_lo = gev.pdf(x_gev_lo, shape1_lo, loc1_lo, scale1_lo)

    pod_lo = np.zeros(len(sigma_array))
    far_lo = np.zeros(len(sigma_array))

    # compute POD and FAR of 2.5-sigma event (from reference climate)
    mean1_lo = gev.mean(shape1_lo, loc=loc1_lo, scale=scale1_lo)
    std1_lo = np.sqrt(gev.var(shape1_lo, loc=loc1_lo,scale=scale1_lo))

    #define dataframes of what we are pulling
    warming_levels = np.linspace(0.1, 1, 10)
    pod_hi = pd.DataFrame(index=sigma_array, columns=warming_levels)
    pod_lo = pd.DataFrame(index=sigma_array, columns=warming_levels)
    far_hi = pd.DataFrame(index=sigma_array, columns=warming_levels)
    far_lo = pd.DataFrame(index=sigma_array, columns=warming_levels)
    far_lo = far_lo.fillna(0)
    y_curves_hi = pd.DataFrame(index=warming_levels, columns=x_gev_hi)
    y_curves_lo = pd.DataFrame(index=warming_levels, columns=x_gev_lo)
    hi_locs = np.zeros(len(warming_levels))
    lo_locs = np.zeros(len(warming_levels))

    for i,level in enumerate(warming_levels):
        loc1_hi_new = loc1_hi + level*std1_hi
        hi_locs[i] = loc1_hi_new
        y2_gev_hi = gev.pdf(x_gev_hi, shape1_hi, loc1_hi_new, scale1_hi)
        y_curves_hi.loc[level] = y2_gev_hi
        for sigma in sigma_array:
            pod, far = get_pod_far_curve(x_gev_hi, y1_gev_hi, y2_gev_hi, mean1_hi, std1_hi, sigma, sig_thresh=2.0)
            pod_hi[level][sigma] = pod * 100.
            far_hi[level][sigma] = far * 100.

        loc1_lo_new = loc1_lo + level*std1_lo
        lo_locs[i] = loc1_lo_new
        y2_gev_lo = gev.pdf(x_gev_lo, shape1_lo, loc1_lo_new, scale1_lo)
        y_curves_lo.loc[level] = y2_gev_lo
        for sigma in sigma_array:
            pod, far = get_pod_far_curve(x_gev_lo, y1_gev_lo, y2_gev_lo, mean1_lo, std1_lo, sigma, sig_thresh=2.0)
            pod_lo[level][sigma] = pod * 100.
            far_lo[level][sigma] = far * 100.

    # POD vs FAR plot
    # labels
    labels = ['2.0$\sigma$', '2.5$\sigma$', '3.0$\sigma$', '3.5$\sigma$', '4.0$\sigma$', '4.5$\sigma$', '5.0$\sigma$']

    # another way of plotting POD vs FAR
    fig = plt.figure()
    fig.set_size_inches(6, 4)

    for i,level in enumerate(warming_levels):
        plt.plot(far_hi[level], pod_hi[level], color=plt.cm.Reds(level-0.05), marker='o', lw=4,
                 label='$\mu$+{}$\sigma$'.format(np.around(level,1)))

        for j,ind in enumerate(far_hi.index):
            if i == 2:
                plt.text(far_hi[level][ind]+2, pod_hi[level][ind]-4, labels[j], color='black', fontsize=8)

    plt.ylabel('POD (%)')
    plt.xlabel('FAR (%)')
    plt.ylim(0, 100)
    plt.xlim(0, 100)

    plt.legend(fontsize=5, loc='upper left')
    plt.title('POD vs FAR {} {} (2.0-$\sigma$ threshold)'.format(site, subset_label))

    plt.savefig('{}pod_vs_far_warming_hi_{}_{}.png'.format(config['PLOT_DIR'], site, subset_label), bbox_inches='tight', dpi=200)
    print('Plotted pod_vs_far for {}'.format(site))

    # same for low --------------------
    fig = plt.figure()
    fig.set_size_inches(6, 4)

    for i,level in enumerate(warming_levels):
        plt.plot(far_lo[level], pod_lo[level], color=plt.cm.Blues(level-0.05), marker='o', lw=4,
                 label='$\mu$+{}$\sigma$'.format(np.around(level,1)))

        for j,ind in enumerate(far_lo.index):
            if i == 2:
                plt.text(far_lo[level][ind]+2, pod_lo[level][ind]-4, labels[j], color='black', fontsize=8)

    plt.ylabel('POD (%)')
    plt.xlabel('FAR (%)')
    plt.ylim(0, 100)
    plt.xlim(0, 100)

    plt.legend(fontsize=5, loc='upper right')
    plt.title('POD vs FAR {} {} (2.0-$\sigma$ threshold)'.format(site, subset_label))

    plt.savefig('{}pod_vs_far_warming_lo_{}_{}.png'.format(config['PLOT_DIR'], site, subset_label), bbox_inches='tight', dpi=200)
    print('Plotted pod_vs_far for {}'.format(site))

    pdb.set_trace()

    # plot the different temperature curves... --------------------------------
    fig = plt.figure()
    fig.set_size_inches(6, 4)

    #plot mean
    plt.plot(x_gev_hi, y1_gev_hi, color='black',
             label='1950-1979')

    for level in warming_levels:
        plt.plot(x_gev_hi, y_curves_hi.loc[level].values, color=plt.cm.Reds(level-0.05),
                 label='$\mu$+{}$\sigma$'.format(np.around(level,1)))

    plt.plot([mean1_hi, mean1_hi], [0, 1], color='red', linestyle=':')
    plt.plot([mean1_hi+std1_hi*2, mean1_hi+std1_hi*2], [0, 1], color='black', linestyle=':')
    plt.plot([mean1_hi+std1_hi*2.5, mean1_hi+std1_hi*2.5], [0, 1], color='black', linestyle=':')

    plt.ylabel('PDF')
    plt.xlabel('Temperature')
    plt.ylim(0, np.max(y_curves_hi.values)+0.02)


    plt.legend(fontsize=5)
    plt.title('{} {}'.format(site, subset_label))

    plt.savefig('{}shift_mean_hi_{}_{}.png'.format(config['PLOT_DIR'], site, subset_label), bbox_inches='tight', dpi=200)
    plt.close()

    # low temp
    fig = plt.figure()
    fig.set_size_inches(6, 4)

    plt.plot(x_gev_lo, y1_gev_lo, color='black',
             label='1950-1979')

    for level in warming_levels:
        plt.plot(x_gev_lo, y_curves_lo.loc[level].values, color=plt.cm.Blues(level-0.05),
                 label='$\mu$+{}$\sigma$'.format(np.around(level,1)))

    plt.plot([mean1_lo, mean1_lo], [0, 1], color='blue', linestyle=':')
    plt.plot([mean1_lo+std1_lo*2, mean1_lo+std1_lo*2], [0, 1], color='black', linestyle=':')
    plt.plot([mean1_lo+std1_lo*2.5, mean1_lo+std1_lo*2.5], [0, 1], color='black', linestyle=':')

    plt.ylabel('PDF')
    plt.xlabel('Temperature')
    plt.ylim(0, np.max(y_curves_lo.values)+0.02)

    plt.legend(fontsize=5)
    plt.title('{} {}'.format(site, subset_label))

    plt.savefig('{}shift_mean_lo_{}_{}.png'.format(config['PLOT_DIR'], site, subset_label), bbox_inches='tight', dpi=200)

    pdb.set_trace()


    return
