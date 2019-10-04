"""
Plot of POD vs. FAR for sigma thresholds from 2.0 to 4.0. For high and low.
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


def plot_pod_vs_far(site, data1_hi, data2_hi, data1_lo ,data2_lo, subset_label=None):
    """
    Compare the POD and FAR from 2-sigma to 4-sigma for high and low
    :param site: site string
    :param data1_hi: array of high temp data from reference period
    :param data2_hi: array of high temp data from new (warmer climate) period
    :param data1_lo: array of low temp data from reference period
    :param data2_lo: array of low temp data from new (warmer climate) period
    :param subset_label: string label for the subset of data (e.g. month/season)
    """


    # get histogram parameters
    range_min_hi = np.nanmin(np.hstack((data1_hi, data2_hi)))-np.nanmin(np.hstack((data1_hi, data2_hi))) % 10
    range_max_hi = np.nanmax(np.hstack((data1_hi, data2_hi))) + (10 - np.nanmax(np.hstack((data1_hi, data2_hi))) % 10 +
                                                                 10)
    bins_hi = int(range_max_hi - range_min_hi)
    range_min_lo = np.nanmin(np.hstack((data1_lo, data2_lo)))-np.nanmin(np.hstack((data1_hi, data2_hi))) % 10
    range_max_lo = np.nanmax(np.hstack((data1_lo, data2_lo))) + (10 - np.nanmax(np.hstack((data1_hi, data2_hi))) % 10)
    bins_lo = int(range_max_lo - range_min_lo)

    # gev fitting--use function to try a couple times to get a good fit
    shape1_hi, loc1_hi, scale1_hi = get_gev_fit(data1_hi)
    shape2_hi, loc2_hi, scale2_hi = get_gev_fit(data2_hi)

    x_gev_hi = np.linspace(range_min_hi, range_max_hi, bins_hi*10+1)
    y1_gev_hi = gev.pdf(x_gev_hi, shape1_hi, loc1_hi, scale1_hi)
    y2_gev_hi = gev.pdf(x_gev_hi, shape2_hi, loc2_hi, scale2_hi)

    sigma_array = np.linspace(2, 5, 7) # do 30 for longer one
    pod_hi = np.zeros(len(sigma_array))
    far_hi = np.zeros(len(sigma_array))

    # compute POD and FAR of 2.5-sigma event (from reference climate)
    mean1_hi = gev.mean(shape1_hi, loc=loc1_hi, scale=scale1_hi)
    std1_hi = np.sqrt(gev.var(shape1_hi, loc=loc1_hi,scale=scale1_hi))

    # same for low
    shape1_lo, loc1_lo, scale1_lo = get_gev_fit(data1_lo)
    shape2_lo, loc2_lo, scale2_lo = get_gev_fit(data2_lo)

    x_gev_lo = np.linspace(range_min_lo, range_max_lo, bins_lo*10+1)
    y1_gev_lo = gev.pdf(x_gev_lo, shape1_lo, loc1_lo, scale1_lo)
    y2_gev_lo = gev.pdf(x_gev_lo, shape2_lo, loc2_lo, scale2_lo)

    pod_lo = np.zeros(len(sigma_array))
    far_lo = np.zeros(len(sigma_array))

    # compute POD and FAR of 2.5-sigma event (from reference climate)
    mean1_lo = gev.mean(shape1_lo, loc=loc1_lo, scale=scale1_lo)
    std1_lo = np.sqrt(gev.var(shape1_lo, loc=loc1_lo,scale=scale1_lo))

    # calculate a, b, and c params from Durran 2019
    for ind, sig in enumerate(sigma_array):
        sig20_thres = np.where((x_gev_hi > mean1_hi + 2.0 * std1_hi))
        sig_new_thres = np.where((x_gev_hi > mean1_hi + sig * std1_hi))
        c_val = np.sum(y1_gev_hi[sig_new_thres])
        a_val = np.sum(y2_gev_hi[sig_new_thres]) - c_val
        b_val = np.sum(y2_gev_hi[sig20_thres]) - np.sum(y1_gev_hi[sig20_thres]) - a_val
        pod = a_val/(a_val+b_val)
        far = c_val/(a_val+c_val)
        pod_hi[ind] = pod
        far_hi[ind] = far

        sig20_thres = np.where((x_gev_lo > mean1_lo + 2.0 * std1_lo))
        sig_new_thres = np.where((x_gev_lo > mean1_lo + sig * std1_lo))
        c_val = np.sum(y1_gev_lo[sig_new_thres])
        a_val = np.sum(y2_gev_lo[sig_new_thres]) - c_val
        b_val = np.sum(y2_gev_lo[sig20_thres]) - np.sum(y1_gev_lo[sig20_thres]) - a_val
        pod = a_val/(a_val+b_val)
        far = c_val/(a_val+c_val)
        pod_lo[ind] = pod
        far_lo[ind] = far

    # labels
    labels = ['2.0$\sigma$', '2.5$\sigma$', '3.0$\sigma$', '3.5$\sigma$', '4.0$\sigma$', '4.5$\sigma$', '5.0$\sigma$']

    # another way of plotting POD vs FAR
    fig = plt.figure()
    fig.set_size_inches(6, 4)

    plt.plot(far_hi * 100., pod_hi * 100., markerfacecolor='red', marker='o', color='lightcoral', lw=5,
             label='High', markersize=8)
    plt.plot(far_lo * 100., pod_lo * 100., markerfacecolor='blue', marker='o', color='skyblue',
             lw=5, label='Low', markersize=8)
    for i in range(0, len(labels)):
        plt.text(far_hi[i]*100.+2, pod_hi[i]*100.-4, labels[i], color='red')
        plt.text(far_lo[i]*100.+2, pod_lo[i]*100.-4, labels[i], color='blue')


    plt.ylabel('POD (%)')
    plt.xlabel('FAR (%)')
    plt.ylim(0, 100)
    plt.xlim(0, 100)

    plt.legend()
    plt.title('POD vs FAR {} {} (2.0-$\sigma$ threshold)'.format(site, subset_label))

    plt.savefig('{}pod_vs_far_v2_{}_{}.png'.format(config['PLOT_DIR'], site, subset_label), bbox_inches='tight', dpi=200)
    print('Plotted pod_vs_far for {}'.format(site))

    pdb.set_trace()

    fig = plt.figure()
    fig.set_size_inches(6, 4)

    plt.plot(sigma_array, pod_hi*100., color='red', label='High POD')
    plt.plot(sigma_array, far_hi*100., color='red', linestyle=':', label='High FAR')
    plt.plot(sigma_array, pod_lo*100., color='blue', label='Low POD')
    plt.plot(sigma_array, far_lo*100., color='blue', linestyle=':', label='Low FAR')

    plt.ylabel('Percent')
    plt.xlabel('Sigma of extreme event')
    plt.ylim(0, 100)
    plt.xlim(2, 5)

    plt.legend()
    plt.title('POD vs FAR {} {} (2.0 sigma threshold)'.format(site, subset_label))

    plt.savefig('{}pod_vs_far_{}_{}.png'.format(config['PLOT_DIR'], site, subset_label), bbox_inches='tight', dpi=200)
    print('Plotted pod_vs_far for {}'.format(site))

    pdb.set_trace()

    return
