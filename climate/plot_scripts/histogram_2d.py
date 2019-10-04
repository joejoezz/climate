"""
make 2-d histograms of monthly temperature statistics at a variety of sites
bootstrap 10,000 samples of size of the new dataset from the old dataset
"""

from sklearn.utils import resample
from util import get_config
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy.stats import mode
import os
import pdb

def bootstrap_mean(ref_data, n_samples=300, runs=10000, alpha=0.95):
    means = []
    for run in range(0, runs):
        boot = resample(ref_data, replace=True, n_samples=n_samples)
        means.append(np.mean(boot))
    # for a 95% conf. interval, p1 = 2.5, p2 = 97.5
    p1 = ((1.0-alpha)/2.0)*100
    ref_lower = np.percentile(means, p1)
    p2 = (alpha+((1.0-alpha)/2.0)) * 100
    ref_upper = np.percentile(means, p2)
    ref_mean = np.mean(means)
    return ref_mean, ref_lower, ref_upper


def get_monthly_2d_stats(df_ref_list, df_new_list, sites, variable):
    """
    makes 2d histogram plots of monthly climate signals
    reference period (30 yr) bootstrapped to match recent period (10 yr)
    :param df_ref_list: array of dfs from reference period
    :param df_new_list: array of dfs from reference period
    :param sites: string list of sites (or whatever should be on plot)
    :param variable:
    :return:
    """

    # first assemble dataframe of mean and std
    df_ref_mean = pd.DataFrame(index=sites, columns=np.linspace(1, 12, 12))
    df_ref_sample = pd.DataFrame(index=sites, columns=np.linspace(1, 12, 12))
    df_ref_std = pd.DataFrame(index=sites, columns=np.linspace(1, 12, 12))
    df_new_mean = pd.DataFrame(index=sites, columns=np.linspace(1, 12, 12))
    df_new_sample = pd.DataFrame(index=sites, columns=np.linspace(1, 12, 12))
    df_ref_ci_lower = pd.DataFrame(index=sites, columns=np.linspace(1, 12, 12))
    df_ref_ci_upper = pd.DataFrame(index=sites, columns=np.linspace(1, 12, 12))


    # loop through sites
    for i,site in enumerate(sites):
        df_ref = df_ref_list[i]
        df_new = df_new_list[i]
        for j in np.arange(1, 13):
            ref_data = df_ref[df_ref.index.month == j].dropna().values
            new_data = df_new[df_new.index.month == j].dropna().values
            ref_mean, ref_ci_bot, ref_ci_top = bootstrap_mean(ref_data, n_samples=300, runs=10000, alpha=0.95)
            df_ref_mean[j][site] = ref_mean
            df_ref_sample[j][site] = len(ref_data)
            df_new_sample[j][site] = len(new_data)
            df_ref_ci_upper[j][site] = ref_ci_top
            df_ref_ci_lower[j][site] = ref_ci_bot
            df_new_mean[j][site] = np.mean(new_data)
            df_ref_std[j][site] = np.std(ref_data)

    mean_diff = df_new_mean - df_ref_mean
    mean_ci_diff = df_ref_ci_upper - df_ref_ci_lower

    # save the data
    conf = get_config('../sites.conf')
    save_dir = '{}histogram_2d/'.format(conf['SAVE_DIR'])
    pd.to_pickle(df_ref_mean, '{}mean_ref_{}.p'.format(save_dir, variable))
    pd.to_pickle(df_new_mean, '{}mean_new_{}.p'.format(save_dir, variable))
    pd.to_pickle(df_ref_ci_lower, '{}mean_ci_lower_{}.p'.format(save_dir, variable))
    pd.to_pickle(df_ref_ci_upper, '{}mean_ci_upper_{}.p'.format(save_dir, variable))
    pd.to_pickle(df_ref_std, '{}std_ref_{}.p'.format(save_dir, variable))

    return


def plot_histogram_2d(df_ref, df_new, df_lower, df_upper, df_ref_std, variable):
    long_names = []
    config = get_config('../sites.conf')
    for site in config['Sites']:
        long_names.append(config['Sites'][site]['long_name'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(7, 5)

    # figure out if warming/cooling is significant at 95% confidence
    ci = df_upper - df_lower
    delta = df_new - df_ref
    delta_lowbound = (df_new - df_upper) > 0
    delta_highbound = df_new - df_lower < 0
    confidence = ((delta_lowbound == True) | (delta_highbound == True)).values
    conf_mask = np.ma.masked_where(confidence == False, confidence)

    X, Y = np.meshgrid(range(1,len(df_ref.columns)+2), range(0,len(df_ref.index)+1))
    meshplot = ax.pcolormesh(X, Y, delta.values.astype('float'), vmin=-5.2, vmax=5.2, cmap='seismic')
    hatchplot = ax.pcolor(X, Y, conf_mask, hatch='.', alpha=0)

    #ax.set_title('Monthly mean temp change (1950-1979) - (2010-2019)')
    subset_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.grid(linestyle='-')

    # Hide major tick labels
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    # Customize minor tick labels
    ax.set_xlim(1, 13)
    ax.set_xticks(np.linspace(1.5, 12.5, 12), minor=True)
    ax.set_xticklabels(subset_labels, minor=True)
    ax.set_yticks(np.linspace(0.5, len(df_ref.index)-0.5, len(df_ref.index)), minor=True)
    ax.set_yticklabels(long_names, minor=True)
    ax.tick_params(which='major', length=0)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cbar = plt.colorbar(meshplot, cax)
    cbar.ax.set_ylabel(r'Mean temperature change ($^\circ$F)')
    config = get_config('../sites.conf')
    plt.savefig('{}monthly_mean_grid_{}.png'.format(config['PLOT_DIR'], variable), bbox_inches='tight', dpi=200)
    pdb.set_trace()
    plt.close()
