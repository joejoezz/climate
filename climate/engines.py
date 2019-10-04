"""
engine to compute stats and make plots
currently one needs to directly run this file to generate stuff...eventually these functions will be called from a
different script
"""

from util import get_config, load_pdo, get_gev_fit
from plot_scripts import histogram, histogram_2d, pod_vs_far, pod_vs_far_shift_mean
import numpy as np
import pandas as pd
import pdb

config = get_config('../sites.conf')
# set paramaters
sites = config['Sites'].keys()
# reference periods are 2-element lists of start and end year
ref_period = [1950, 1979]
new_period = [2010, 2019]
# months is a list of months to use--could be customized to do seasons as well at some point
months = np.linspace(1, 12, 12)

# variable options: 'tmax', 'tmin', 'precip', 'snow'
variables = ['tmax', 'tmin']
subset_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def get_data(site):
    """
    loads site data
    """
    df = pd.read_pickle('../archive/{}_climo.p'.format(site))
    return df

def check_data_size(site, data, period, months):
    """
    Checks the number of elements compared to what is expected based on input dates
    Prints warning if data is missing
    ***Currently a placeholder, not functional***
    """

    actual_ele = len(data)
    expected_ele = float('nan')
    if ():
        print('Warning: {} data has {} elements, {} expected')

    else:
        return


def rainy_day_analysis():
    # rainy day analysis
    site = 'KSEA'
    df = get_data(site)
    from plot_scripts import monthly_summary
    monthly_summary.monthly_summary_stats(df, 9)
    return


def histogram_2d_plot_engine(variable):
    """
    Engine to generate 2d histogram plots
    Need to run histogram_2d_stats_engine() first or this will not work
    It will plot all sites in the dataframes but it would be possible to customize
    """
    dir_2d = '{}histogram_2d/'.format(config['SAVE_DIR'])
    df_ref = pd.read_pickle('{}mean_ref_{}.p'.format(dir_2d, variable))
    df_new = pd.read_pickle('{}mean_new_{}.p'.format(dir_2d, variable))
    df_lower = pd.read_pickle('{}mean_ci_lower_{}.p'.format(dir_2d, variable))
    df_upper = pd.read_pickle('{}mean_ci_upper_{}.p'.format(dir_2d, variable))
    df_ref_std = pd.read_pickle('{}std_ref_{}.p'.format(dir_2d, variable))
    histogram_2d.plot_histogram_2d(df_ref, df_new, df_lower, df_upper, df_ref_std, variable)
    return


def histogram_2d_stats_engine(sites, variable):
    """
    Engine to generate statistics for 2d histograms
    The dataframes are saved in the archive folder
    """
    df_ref_list = []
    df_new_list = []
    for site in sites:
        # get data
        df = get_data(site)
        # create two dataframes with specified years
        df_ref = df[((df.index.year >= ref_period[0]) & (df.index.year <= ref_period[1]))]
        df_new = df[((df.index.year >= new_period[0]) & (df.index.year <= new_period[1]))]
        df_ref_list.append(df_ref[variable])
        df_new_list.append(df_new[variable])
    histogram_2d.get_monthly_2d_stats(df_ref_list, df_new_list, sites, variable)
    return


def histogram_plot_engine(site, variable, month, subset_label):
    """
    Engine to make quick histogram plots for a given site, variable, and month
    Run in a loop to mass-generate plots
    """
    df = get_data(site)
    # get data
    # create two dataframes with specified years
    df_ref = df[((df.index.year >= ref_period[0]) & (df.index.year <= ref_period[1]))]
    df_new = df[((df.index.year >= new_period[0]) & (df.index.year <= new_period[1]))]
    # trim to only specified months
    df_ref = df_ref[df_ref.index.month.isin([month])]
    df_new = df_new[df_new.index.month.isin([month])]
    pdb.set_trace()
    # plot a histogram
    label1 = '{}-{}'.format(str(ref_period[0]), str(ref_period[1]))
    label2 = '{}-{}'.format(str(new_period[0]), str(new_period[1]))
    # ---- EVENTUALLY CHECK DATA SIZE HERE -----
    histogram.plot_histogram(site, df_ref[variable].dropna().values, df_new[variable].dropna().values,
                             label1=label1, label2=label2, subset_label=subset_label, variable=variable)
    return


# code to run the regular histogram plotting part
for site in sites:
    for variable in variables:
        for month in months:  
            subset_label = subset_labels[int(month)-1]
            histogram_plot_engine(site, variable, month, subset_label)


# code to run the 2d histogram part
# eventually move this into a different file
for variable in variables:
    histogram_2d_stats_engine(sites, variable)
    histogram_2d_plot_engine(variable)


pdb.set_trace()
#-------------
# code below this still needs to be put into functions
#-------------



# POD vs FAR shifts (this is an example--put in function eventually)
site = 'KPDX'
df = get_data(site)
df_ref = df[((df.index.year >= ref_period[0]) & (df.index.year <= ref_period[1]))]
df_new = df[((df.index.year >= new_period[0]) & (df.index.year <= new_period[1]))]
# trim to only specified months
df_ref = df_ref[df_ref.index.month.isin(months)]  # BUG HERE!!!!
df_new = df_new[df_new.index.month.isin(months)]
pod_vs_far_shift_mean.plot_pod_vs_far(site, df_ref['tmax'].dropna().values, df_ref['tmin'].dropna().values,
                           subset_label=subset_labels[0])

pod_vs_far.plot_pod_vs_far(site, df_ref['tmax'].dropna().values, df_new['tmax'].dropna().values,
                           df_ref['tmin'].dropna().values, df_new['tmin'].dropna().values,
                           subset_label=subset_labels[0])




