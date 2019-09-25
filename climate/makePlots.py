"""
engine to make plots using plot_scripts
"""

from util import get_config, load_pdo
from plot_scripts import histogram
import pandas as pd

# set paramaters
sites = ['KALW'] #eventually set this to whatever is in the config file
# reference periods are 2-element lists of start and end year
ref_period = [1950, 1979]
new_period = [2010, 2019]
# months is a list of months to use (for a season list 3 months)
months = [5]

# variable options: 'tmax', 'tmin', 'precip', 'snow'
variable = 'tmax'
subset_label = 'May TMAX'

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
    NOT COMPLETE YET
    """

    actual_ele = len(data)
    expected_ele = float('nan')
    if ():
        print('Warning: {} data has {} elements, {} expected')

    else:
        return


for site in sites:
    # get data
    df = get_data(site)
    # create two dataframes with specified years
    df_ref = df[((df.index.year >= ref_period[0]) & (df.index.year <= ref_period[1]))]
    df_new = df[((df.index.year >= new_period[0]) & (df.index.year <= new_period[1]))]
    # trim to only specified months
    df_ref = df_ref[df_ref.index.month.isin(months)]
    df_new = df_new[df_new.index.month.isin(months)]
    # plot a histogram
    label1 = '{}-{}'.format(str(ref_period[0]), str(ref_period[1]))
    label2 = '{}-{}'.format(str(new_period[0]), str(new_period[1]))
    # ---- EVENTUALLY CHECK DATA SIZE HERE -----
    histogram.plot_histogram(site, df_ref[variable].values, df_new[variable].values, label1=label1, label2=label2,
                             subset_label=subset_label)

