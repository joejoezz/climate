"""
Climate utility functions
"""

import pandas as pd
from scipy.stats import genextreme as gev
from scipy.stats import mode

def get_config(config_path):
    """
    Retrieve the config dictionary from config_path.
    Written by Jonathan Weyn jweyn@uw.edu
    Modified by Joe Zagrodnik joe.zagrodnik@wsu.edu
    """
    import configobj
    try:
        config_dict = configobj.ConfigObj(config_path, file_error=True)
    except IOError:
        print('Error: unable to open configuration file %s' % config_path)
        raise
    except configobj.ConfigObjError as e:
        print('Error while parsing configuration file %s' % config_path)
        print("*** Reason: '%s'" % e)
        raise

    return config_dict


def load_pdo():
    """
    load monthly pdo dataset
    stored in archive directory, available here: https://www.ncdc.noaa.gov/teleconnections/pdo/
    :return: Pandas series of pdo
    """
    df = pd.read_csv('./archive/pdo_monthly.csv')
    pdo = pd.DataFrame(index = pd.to_datetime(df['yyyymm'], format='%Y%m'), columns=['pdo'])
    pdo['pdo'] = df.pdo.values
    return pdo


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



