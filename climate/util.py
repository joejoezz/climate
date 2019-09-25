"""
Climate utility functions
"""

import pandas as pd

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



