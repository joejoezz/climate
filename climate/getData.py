"""
Script to download data from a specified list of stations into a pandas dataframe
takes an input file which contains site information (see sites.conf)
"""

from util import get_config
from argparse import ArgumentParser
import ulmo
import numpy as np
import pandas as pd

def parse_args():
    """
    Parse input arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to sites config file")
    arguments = parser.parse_args()
    return arguments

args = parse_args()

def get_data(config):
    """
    Function to get data using ulmo and save to directory
    Config file determines which sites to pull and where to store data
    """
    archive_dir = config['SAVE_DIR']
    sites = config['Sites'].keys()
    for site in sites:
        print('Getting climate data for {}'.format(config['Sites'][site]['long_name']))
        id = config['Sites'][site]['id']
        data = ulmo.ncdc.ghcn_daily.get_data(id, as_dataframe=True)
        print('Got data -- length {}'.format(len(data['TMAX'])))

        # collect and reformat various data types
        tmax = np.round((data['TMAX'].copy().values[:, 0].astype('float')) / 10. * 9. / 5. + 32., 0)
        tmin = np.round((data['TMIN'].copy().values[:, 0].astype('float')) / 10. * 9. / 5. + 32., 0)
        if 'WSF2' in data.keys():
            wind = np.round((data['WSF2'].copy().values[:, 0].astype('float')) / 10. * 1.94384, 0)
            drct = data['WSF2'].copy().values[:, 0].astype('float')
        else:
            wind = []
            drct = []
        if 'PRCP' in data.keys():
            precip = np.round((data['PRCP'].copy().values[:, 0].astype('float')) / 2.54 / 100., 2)
        else:
            precip = []
        if 'SNOW' in data.keys():
            snow = np.round((data['SNOW'].copy().values[:, 0].astype('float')) * 0.0393700787, 1)
        else:
            snow = []

        # convert and combine into dataframe
        tmax = pd.Series(tmax, index=data['TMAX'].index.astype('datetime64'))
        tmin = pd.Series(tmin, index=data['TMIN'].index.astype('datetime64'))
        if len(wind) > 0:
            wind = pd.Series(wind, index=data['WSF2'].index.astype('datetime64'))
            drct = pd.Series(drct, index=data['WDF2'].index.astype('datetime64'))
        else:
            wind = pd.Series()
            drct = pd.Series()
        if len(precip) > 0:
            precip = pd.Series(precip, index=data['PRCP'].index.astype('datetime64'))
        else:
            precip = pd.Series()
        if len(snow) > 0:
            snow = pd.Series(snow, index=data['SNOW'].index.astype('datetime64'))
        else:
            snow = pd.Series()

        df = pd.DataFrame(index=tmax.index, columns=['tmax', 'tmin', 'wind', 'drct', 'precip', 'snow'])
        df['tmax'] = tmax
        df['tmin'] = tmin
        df['wind'] = wind
        df['drct'] = drct
        df['precip'] = precip
        df['snow'] = snow

        # drop where all columns are missing data
        df = df.dropna(thresh=1)
        # save as pickle
        df.to_pickle('{}/{}_climo.p'.format(archive_dir, site))

        print ('got data for {}'.format(config['Sites'][site]['long_name']))

    return


if __name__ == "__main__":
    '''
    download climate data
    '''
    config = get_config(args.config)
    get_data(config)
