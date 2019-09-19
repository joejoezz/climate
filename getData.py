"""
Script to download data from a specified list of stations into a pandas dataframe
takes an input file with
"""

import pdb
from util import get_config
from argparse import ArgumentParser

def parse_args():
    """
    Parse input arguments.
    """

    parser = ArgumentParser()
    parser.add_argument("config", help="Path to theta-e config file")
    arguments = parser.parse_args()
    return arguments

args = parse_args()

def get_data():
    """
    Function to get data using ulmo and save to directory
    :return:
    """



if __name__ == "__main__":
    '''
    download climate data
    '''
    pdb.set_trace()
    config = get_config(args.config)




