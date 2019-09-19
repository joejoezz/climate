'''
uses ulmo to find station information
sample usage to find seattle: find_sites.py -s wa -n seattle
'''

import pdb
import ulmo
import numpy as np
import pandas as pd
from optparse import OptionParser

def get_command_options():
    parser = OptionParser()
    parser.add_option('-s', '--specify state', dest='state', type='string', action='store', default=None,
                      help='Limit search to a particular state, abbreviate states with 2 letters')
    parser.add_option('-n', '--specify station name', dest='name', type='string', action='store', default=None,
                      help='Specify part of station name to narrow search--must match exactly')
    parser.add_option('-t', '--specify network type', dest='type', type='string', action='store', default='W',
                      help='Overwrite network type (default = W, other options are 1 and C')
    (opts, args) = parser.parse_args()
    return opts, args

def list_stations(state=None, name=None, type=type):
    st = ulmo.ncdc.ghcn_daily.get_stations(state=state, as_dataframe=True)
    # shorten list if name is specified
    st = st[st['network'].str.contains(type)]
    if name is not None:
        st = st[st['name'].str.contains(name)]

    if len(st) > 0:
        print('{} stations found'.format(len(st)))
        print(st)
    else:
        print('No stations found matching state:{} name:{} type{}'.format(state, name, type))
    pdb.set_trace()

    return

def download_data(id):

    pdb.set_trace()
    st = st.dropna(subset=['wm_oid'])
    # narrow to the 16 washington state stations
    st_wa = st[st['state'] == 'WA']
    #pdb.set_trace()
    st_wa.to_pickle('./Metadata/wa_stations.p')

    # load station data
    st_wa = pd.read_pickle('./Metadata/wa_stations.p')

    # get a subset (KUIL ONLY)
    #st_wa = st_wa[st_wa['id'] == 'USW00094240']

    wa_names = st_wa['name'].values
    wa_ids = st_wa['id'].values
    wa_lats = st_wa['latitude'].values
    wa_lons = st_wa['longitude'].values
    wa_elev = st_wa['elevation'].values

    return
  

if __name__ == "__main__":
    options, arguments = get_command_options()
    state = options.state
    name = options.name
    type = options.type
    if state is not None:
        state = state.upper()
    if name is not None:
        name = name.upper()
    df_stations = list_stations(state=state, name=name, type=type)
    print(df_stations)



main()
