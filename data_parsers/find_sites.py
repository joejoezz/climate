'''
uses ulmo to find station information
'''

import pdb
import ulmo
import numpy as np
import pandas as pd


def list_stations(state=None):
  st = ulmo.ncdc.ghcn_daily.get_stations(state=state, as_dataframe=True)
  # abbreviate to ones with a WMO ID
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
  
  
def main(state=None):
    """
    print a list of metadata associated with sites meeting criteria
    """
    
    df_stations = list_stations(state=None)
    print(df_stations)

    return 
