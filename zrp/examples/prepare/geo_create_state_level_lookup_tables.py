from os.path import join, expanduser
import datetime as dt
import pandas as pd
import numpy as np
import warnings
import glob
import json
import sys
import os
import re
from prepare.utils import *

warnings.filterwarnings(action='once')
home = expanduser('~')

state_list = ['02', '01', '05', '04', '06', '08', '12', '09', '10', '11', '13',
       '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
       '26', '27', '28', '29', '30', '31', '32', '35', '33', '34', '36',
       '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48',
       '49', '50', '51', '53', '54', '55', '56', '72']


from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
support_files_path
path = os.path.join(support_files_path, 'processed/geo/2019')

def geo_stuff(s, path):
    data_tmp = []
    parsed_files = [file for file in os.listdir(path)  if f'Zest_Geo_Lookup_2019_{s}' in file]
    print("State:",s, len(parsed_files))
    for file in parsed_files:
        tmp = load_file(os.path.join(path, file))
        data_tmp.append(tmp)
    df_out = pd.concat(data_tmp)
    df_out.to_parquet(os.path.join(path, f'Zest_Geo_Lookup_2019_State_{s}.parquet'))
    lnd =len(df_out)
    print('   Saved state', s, lnd)
    df_out = None
    print("")
    return( print('   Saved state', s, lnd))

results = Parallel(n_jobs = 80, prefer='threads', verbose=1)(delayed((geo_stuff))(s, path) for s in tqdm(list(state_list)))
