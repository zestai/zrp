from os.path import join, expanduser
from zrp.prepare.utils import *
from zrp.prepare.preprocessing import *
from zrp.prepare.geo_geocoder import *
from zrp.prepare.acs_mapper import *
from zrp.prepare.base import BaseZRP
import datetime as dt
import pandas as pd
import numpy as np
import warnings
import glob
import json
import tqdm
import sys
import os
import re
import tqdm
import pycm
import pickle
import joblib
import surgeo

class BISGWrapper(BaseZRP):
    """Wrapper function for bisg"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def fit(self):
        return self
    
    def transform(self, data):
        df = data.copy()
        df = df.reset_index().drop_duplicates(self.key)
        df = df.filter([self.last_name, self.zip_code, self.census_tract])
        bisg = surgeo.SurgeoModel()
        bisg_results = bisg.get_probabilities(names  = df[self.last_name] ,  geo_df = df[self.zip_code].astype(int))
        
        combo = df.merge(bisg_results, 
                           how="left", 
                           left_on=[self.last_name, self.zip_code], 
                           right_on=["name", "zcta5"]
                          )
        combo.drop(columns = ['zcta5', 'name'], inplace=True)
        
        combo.rename(columns = {
            'white': 'BISG_WHITE',
            'black': 'BISG_BLACK',
            'api': 'BISG_AAPI',
            'native': 'BISG_AIAN',
            'multiple': 'BISG_OTHER',
            'hispanic': 'BISG_HISPANIC'
        }, inplace=True) 
        
        # Generate proxy at threshold
        subset = combo[['BISG_WHITE', 'BISG_BLACK', 'BISG_AAPI', 'BISG_AIAN', 'BISG_OTHER', 'BISG_HISPANIC'
                       ]]
        identifiedRaces = subset.idxmax(axis=1)
        identifiedRaces = identifiedRaces.astype(str).str.replace(
            "BISG_", "", 1)
        combo[self.race] = identifiedRaces
        combo['source_bisg'] = 1
        if self.proxy =='labels':
            proxies = combo[[self.race, "source_bisg"]]
        if self.proxy =='labels':
            proxies = combo[['BISG_WHITE', 'BISG_BLACK', 'BISG_AAPI', 'BISG_AIAN', 'BISG_OTHER', 'BISG_HISPANIC', "source_bisg"]]
        proxies_out = proxies.copy()
        return(proxies_out)


