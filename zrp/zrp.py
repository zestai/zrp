from zrp.modeling.predict import BISGWrapper, ZRP_Predict
from os.path import dirname, join, expanduser
from zrp.prepare.prepare import ZRP_Prepare
from zrp.prepare.base import BaseZRP
from zrp.prepare.utils import *
import pandas as pd
import numpy as np
import warnings
import surgeo
import pickle
import joblib
import json
import pycm
import sys
import os
import re


class ZRP(BaseZRP):
    """
    Zest Race Predictor, predicts race & ethnicity using name & geograhpic data
    
    Parameters
    ----------
    key: str 
        Key to set as index. If not provided, a key will be generated.
    first_name: str
        Name of first name column
    middle_name: str
        Name of middle name column
    last_name: str
        Name of last name/surname column
    house_number: str
        Name of house number column. Also known as primary address number this is the unique number assigned to a building to delineate it from others on a street. This is usually the first component of a delivery address line.
    street_address: str
        Name of street address column. The street address is usually comprised of predirectional, street name, and street suffix. 
    city: str
        Name of city column
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    support_files_path:
        File path with support data
    street_address_2: str, optional
        Name of additional address column
    name_prefix: str, optional
        Name of column containing full name preix (ie Dr, Sr, and Esq )
    name_suffix: str, optional
        Name of column containing full name suffix (ie jr, iii, and phd)
    na_values: list
        List of missing values to replace 
    file_path: str
        Input data file path
    geocode: bool
        Whether to geocode
    race: str
        Name of race column
    proxy: str
        Type of proxy to return, default is race probabilities
    bisg: bool, default True
        Whether to return BISG proxies
    readout: bool
        Whether to return a readout
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        """
        Processes input data and generates ZRP predictions. Generates BISG predictions additionally if specified.

        Parameters
        -----------
        input_data: pd.Dataframe
            Dataframe to be transformed
        """
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.file_path)
        make_directory()
            
        z_prepare = ZRP_Prepare()
        z_prepare.fit(data)
        prepared_data = z_prepare.transform(data)
               
        print("POST ACS", prepared_data.index.values)
            
        curpath = dirname(__file__)
        pipe_path = join(curpath, "modeling/models")
        
        z_predict = ZRP_Predict(pipe_path = pipe_path)
        z_predict.fit(data)
        predict_out = z_predict.transform(prepared_data)
        
        if self.bisg:
            bisg_data = prepared_data.copy()
            bisg_data = bisg_data[~bisg_data.index.duplicated(keep = 'first')]
            
            bisgw = BISGWrapper()
            full_bisg_proxies = bisgw.transform(bisg_data)
            save_feather(full_bisg_proxies, self.out_path, f"bisg_proxy_{self.proxy}.feather")
            
        try:
            predict_out = input_data.merge(predict_out.reset_index(drop = False), on = self.key)
        except KeyError:
            pass
                    
        return(predict_out)
        

