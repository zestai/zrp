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
    support_files_path: "data/processed"
        File path with support data.
    key: str, default 'ZEST_KEY'
        Key to set as index. If not provided, a key will be generated.
    first_name: str, default 'first_name'
        Name of first name column
    middle_name: str, default 'middle_name'
        Name of middle name column
    last_name: str, default 'last_name'
        Name of last name/surname column
    house_number: str, default 'house_number'
        Name of house number column. Also known as primary address number this is the unique number assigned to a building to delineate it from others on a street. This is usually the first component of a delivery address line.
    street_address: str, default 'street_address'
        Name of street address column. The street address is usually comprised of predirectional, street name, and street suffix. 
    city: str, default 'city'
        Name of city column.
    state: str, default 'state'
        Name of state column.
    zip_code: str, default 'zip_code'
        Name of zip or postal code column.
    race: str, default 'race'
        Name of the race column
    proxy: str, default 'probs'
        Specifies whether to return proxy results as 'probs' (the proxy probabilities) or 'labels'.
    census_tract: str
        Name of census tract column
    street_address_2: str, optional
        Name of additional address column
    name_prefix: str, optional
        Name of column containing full name preix (ie Dr, Sr, and Esq )
    name_suffix: str, optional
        Name of column containing full name suffix (ie jr, iii, and phd)
    na_values: list, optional
        List of missing values to replace.
    file_path: str, optional
        Path where to put artifacts and other files generated during intermediate steps. 
    geocode: bool, default True
        Whether to geocode.
    bisg: bool, default True
        Whether to return BISG proxies. 
    readout: bool, default True
        Whether to return a readout.
    n_jobs: int, default 49
        Number of jobs in parallel
    year: str, default '2019'
        ACS year to use.
    span: str, default '5'
        Year span of ACS data to use.
    runname: str, default 'test'
    """

    def __init__(self, file_path=None, pipe_path=None, *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.pipe_path = pipe_path
        self.params_dict =  kwargs

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
        
        make_directory(self.out_path)

        z_prepare = ZRP_Prepare(file_path=self.file_path, **self.params_dict)
        z_prepare.fit(data)
        prepared_data = z_prepare.transform(data)

        curpath = dirname(__file__)
        if self.pipe_path is None:
            self.pipe_path = join(curpath, "modeling/models")

        z_predict = ZRP_Predict(file_path=self.file_path, pipe_path=self.pipe_path, **self.params_dict)
        z_predict.fit(prepared_data)
        predict_out = z_predict.transform(prepared_data)

        if self.bisg:
            bisg_data = prepared_data.copy()
            bisg_data = bisg_data[~bisg_data.index.duplicated(keep='first')]

            bisgw = BISGWrapper(**self.params_dict)
            full_bisg_proxies = bisgw.transform(bisg_data)
            save_feather(full_bisg_proxies, self.out_path, f"bisg_proxy_output.feather")

        try:
            predict_out = input_data.merge(predict_out.reset_index(drop=False), on=self.key)
        except KeyError:
            pass

        return (predict_out)
