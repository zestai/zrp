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
    key: str, default 'ZEST_KEY', 
        Key to set as index. Optional. If not provided, a key will be generated.
    first_name: str, default 'first_name'
        Name of first name column
    middle_name: str, default 'middle_name'
        Name of middle name column
    last_name: str, default 'last_name'
        Name of last name/surname column
    house_number: str, default 'house_number'
        Name of house number column. Also known as primary address number this is the unique number assigned to a building to delineate it from others on a street. This is usually the first component of a delivery address line. Optional to not provide if Census GEOIDs are provided and no geocoding is required.
    street_address: str, default 'street_address'
        Name of street address column. The street address is usually comprised of predirectional, street name, and street suffix. 
    city: str, default 'city'
        Name of city column.
    state: str, default 'state'
        Name of state column.
    zip_code: str, default 'zip_code'
        Name of zip or postal code column.
    race: str, default 'race', optional
        Name of the race column
    census_tract: str, optional
        Name of Census tract column. Recommended to provide if available or a geocoder is typically implemented.
    block_group: str, optional
        Name of Census block group column. Recommended to provide if available or a geocoder is typically implemented.
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
        Geocoding indicator, to be deprecated by version 0.4.0.
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
    
    def rename_data_columns(self, data):
        """
        Renames the user specified columns of the input data to the default column names expected by ZRP.

        Parameters
        -----------
        data: pd.Dataframe
            Dataframe to be transformed
        """
        renamed_columns = {self.first_name: "first_name", self.middle_name: "middle_name", self.last_name: "last_name", self.house_number: "house_number", self.street_address: "street_address", self.city: "city", self.state: "state", self.zip_code: "zip_code"}
        data = data.rename(columns=renamed_columns)
        # self.params_dict = {}
        return data
    
    def check_for_old_files(self):
        """
        Checks if there are no files created in previous runs.

        Parameters
        -----------
        """
        old_files = []
        if self.runname is not None:
            file_like_geo = f'Zest_Geocoded_{self.runname}__{self.year}__'
            file_like_zrp_proxy = f'proxy_output_{self.runname}.feather'
            file_like_bisg_proxy = f'bisg_proxy_output__{self.runname}.feather'
        else:
            file_like_geo = f'Zest_Geocoded__{self.year}__'
            file_like_zrp_proxy = 'proxy_output.feather'
            file_like_bisg_proxy = 'bisg_proxy_output.feather'
        
        for file in os.listdir(self.out_path):
            if file_like_geo in file:
                old_files.append(os.path.join(self.out_path,file))
           
        file = os.path.join(self.out_path,file_like_zrp_proxy)
        if os.path.exists(file):
            old_files.append(file)
        
        file = os.path.join(self.out_path,file_like_bisg_proxy)
        if os.path.exists(file):
            old_files.append(file)
        
        if len(old_files) > 0:
            raise Exception(f"New value of 'runname' parameter needs to be specified or the following files need to be moved or deleted: {old_files}")
        
                  
    
    def transform(self, input_data, chunk_size = 25000):
        """
        Processes input data and generates ZRP predictions. Generates BISG predictions additionally if specified.

        Parameters
        -----------
        input_data: pd.Dataframe
            Dataframe to be transformed
        chunk_size: int
            Numer of rows to be processed in each iteration. input_data processed all at once if None is provided.
        """
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.file_path)

        data = self.rename_data_columns(data)
        self.reset_column_names()
        
        make_directory(self.out_path)
        self.check_for_old_files()
        curpath = dirname(__file__)
        if self.pipe_path is None:
            self.pipe_path = join(curpath, "modeling/models")

        data = data.sort_values('state')
        if chunk_size is None:
            chunk_size = len(data)
        chunk_max = int((len(data)-1)/chunk_size) + 1
        predict_out_list = list()
        full_bisg_proxies_list = list()
        for chunk in range(chunk_max):
            print("####################################")
            print(f'Processing rows: {chunk*chunk_size}:{(chunk+1)*chunk_size}')
            print("####################################")
            data_chunk = data[chunk*chunk_size:(chunk+1)*chunk_size]
            z_prepare = ZRP_Prepare(file_path=self.file_path, **self.params_dict)
            z_prepare.fit(data_chunk)
            prepared_data_chunk = z_prepare.transform(data_chunk)

            z_predict = ZRP_Predict(file_path=self.file_path, pipe_path=self.pipe_path, **self.params_dict)
            z_predict.fit(prepared_data_chunk)
            predict_out_chunk = z_predict.transform(prepared_data_chunk, save_table = False)
            predict_out_list.append(predict_out_chunk)
            
            if self.bisg:
                bisgw = BISGWrapper(**self.params_dict)
                bisg_proxies_chunk = bisgw.transform(prepared_data_chunk[~prepared_data_chunk.index.duplicated(keep='first')])
                full_bisg_proxies_list.append(bisg_proxies_chunk)
           
        predict_out = pd.concat(predict_out_list)
        
        if self.runname is not None:
            file_name = f'proxy_output_{self.runname}.feather'
        else:
            file_name = 'proxy_output.feather'
        save_feather(predict_out, self.out_path, file_name)    
        if self.bisg:
            full_bisg_proxies = pd.concat(full_bisg_proxies_list)
            if self.runname is not None:
                file_name = f'bisg_proxy_output_{self.runname}.feather'
            else:
                file_name = 'bisg_proxy_output.feather'
            save_feather(full_bisg_proxies, self.out_path, file_name) 

        try:
            predict_out = input_data.merge(predict_out.reset_index(drop=False), on=self.key)
        except KeyError:
            pass

        return (predict_out)