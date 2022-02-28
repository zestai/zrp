from os.path import dirname, join, expanduser
from zrp.validate import *
from .preprocessing import *
from .geo_geocoder import *
from .acs_mapper import *
from .base import BaseZRP
from .utils import *
import pandas as pd
import numpy as np
import warnings
import glob
import json
import sys
import os
import re

import warnings
warnings.filterwarnings(action='ignore')

class ZRP_Prepare(BaseZRP):
    """
    Prepares data to generate race & ethnicity proxies

    Parameters
    ------------
    file_path: str, optional
        Path where to put artifacts and other files generated during intermediate steps.
    """
    
    def __init__(self, file_path=None, *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.params_dict =  kwargs



        
    def fit(self, data):
        if self.census_tract:
            tract_lengths =  data[self.census_tract].str.len()
            tract_len  = most_common(tract_lengths)
            assert (data[self.census_tract].isalnum).any(), "Cannot provide non-numeric Census Tract code, please remove non-numeric census tract records "
            assert tract_len == 11,  "Improper Census Tract format provided. The tool requires the full state fips, county fips, and tract format. (ie '010010202001')"
        if self.block_group:
            bg_lengths =  data[self.block_group].str.len()
            bg_len  = most_common(bg_lengths)
            assert bg_len == 12,  "Improper Census Block Group format provided. The tool requires the full state fips, county fips, tract, and block group format. (ie '0100102020011')"            
    
    def transform(self, input_data):
        """
        Parameters
        ----------
        input_data: pd.Dataframe
            Dataframe to be transformed
        """  
        curpath = dirname(__file__)
        
        # Load Data
        try:
            data = input_data.copy()
            print("Data is loaded")
        except AttributeError:
            data = load_file(self.file_path)
            print("Data file is loaded")
            
        gen_process = ProcessStrings(file_path=self.file_path, **self.params_dict)
        gen_process.fit(data)
        data = gen_process.transform(data)
        
        processed = True
        replicate = True
        print("")

        print("[Start] Preparing geo data")
        data_path = join(curpath, f'../data/processed')
        assert len(os.listdir(data_path)) > 0, "Missing required support files please see the README for how to download the support files: https://github.com/zestai/zrp/blob/main/README.rst#install "        
        
        inv_state_map = load_json(join(data_path, "inv_state_mapping.json"))
        data['zest_in_state_fips'] = data[self.state].replace(inv_state_map)
        print("")

        if self.census_tract is not None:
            try:
                data[self.zip_code] = np.where((data[self.zip_code].isna()) |\
                                        (data[self.zip_code].str.contains("None")),
                                           None,
                                           data[self.zip_code].apply(lambda x: x.zfill(5)))
                data['GEOID_ZIP'] = data[self.zip_code]

            except (ValueError, KeyError) as e:
                pass
            geo_coded = data.copy()
            try:
                geo_coded['GEOID_BG'] = geo_coded[self.block_group]
            except (ValueError, KeyError) as e:
                pass
            geo_coded['GEOID_CT'] = geo_coded[self.census_tract]
            
        elif (self.census_tract is not None) & (self.street_address is not None):
            geocode = ZGeo(file_path=self.file_path, **self.params_dict)
            geocode.fit(geo_coded)
            geocode_out = [] 
            geo_grps = data.groupby([self.state])
            geo_dict = {}
            for s, g in geo_grps:
                geo_dict[s] = g
            gdkys = list(geo_dict.keys())
            print("  The following states are included in the data:", gdkys)
            
            assert set(gdkys) <= set(list(inv_state_map.keys())), "Provided non-standard state codes. Please use standard 2-letter abbreviation to indicate states to geocode, ex:'CA' for Californina"
            
            geo_out = [] 
            for s in tqdm(gdkys):                
                print(" ... on state:", str(s))
                geo = inv_state_map[s].zfill(2)
                output = geocode.transform(geo_dict[s], geo, True)
                geocode_out.append(output)
            geo_coded = pd.concat(geocode_out)
            geo_coded = geo_coded.drop_duplicates()
        else:
            geocode = ZGeo(file_path=self.file_path, **self.params_dict)
            geocode_out = [] 
            geo_grps = data.groupby([self.state])
            geo_dict = {}
            for s, g in geo_grps:
                geo_dict[s] = g
            gdkys = list(geo_dict.keys())
            print("  The following states are included in the data:", gdkys)
                  
            assert set(gdkys) <= set(list(inv_state_map.keys())), "Provided non-standard state codes. Please use standard 2-letter abbreviation to indicate states to geocode, ex:'CA' for Californina"

            geo_out = [] 
            for s in tqdm(gdkys):
                print("   ... on state:", str(s))
                geo = inv_state_map[s].zfill(2)
                output = geocode.transform(geo_dict[s], geo, processed, replicate, True)
                geocode_out.append(output)
            geo_coded = pd.concat(geocode_out)
        
        # append data unable to enter geo mapping
        geo_coded_keys = list(geo_coded.ZEST_KEY_COL.values)
        data = data[~data.index.isin(geo_coded_keys)]
        geo_coded = pd.concat([geo_coded, data])
        

        print("")
        
        print("[Completed] Preparing geo data")
        print("")
        print("[Start] Preparing ACS data")
        
        print("   [Start] Validating ACS input data")
        validate = ValidateGeocoded()
        validate.fit()
        acs_validator = validate.transform(geo_coded)
        save_json(acs_validator, self.out_path, "input_acs_validator.json")
        print("   [Completed] Validating ACS input data")
        print("")

        amp = ACSModelPrep(**self.params_dict)
        amp.fit()
        data_out = amp.transform(geo_coded, False)
        print("[Complete] Preparing ACS data")
        print("")
        
        return(data_out)
