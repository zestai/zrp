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

        
    def fit(self, input_data):
        if self.census_tract:
            tract_lengths =  input_data[self.census_tract].str.len()
            tract_len  = most_common(tract_lengths)
            if not (input_data[self.census_tract].apply(lambda x: str(x).isalnum()).any()):
                raise ValueError("Cannot provide non-numeric Census Tract code, please remove non-numeric census tract records.")
            if tract_len != 11:
                raise ValueError("Improper Census Tract format provided. The tool requires the full state fips, county fips, and tract format. (ie '06037311600')")

        if self.block_group:
            bg_lengths =  input_data[self.block_group].str.len()
            bg_len  = most_common(bg_lengths)
            if bg_len != 12:  
                raise ValueError("Improper Census Block Group format provided. The tool requires the full state fips, county fips, tract, and block group format. (ie '060373116003')")

    
    def transform(self, input_data):
        """
        Transforms the data
        
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
            
        data_path = join(curpath, f'../data/processed')
        lookup_tables_config = load_json(join(data_path, "lookup_tables_config.json"))

        geo_folder = os.path.join(data_path, "geo", lookup_tables_config['geo_year'])
        acs_folder = os.path.join(data_path, 'acs', lookup_tables_config['acs_year'], lookup_tables_config['acs_span'])

        if not ((os.path.isdir(geo_folder)) &
                (os.path.isdir(acs_folder ))
               ):
            raise AssertionError("Missing required support files please see the README for how to download the support files: https://github.com/zestai/zrp/blob/main/README.rst#install ")
        if not ((len(os.listdir(geo_folder)) > 0) &
                (len(os.listdir(acs_folder)) > 0)):
            raise AssertionError("Missing required support files please see the README for how to download the support files: https://github.com/zestai/zrp/blob/main/README.rst#install ") 
        gen_process = ProcessStrings(file_path=self.file_path, **self.params_dict)
        gen_process.fit(data)
        data = gen_process.transform(data)
        
        print("")

        print("[Start] Preparing geo data")

        inv_state_map = load_json(join(data_path, "inv_state_mapping.json"))
        data['zest_in_state_fips'] = data[self.state].replace(inv_state_map)
        print("")

        geocode = ZGeo(file_path=self.file_path, **self.params_dict)
        geocode_out = [] 
        geo_grps = data.groupby([self.state])
        geo_dict = {}
        for s, g in geo_grps:
            geo_dict[s] = g
        gdkys = list(geo_dict.keys())
        print("  The following states are included in the data:", gdkys)
              
        if not set(gdkys) <= set(list(inv_state_map.keys())):
            raise ValueError("Provided unrecognizable state codes. Please use standard 2-letter abbreviation to indicate states to geocode, ex:'CA' for Californina")

        geo_out = [] 
        for s in tqdm(gdkys):
            print("   ... on state:", str(s))
            geo = inv_state_map[s].zfill(2)
            output = geocode.transform(geo_dict[s], geo, processed = True, replicate = True, save_table = True)
            geocode_out.append(output)
        if len(geocode_out) > 0:
            geo_coded = pd.concat(geocode_out)

            # append data unable to enter geo mapping
            geo_coded_keys = list(geo_coded.ZEST_KEY_COL.values)
            data_not_geo_coded = data[~data.index.isin(geo_coded_keys)]
            geo_coded = pd.concat([geo_coded, data_not_geo_coded])
        else:
            geo_coded = data
            geo_coded['GEOID'] = None
            geo_coded['GEOID_BG'] = None
            geo_coded['GEOID_CT'] = None
            geo_coded['GEOID_ZIP'] = None
            geo_coded["ZEST_KEY_COL"] = geo_coded.index 
        # replace GEOIDs with user-defined values where avaliable
        if self.block_group is not None and self.census_tract is not None:
            geo_coded = geo_coded.drop([self.block_group, self.census_tract], axis = 1)
            geo_coded = geo_coded.merge(data[[self.block_group, self.census_tract]], right_index = True, left_index = True, how = 'left')
            geo_coded['GEOID_BG'] = np.where((geo_coded[self.block_group].isna()) | (geo_coded[self.block_group].str.contains("None") | (geo_coded[self.block_group] == ''))
                                             ,geo_coded['GEOID_BG']
                                             ,geo_coded[self.block_group])
            geo_coded['GEOID_CT'] = np.where((geo_coded[self.census_tract].isna()) | (geo_coded[self.census_tract].str.contains("None") | (geo_coded[self.census_tract] == ''))
                                             ,geo_coded['GEOID_CT']
                                             ,geo_coded[self.census_tract])
            geo_coded = geo_coded.drop([self.block_group, self.census_tract], axis = 1) 
        elif self.block_group is not None:
            geo_coded = geo_coded.drop(self.block_group, axis = 1)
            geo_coded = geo_coded.merge(data[self.block_group], right_index = True, left_index = True, how = 'left')
            geo_coded['GEOID_BG'] = np.where((geo_coded[self.block_group].isna()) | (geo_coded[self.block_group].str.contains("None") | (geo_coded[self.block_group] == ''))
                                             ,geo_coded['GEOID_BG']
                                             ,geo_coded[self.block_group])
            geo_coded = geo_coded.drop(self.block_group, axis = 1)            
        elif self.census_tract is not None:
            geo_coded['GEOID_BG'] = np.nan
            geo_coded = geo_coded.drop(self.census_tract, axis = 1)
            geo_coded = geo_coded.merge(data[self.census_tract], right_index = True, left_index = True, how = 'left')
            print(geo_coded[self.census_tract] == '')
            geo_coded['GEOID_CT'] = np.where((geo_coded[self.census_tract].isna()) | (geo_coded[self.census_tract].str.contains("None") | (geo_coded[self.census_tract] == ''))
                                             ,geo_coded['GEOID_CT']
                                             ,geo_coded[self.census_tract])
            geo_coded = geo_coded.drop(self.census_tract, axis = 1)
                                                   
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
