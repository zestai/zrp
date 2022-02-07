from os.path import dirname, join, expanduser
from .generate_bisg import *
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


class ZRP_Prepare(BaseZRP):
    """
    Prepares data to generate race & ethnicity proxies
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data

    state_mapping: dictionary
        dictionary mapping state names & abbreviations
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
    bisg: True
        Whether to return BISG proxies
    readout: bool
        Whether to return a readout
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self, data):
        if self.census_tract:
            tract_lengths =  data[self.census_tract].str.len()
            tract_len  = most_common(tract_lengths)
            assert tract_len == 11,  "Improper Census Tract format provided. The tool requires the full state fips, county fips, and tract format. (ie '010010202001')"
    
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
            
        gen_process = ProcessStrings()
        data = gen_process.transform(data)
        
        processed = True
        replicate = True
        print("")

        print("[Start] Preparing geo data")
        data_path = join(curpath, f'../data/processed')
        
        inv_state_map = load_json(join(data_path, "inv_state_mapping.json"))
        data['zest_in_state_fips'] = data[self.state].replace(inv_state_map)


        if self.census_tract:
            data[self.zip_code] = np.where((data[self.zip_code].isna()) |\
                                        (data[self.zip_code].str.contains("None")),
                                           None,
                                           data[self.zip_code].apply(lambda x: x.zfill(5)))
            geo_coded = data.copy()
            
        elif (self.census_tract is not None) & (self.street_address is not None):
            geocode = ZGeo()
            geocode.fit()
            geocode_out = [] 
            geo_grps = data.groupby([self.state])
            geo_dict = {}
            for s, g in geo_grps:
                geo_dict[s] = g
            print("    The following states are included in the data:", list(geo_dict.keys()))

            geo_out = [] 
            for s in tqdm(list(geo_dict.keys())):                
                #print(" ... on state:", str(s))
                geo = inv_state_map[s].zfill(2)
                output = geocode.transform(geo_dict[s], geo, True)
                geocode_out.append(output)
            geo_coded = pd.concat(geocode_out)
            geo_coded = geo_coded.drop_duplicates()  
        else:
            geocode = ZGeo()
            geocode_out = [] 
            geo_grps = data.groupby([self.state])
            geo_dict = {}
            for s, g in geo_grps:
                geo_dict[s] = g
            print("  The following states are included in the data:",
                  list(geo_dict.keys()))

            geo_out = [] 
            for s in list(geo_dict.keys()):
                print("   ... on state:", str(s))
                geo = inv_state_map[s].zfill(2)
                output = geocode.transform(geo_dict[s], geo, processed, replicate, True)
                geocode_out.append(output)
            geo_coded = pd.concat(geocode_out)
            
        print("[Completed] Preparing geo data")
        print("")
        print("[Start] Preparing ACS data")
        
        amp = ACSModelPrep()
        data_out = amp.transform(geo_coded, False)
        print("[Complete] Preparing ACS data")
        return(data_out)
