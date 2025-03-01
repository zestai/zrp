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
        """
        Validates the format of Census Tract and Block Group codes. 
        Incorrectly formatted records are added to a list for geocoding.
    
        Parameters
        ----------
        input_data : pd.DataFrame
            DataFrame containing census data.
        """
    
        self.records_to_geocode = []
        
        # If neither column is in input_data, mark all records for geocoding
        if self.census_tract is None and self.block_group is None:
            if self.key in input_data.columns:
                self.records_to_geocode = input_data[self.key].tolist()
            else:
                self.records_to_geocode = input_data.index.tolist()
            print(f"Warning: Both Census Tract and Block Group are missing. All {len(self.records_to_geocode)} records need geocoding.")

        
        if self.census_tract:
            input_data[self.census_tract] = input_data[self.census_tract].astype(str).str.strip()
    
            # Identify invalid or missing census tract records
            invalid_tracts = input_data[
                (input_data[self.census_tract].str.len() != 11) |
                (~input_data[self.census_tract].str.isnumeric()) |
                (input_data[self.census_tract].isnull()) 
            ]
    
            if not invalid_tracts.empty:
                if self.key in input_data.columns:
                    self.records_to_geocode.extend(invalid_tracts[self.key].tolist())
                else:
                    self.records_to_geocode.extend(invalid_tracts.index.tolist())
    
        if self.block_group:
            input_data[self.block_group] = input_data[self.block_group].astype(str).str.strip()
    
            # Identify invalid or missing block group records
            invalid_bgs = input_data[
                (input_data[self.block_group].str.len() != 12) |
                (~input_data[self.block_group].str.isnumeric()) |
                (input_data[self.block_group].isnull())  
            ]
    
            if not invalid_bgs.empty:
                if self.key in input_data.columns:
                    self.records_to_geocode.extend(invalid_bgs[self.key].tolist())
                else:
                    self.records_to_geocode.extend(invalid_bgs.index.tolist())
    
        self.records_to_geocode = list(set(self.records_to_geocode))
        
        if self.records_to_geocode:
            print(f"Warning: {len(self.records_to_geocode)} records need to be geocoded due to missing or improperly formated Census Tract or Block Group.")

                
    
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
        
        if self.key in data.columns:
            to_geocode = data[data[self.key].isin(self.records_to_geocode)]
        else:
            to_geocode = data[data.index.isin(self.records_to_geocode)]
            
        geocode = ZGeo(file_path=self.file_path, **self.params_dict)
        geocode_out = [] 
        geo_grps = to_geocode.groupby([self.state])
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
            geo_coded_keys = list(geo_coded[f"{self.key}_COL"].values)
            rename_dict = {self.block_group:'GEOID_BG',
                           self.census_tract:'GEOID_CT', 
                           self.zip_code:'GEOID_ZIP'}
            
            data_not_geo_coded = data[~data.index.isin(geo_coded_keys)]
            if 'GEOID_ZIP' in data_not_geo_coded.columns:
                if self.zip_code!="GEOID_ZIP":
                    if data_not_geo_coded[self.zip_code].equals(data_not_geo_coded["GEOID_ZIP"]):
                        data_not_geo_coded =data_not_geo_coded.drop(["GEOID_ZIP"], axis=1)
                    else:
                        if data_not_geo_coded[self.zip_code].isna().mean()>data_not_geo_coded["GEOID_ZIP"].isna().mean():
                            data_not_geo_coded =data_not_geo_coded.drop([self.zip_code], axis=1)
                        else:
                            data_not_geo_coded =data_not_geo_coded.drop(["GEOID_ZIP"], axis=1)

            data_not_geo_coded = data_not_geo_coded.rename(columns={self.block_group:'GEOID_BG', self.census_tract:'GEOID_CT', self.zip_code:'GEOID_ZIP'})
            data_not_geo_coded = data_not_geo_coded.drop(['house_number_LEFT', 'house_number_RIGHT'], axis=1)
            
            # Save data that does not require geocoding
            if self.runname is not None:
                file_like = f"Zest_Geocoded_{self.runname}__{self.year}__00"
            else:
                file_like = f"Zest_Geocoded__{self.year}__00"
            file_name = f'{file_like}_n.parquet'
            save_dataframe(data_not_geo_coded, self.out_path, file_name)
        
            geo_coded = pd.concat([geo_coded, data_not_geo_coded])  

        else:
            if self.zip_code!="GEOID_ZIP":
                if data[self.zip_code].equals(data["GEOID_ZIP"]):
                    data =data.drop(["GEOID_ZIP"], axis=1)
                else:
                    if data[self.zip_code].isna().mean()>data["GEOID_ZIP"].isna().mean():
                        data =data.drop([self.zip_code], axis=1)
                    else:
                        data =data.drop(["GEOID_ZIP"], axis=1)

            geo_coded = data.rename(columns={self.block_group:'GEOID_BG', self.census_tract:'GEOID_CT', self.zip_code:'GEOID_ZIP'})
            for col in ["GEOID_BG", "GEOID_CT", "GEOID_ZIP", "GEOID"]:
                if col not in geo_coded.columns:
                     geo_coded[col] = None 
            geo_coded[f"{self.key}_COL"] = geo_coded.index                                       
            
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

        # set geoid
        data_out['GEOID'] = np.where(data_out.acs_source=='BG', data_out['GEOID_BG'],
                                    np.where(data_out.acs_source=='CT', data_out['GEOID_CT'],
                                            np.where(data_out.acs_source=='ZIP', data_out['GEOID_ZIP'],
                                                     None)))
        
        #######################################
        # cleanup data_out columns
        #######################################
        data_columns=list(data_out.columns)
        if self.key in data_columns:
            data_columns.remove(self.key)
        object_columns = ['GEOID_BG', 'GEOID_CT', 'GEOID_ZIP',
                          'GEOID', 'acs_source',self.key+'_COL',
                          'GEO_NAME','EXT_GEOID', 'FROMHN_LEFT','TOHN_LEFT']
        drop_columns = ['GEO_NAME','EXT_GEOID','FROMHN_LEFT','TOHN_LEFT','original_race','original_sex']
        data_out = data_out[list(set(data_out.columns)-set(drop_columns))]
        data_out_float_columns = list(set(data_out.columns)-set(data_columns))
        for col in object_columns:
            if col in data_out_float_columns:
                data_out_float_columns.remove(col)
                    
        for col in data_out_float_columns:
            try:
                data_out[col] = data_out[col].astype('float32')  
            except ValueError as e:
                print(e)
        if 'age' in data_out.columns:
            data_out['age'] = data_out['age'].astype('float32')
        cat_columns = set(data_out.columns).intersection(set([self.street_address, self.city, self.first_name, self.last_name, self.house_number, self.middle_name,  self.zip_code, self.state, self.race, self.county, 'sex', 'GEOID_BG', 'GEOID_ZIP', 'GEOID_CT', 'acs_source', 'GEOID']))
        
        for col in cat_columns:
            if col in data_out.columns:
                data_out[col] = data_out[col].astype('category')
        return(data_out)
