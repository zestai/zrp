from os.path import dirname, join, expanduser
from zrp.validate import *
from .preprocessing import *
from .geo_geocoder import *
from .acs_mapper import *
from .base import BaseZRP
from .utils import *
from concurrent.futures import ThreadPoolExecutor
from censusgeocode import CensusGeocode
from tqdm import tqdm
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
        
        if self.geocoding_type not in ['zrp', 'api', 'zrp->api', 'api->zrp']:
            raise ValueError(f"'geocoding_type' argument of out rannge. Allowed values: 'zrp', 'api', 'zrp->api', 'api->zrp'")
        if self.censusapi_vintage not in ['Census2020_Current', 'ACS2019_Current']:
            raise ValueError(f"'censusapi_vintage' argument of out rannge. Allowed values: 'Census2020_Current', 'ACS2019_Current'")
        
    def censusapi_geocoding(self, df_in):
        """
        Uses Census Bureau API to geocode input_data. 

        Parameters
        ------------
        df_in: Dataframe
            Data to be geocoded.
        """
        def geocode(row):
            index, adr, state, city, hn, zip_code = row
            data = dict() 
            try:
                census_group = cg.address(hn + ' '+ adr, city=city, state=state)
                
                for census in census_group:
                    try:
                        if census['addressComponents']['zip'] == zip_code:
                            data['GEOID_CT'] = census['geographies']['Census Tracts'][0]['GEOID']
                            if self.censusapi_vintage == 'Census2020_Current':
                                data['GEOID_BG'] = census['geographies']['Census Blocks'][0]['GEOID']
                            break
                    except:
                        pass
            except:
                pass

            return data
        print("      ...Census Bureau API geocoding") 
        df_in = df_in.reset_index(drop = False)
        num_of_chunks = int(len(df_in)/1000) + 1
        cg = CensusGeocode(benchmark='Public_AR_Current', vintage=self.censusapi_vintage)

        for i in range(num_of_chunks):
            chunk = df_in[[self.street_address,self.state, self.city, self.house_number, self.zip_code]][i*1000:(i+1)*1000]
            with ThreadPoolExecutor() as tpe:
                data = list(tqdm(tpe.map(geocode, chunk.itertuples()), total=len(chunk)))
            if i == 0:
                df = pd.DataFrame.from_records(data)
            else:
                df_chunk = pd.DataFrame.from_records(data)
                df = df.append(df_chunk)
                
        df = df.reset_index(drop = True)
        if 'GEOID_CT' in df.columns:
            df['GEOID_CT'] = df['GEOID_CT'].str[:11]
        else:
            df['GEOID_CT'] = None
        if 'GEOID_BG' in df.columns:
            df['GEOID_BG'] = df['GEOID_BG'].str[:12]
            df['GEOID_CT'] = df['GEOID_CT'].fillna(df['GEOID_BG'].str[:11])
        else:
            df['GEOID_BG'] = None
        df = pd.concat([df_in, df], axis = 1)
        df['GEOID_ZIP'] = df[self.zip_code]
        df = df.set_index(self.key)
        return df       
        
    
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

        geo_folder = os.path.join(data_path, "geo", self.year)
        acs_folder = os.path.join(data_path, 'acs', self.year, self.span+'yr')

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
        # Census API geocoding
        if self.geocoding_type == 'api' or self.geocoding_type == 'api->zrp':
            data = self.censusapi_geocoding(data)
            mask = (data['GEOID_CT'].notna()) | (data['GEOID_BG'].notna())
            data_api = data[mask]
            data = data[~mask]
        # ZRP geocoding
        if self.geocoding_type == 'zrp' or self.geocoding_type == 'api->zrp' or self.geocoding_type == 'zrp->api':
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
                geo_coded['GEOID_BG'] = None
                geo_coded['GEOID_CT'] = None
                geo_coded['GEOID_ZIP'] = geo_coded[self.zip_code]
        # Census API geocoding
        if self.geocoding_type == 'zrp->api':
            mask = (geo_coded['GEOID_CT'].isna()) & (geo_coded['GEOID_BG'].isna())
            data_api = geo_coded[mask]
            geo_coded = geo_coded[~mask]
            data_api = data_api.drop(['GEOID_CT', 'GEOID_BG'], axis = 1)
            data_api = self.censusapi_geocoding(data_api)
        
        if self.geocoding_type == 'api':
            geo_coded = pd.concat([data_api, data])
        elif self.geocoding_type == 'api->zrp' or self.geocoding_type == 'zrp->api':
            geo_coded = pd.concat([geo_coded, data_api])
         
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
            
        geo_coded["GEOID"] = None
        geo_coded["GEOID"] = geo_coded["GEOID"].fillna(geo_coded["GEOID_BG"])\
                                               .fillna(geo_coded["GEOID_CT"])\
                                               .fillna(geo_coded["GEOID_ZIP"])                                           
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
        
        return data_out
