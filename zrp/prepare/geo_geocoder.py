from os.path import dirname, join, expanduser
from zrp.validate import ValidateGeo
from .preprocessing import *
from .base import BaseZRP
from .utils import *
import pandas as pd
import numpy as np
import statistics
import json
import sys
import os
import re

import warnings
warnings.filterwarnings(action='ignore')


def geo_search(geo_files_path, year, st_cty_code):
    """
    Returns a list of files associated with the state county code

    Parameters
    ----------
    geo_files_path:
        A string representing file path of the folder containing geo lookup tables
    year:
        A string year
    st_cty_code:
        A string for the state city code
    """
    file_list = []
    for root, dirs, files in os.walk(os.path.join(geo_files_path)):
        for file in files:
            if (st_cty_code in file):
                if year in file:
                    file_list.append(os.path.join(root, file))
    return (file_list)


def geo_read(file_list):
    """
    Returns a dataframe from files associated with the state county code

    Parameters
    ----------
    file_list:
        A list of strings representing file paths
    """
    aef = pd.DataFrame()
    for file in file_list:
        tmp = load_file(file)
        aef = pd.concat([aef, tmp], axis=0)
    return (aef)

class ZGeo(BaseZRP):
    """
    This class geocodes addresses.
    
    Parameters
    ----------
    file_path: str
        Path indicating where to put artifacts folder its files (pipeline, model, and supporting data), generated during intermediate steps.
    """

    def __init__(self, file_path=None, *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.key = 'ZEST_KEY'
        self.params_dict =  kwargs
    

    def fit(self):
        return self
  
    def __majority_vote_deduplication(self, data, key):
        """
        When other deduplication methods fail we leave the most prevalent prediction
        
        Parameters
        ----------
        data: Dataframe
            Data to deduplicate
        key: string
            Key used in deduplication
        """
        data.sort_values(['NEW_SUPER_ZIP', 'COUNTYFP', 'TRACTCE', 'TRACTCE', 'BLKGRPCE'])
        data['TRACTCE'] = data.groupby(key)['TRACTCE'].transform(lambda x: x.mode()[0])
        data['BLKGRPCE'] = data.groupby(key)['BLKGRPCE'].transform(lambda x: x.mode()[0])
        data['NEW_SUPER_ZIP'] = data.groupby(key)['NEW_SUPER_ZIP'].transform(lambda x: x.mode()[0])
        data['COUNTYFP'] = data.groupby(key)['COUNTYFP'].transform(lambda x: x.mode()[0])
        data = data.drop_duplicates(keep = 'first', subset = [key])
        return data
    
    def __is_odd(self, s):
        """
        Checks if a number is odd.
        
        Parameters
        ----------
        s: string
            String with a number to be checked
        """
        try:
            return((int(str(s)[-1]) % 2)) 
        except:
            return(0)
    
    def transform(self, input_data, geo, processed, replicate, save_table=True):
        """
        Returns a DataFrame of geocoded addresses.

        :param input_data: A pd.DataFrame.
        :param geo: A String
        :param processed: A boolean.
        :param replicate: A boolean.
        :param save_table: A boolean. Tables are saved if True. Default is True.
        :return: A DataFrame
        """
        curpath = dirname(__file__)
        out_geo_path = os.path.join(curpath, '../data/processed/geo/2019')

        print("")
        # Load Data
        try:
            data = input_data.copy()
            print("   Data is loaded")
        except AttributeError:
            data = load_file(self.file_path)
            print("   Data file is loaded")
            
        prg = ProcessGeo(**self.params_dict)
        data = prg.transform(data, processed=processed, replicate=replicate)
        print("   [Start] Mapping geo data")        
        if len(geo)>2:
            file_list = geo_search(out_geo_path, self.year, geo)
            aef = geo_read(file_list)
        if len(geo) <= 2:
            aef = load_file(os.path.join(out_geo_path, f"Zest_Geo_Lookup_{self.year}_State_{geo}.parquet"))
                
        data["ZEST_FULLNAME"] = data[self.street_address]
        data['ZEST_KEY_LONG'] = data[[self.key, 'replicate_flg']].apply(lambda x: "".join(x.dropna()), axis=1)
        print("      ...merge user input & lookup table")
        print(data.shape)
        geo_df = data.merge(aef, on=["ZEST_FULLNAME"], how="left")

        ########
        #Here non-numeric HN matching could be added
        geo_df['FROMHN_numeric'] = geo_df.FROMHN.apply(lambda x: int(x) if str(x).isnumeric() else -1)
        geo_df['TOHN_numeric'] = geo_df.TOHN.apply(lambda x: int(x) if str(x).isnumeric() else -1)
        geo_df["house_numer_numeric"] = geo_df[self.house_number].apply(lambda x: int(x) if str(x).isnumeric() else -1)       
        ########
        
        geo_df["small"] = np.where(
            (geo_df.FROMHN_numeric > geo_df.TOHN_numeric),
            geo_df.TOHN_numeric,
            geo_df.FROMHN_numeric)

        geo_df["big"] = np.where(
            (geo_df.FROMHN_numeric > geo_df.TOHN_numeric),
            geo_df.FROMHN_numeric,
            geo_df.TOHN_numeric)
      
        geo_df["HN_Match"] = np.where(
            (geo_df.house_numer_numeric <= geo_df.big) &
            (geo_df.house_numer_numeric >= geo_df.small),
            1,
            0)

        geo_df["Parity_Match"] = np.where(
            (geo_df.FROMHN_numeric.apply(self.__is_odd)) == (geo_df.house_numer_numeric.apply(self.__is_odd)) ,
            1,
            0)

        geo_df["ZIP_Match_1"] = np.where(geo_df.ZEST_ZIP == geo_df[self.zip_code], 1, 0)
        geo_df["ZIP_Match_2"] = np.where(geo_df.ZCTA5CE10 == geo_df[self.zip_code], 1, 0)   
        geo_df["NEW_SUPER_ZIP"] = np.where(geo_df.ZIP_Match_1 == 1, geo_df.ZEST_ZIP, geo_df.ZCTA5CE10)
        geo_df["ZIP_Match"] = np.where(geo_df.NEW_SUPER_ZIP == geo_df[self.zip_code], 1, 0)

        print("      ...mapping")    
        #ZIP not matched
        all_keys = list(geo_df['ZEST_KEY_LONG'].unique())
        odf = geo_df.copy()
        
        geo_df = geo_df[geo_df.ZIP_Match == 1]
        zip_match_keys = list(geo_df['ZEST_KEY_LONG'].unique())
        no_zip_match_keys = list(set(all_keys) - set(zip_match_keys))
        
        df_zip_only = odf[odf['ZEST_KEY_LONG'].isin(no_zip_match_keys)]
        na_match_cols = ['BLKGRPCE', 'COUNTYFP', 'FROMHN', 'TOHN', 'TRACTCE', 'ZCTA5CE',
                         'ZCTA5CE10', 'ZEST_FULLNAME', 'ZEST_ZIP', 'small', 'big','HN_Match', 
                         'Parity_Match', 'ZIP_Match_1', 'ZIP_Match_2','NEW_SUPER_ZIP', 'ZIP_Match',
                         'FROMHN_numeric', 'TOHN_numeric', 'house_numer_numeric']
        df_zip_only[na_match_cols] = None

        df_zip_only = df_zip_only.drop_duplicates(subset = ['ZEST_KEY_LONG'])
        
        #ZIP matched, HN not match
        all_keys = list(geo_df['ZEST_KEY_LONG'].unique())
        odf = geo_df.copy()
        
        geo_df = geo_df[geo_df.HN_Match == 1]
        HN_match_keys = list(geo_df['ZEST_KEY_LONG'].unique())
        no_HN_match_keys = list(set(all_keys) - set(HN_match_keys))
        
        df_no_HN = odf[odf['ZEST_KEY_LONG'].isin(no_HN_match_keys)]       
        df_no_HN = self.__majority_vote_deduplication(df_no_HN, 'ZEST_KEY_LONG')
              
        #ZIP matched, HN matched, Parity not matched

        all_keys = list(geo_df['ZEST_KEY_LONG'].unique())
        odf = geo_df.copy()
        
        geo_df = geo_df[geo_df.Parity_Match == 1]
        parity_match_keys = list(geo_df['ZEST_KEY_LONG'].unique())
        no_parity_match_keys = list(set(all_keys) - set(parity_match_keys))
        
        df_no_parity = odf[odf['ZEST_KEY_LONG'].isin(no_parity_match_keys)]
        df_no_parity = self.__majority_vote_deduplication(df_no_parity, 'ZEST_KEY_LONG')

        #ZIP matched, HN matched, Parity matched
        df_parity = geo_df.copy()
        df_parity = self.__majority_vote_deduplication(df_parity, 'ZEST_KEY_LONG')
        
        #Merge all results
        geo_df_merged = pd.concat([df_zip_only, df_no_HN, df_no_parity, df_parity])
               
        # Create GEOIDs
        geo_df_merged['GEOID_ZIP'] = geo_df_merged['NEW_SUPER_ZIP']
        geo_df_merged["GEOID_CT"] = geo_df_merged[["STATEFP", "COUNTYFP", "TRACTCE"]].apply(lambda x: "".join(x.dropna()) if "".join(x.dropna()) != "" else None, axis=1)
        geo_df_merged["GEOID_BG"] = geo_df_merged[["GEOID_CT", "BLKGRPCE"]].apply(lambda x: "".join(x.dropna()) if "".join(x.dropna()) != "" else None, axis=1)
        
        # Choose one entry from 'replicate_flg'
        if replicate:
            geo_df_no_duplicates = geo_df_merged.sort_values(['ZEST_KEY_LONG']).groupby('ZEST_KEY').first()

        geo_validate = ValidateGeo()
        geo_validate.fit()
        geo_validators_in = geo_validate.transform(geo_df_no_duplicates)
        save_json(geo_validators_in, self.out_path, "input_geo_validator.json") 
        print("   [Completed] Validating input geo data")        

        cols_to_drop = ['BLKGRPCE', 'COUNTYFP', 'FROMHN', 'TOHN', 'TRACTCE', 'ZCTA5CE', 'ZCTA5CE10', 'ZEST_FULLNAME', 
                        'ZEST_ZIP', 'ZEST_KEY_LONG', 'replicate_flg', 'FROMHN_numeric', 'TOHN_numeric', 'house_numer_numeric']
        geo_df_no_duplicates = geo_df_no_duplicates.drop(cols_to_drop, axis = 1)
        geo_df_no_duplicates["GEOID"] = None
        geo_df_no_duplicates["GEOID"].fillna(geo_df_no_duplicates["GEOID_BG"])\
                                     .fillna(geo_df_no_duplicates["GEOID_CT"])\
                                     .fillna(geo_df_no_duplicates["GEOID_ZIP"])
        
        if save_table:
            make_directory(self.out_path)
            if self.runname is not None:
                file_name = f"Zest_Geocoded_{self.runname}_{self.year}__{geo}.parquet"
            else:
                file_name = f"Zest_Geocoded__{self.year}__{geo}.parquet"
            save_dataframe(geo_df_no_duplicates, self.out_path, file_name)
        print("   [Completed] Mapping geo data")
        return (geo_df_no_duplicates)
