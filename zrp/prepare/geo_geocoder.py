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
            return (int(str(s)[-1]) % 2)
        except:
            return 0
    
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
        geo_df = data.merge(aef, on=["ZEST_FULLNAME"], how="left")

        geo_df['FROMHN_RIGHT'] = geo_df['FROMHN_RIGHT'].replace('nan', np.nan).fillna(-2).astype(float).astype(int)
        geo_df['TOHN_RIGHT'] = geo_df['TOHN_RIGHT'].replace('nan', np.nan).fillna(-2).astype(float).astype(int)
        geo_df["house_number_RIGHT"] = geo_df[self.house_number+'_RIGHT'].replace("", np.nan).fillna(-1).astype(float).astype(int)
            
        geo_df["HN_Match"] = np.where(
            (geo_df.house_number_LEFT == geo_df.FROMHN_LEFT) &
            (geo_df.house_number_RIGHT >= geo_df.FROMHN_RIGHT) &
            (geo_df.house_number_RIGHT <= geo_df.TOHN_RIGHT),
            1,
            0)

        geo_df["Parity_Match"] = np.where(
            (geo_df.FROMHN_RIGHT.apply(self.__is_odd)) == (geo_df.house_number_RIGHT.apply(self.__is_odd)) ,
            1,
            0)

        geo_df['ZCTA5CE'] = geo_df['ZCTA5CE'].replace('None', np.nan)
        geo_df["NEW_SUPER_ZIP"] = np.where(geo_df.ZCTA5CE == geo_df[self.zip_code], geo_df.ZCTA5CE, geo_df.ZEST_ZIP)
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
        na_match_cols = ['BLKGRPCE', 'FROMHN', 'TOHN', 'NEW_SUPER_ZIP', 'ZIP_Match']
        
        df_no_HN = self.__majority_vote_deduplication(df_no_HN, 'ZEST_KEY_LONG')
        df_no_HN[na_match_cols] = None    
            
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
        geo_df_merged["GEOID_CT"] = geo_df_merged[["STATEFP", "COUNTYFP", "TRACTCE"]].apply(lambda x: "".join(x.dropna()) if "".join(x.dropna()) != "" else None, axis=1)
        geo_df_merged["GEOID_CT"] = geo_df_merged["GEOID_CT"].apply(lambda x: x if x is None or len(x) == 11 else None)
        geo_df_merged["GEOID_BG"] = geo_df_merged[["GEOID_CT", "BLKGRPCE"]].apply(lambda x: "".join(x.dropna()) if "".join(x.dropna()) != "" else None, axis=1)
        geo_df_merged["GEOID_BG"] = geo_df_merged["GEOID_BG"].apply(lambda x: x if x is None or len(x) == 12 else None)
        geo_df_merged["GEOID_ZIP"] = np.where(geo_df_merged["ZCTA5CE"].notna(), geo_df_merged["ZCTA5CE"], geo_df_merged[self.zip_code])
        
        # Choose one entry from 'replicate_flg'
        if replicate:
            geo_df_no_duplicates = geo_df_merged.sort_values(['ZEST_KEY_LONG']).groupby('ZEST_KEY').first()

        geo_validate = ValidateGeo()
        geo_validate.fit()
        geo_validators_in = geo_validate.transform(geo_df_no_duplicates)
        save_json(geo_validators_in, self.out_path, "input_geo_validator.json") 
        print("   [Completed] Validating input geo data")        

        cols_to_drop = ['BLKGRPCE', 'COUNTYFP', 'FROMHN', 'TOHN', 'TRACTCE', 'ZCTA5CE', 'ZCTA5CE10', 'ZEST_FULLNAME',
                        'ZEST_ZIP', 'ZEST_KEY_LONG', 'replicate_flg', 'FROMHN_LEFT', "FROMHN_RIGHT", 'TOHN_LEFT',
                        'TOHN_RIGHT', 'HN_Match', 'NEW_SUPER_ZIP', 'PARITY', 'Parity_Match', 'STATEFP', 'ZIP_Match']


        geo_df_no_duplicates = geo_df_no_duplicates.drop(cols_to_drop, axis = 1)
        geo_df_no_duplicates["GEOID"] = None
        
        if save_table:
            make_directory(self.out_path)
            #Finding next name
            if self.runname is not None:
                file_like = f"Zest_Geocoded_{self.runname}__{self.year}__{geo}"
            else:
                file_like = f"Zest_Geocoded__{self.year}__{geo}"
            i = 1
            for file in os.listdir(self.out_path):
                if file_like in file:
                    number = int(file[len(file_like)+1:-8])
                    if number >= i:
                        i = number + 1
            file_name = f'{file_like}_{i}.parquet'
            
            save_dataframe(geo_df_no_duplicates, self.out_path, file_name)
        print("   [Completed] Mapping geo data")
        return geo_df_no_duplicates