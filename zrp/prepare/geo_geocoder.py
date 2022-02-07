from os.path import dirname, join, expanduser
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



def get_reduced(tmp_data):
    keep_cols = ['ZEST_KEY', 'first_name', 'middle_name', 'last_name', 
                 'house_number', 'street_address', 'city', 'state', 'zip_code',
                 'BLKGRPCE','BLKGRPCE10','COUNTYFP', 'COUNTYFP10','FROMHN', 'TOHN',
                 'LFROMADD',  'LTOADD',  'PUMACE','PUMACE10',  'RFROMADD','RTOADD', 'SIDE', 
                 'STATEFP','STATEFP10', 'TBLKGPCE',  'TRACTCE','TRACTCE10', 'TTRACTCE',
                 'ZCTA5CE', 'ZCTA5CE10', 'ZEST_FULLNAME','ZEST_KEY_COL', 'ZEST_STATE',
                 'ZEST_ZIP','GEOID_ZIP', 'GEOID_CT', 'GEOID_BG','age', 'original_ethnicity',
                 'original_race', 'original_sex',  'ethnicity', 'race', 'sex', 'source']
    na_match_cols =['BLKGRPCE', 'BLKGRPCE10','COUNTYFP', 'COUNTYFP10', 'FROMHN', 'TOHN',
                'LFROMADD',  'LTOADD',  'PUMACE','PUMACE10',  'RFROMADD','RTOADD', 'SIDE',
                'STATEFP','STATEFP10', 'TBLKGPCE',  'TRACTCE','TRACTCE10', 'TTRACTCE', 
                'ZCTA5CE', 'ZCTA5CE10', 'ZEST_FULLNAME','ZEST_STATE', 'ZEST_ZIP']
    red_bit =keep_cols + ['HN_Match', 'ZIP_Match', 'RAW_ZEST_STATEFP']

    tmp_data = tmp_data.filter(red_bit)

    geocd = tmp_data.copy()
    nomatch = tmp_data.copy()
    
    geocd = geocd[(geocd.HN_Match.astype(float) == 1)  & (geocd.ZIP_Match.astype(float) == 1)]
    geokeys = list(geocd['ZEST_KEY'].unique())
    
    nomatch = nomatch[~nomatch['ZEST_KEY'].isin(geokeys)]
    nomatch = nomatch.drop_duplicates('ZEST_KEY')

    geocd['TRACTCE'] = geocd.groupby('ZEST_KEY')['TRACTCE'].transform(lambda x: x.value_counts().idxmax())
    geocd['BLKGRPCE'] = geocd.groupby('ZEST_KEY')['BLKGRPCE'].transform(lambda x: x.value_counts().idxmax())
    geocd['ZCTA5CE'] = geocd.groupby('ZEST_KEY')['ZCTA5CE'].transform(lambda x: x.value_counts().idxmax())
    geocd['COUNTYFP'] = geocd.groupby('ZEST_KEY')['COUNTYFP'].transform(lambda x: x.value_counts().idxmax())

    geocd = geocd.drop_duplicates('ZEST_KEY')
    geocd["GEOID_CT"] = geocd[["RAW_ZEST_STATEFP", "COUNTYFP", "TRACTCE"]].apply(lambda x: "".join(x.dropna()), axis=1)
    geocd["GEOID_BG"] = geocd[["GEOID_CT", "BLKGRPCE"]].apply(lambda x: "".join(x.dropna()), axis=1)
    geocd = geocd.set_index('ZEST_KEY')

    if len(nomatch)>1:
        nomatch[na_match_cols] = None
        nomatch = nomatch.set_index('ZEST_KEY')
        data_out = pd.concat([geocd, nomatch])
        data_out["GEOID_ZIP"] = data_out["ZCTA5CE"].fillna(data_out.zip_code)
        
        data_out = data_out.filter(keep_cols)
    else: 
        data_out = geocd.filter(keep_cols)
        data_out["GEOID_ZIP"] = data_out["ZCTA5CE"].fillna(data_out.zip_code)
    return(data_out)



def geo_search(geo_files_path, year, st_cty_code):
    """
    Returns a list of files associated with the state county code
    """
    file_list = []
    for root, dirs, files in os.walk(os.path.join(geo_files_path)):
        for file in files:
            if (st_cty_code in file):
                if year in file:
                    file_list.append(os.path.join(root,file))
    return(file_list)
        
    
def geo_read(file_list):
    """
    Returns a dataframe from files associated with the state county code
    """
    aef = pd.DataFrame()
    for file in file_list:
        tmp = load_file(file_path)
        aef = pd.concat([aef, tmp], axis=0)
    return(aef)    
    
def geo_zoom(geo_df):
    """
    Matches census tract
    
    Parameters
    ----------
    geo_df: pd.DataFrame
        Dataframe with geo data
    """
    geo_df = geo_df[(geo_df.HN_Match == 1) &
                    (geo_df.ZIP_Match == 1)]
    return(geo_df)
    
def geo_range(geo_df):
    """
    Define house number range indicators
    
    Parameters
    ----------
    geo_df: pd.DataFrame
        Dataframe with geo data
    """
    geo_df["small"] = np.where(
        geo_df.FROMHN > geo_df.TOHN,
        geo_df.TOHN,
        geo_df.FROMHN)
    
    geo_df["big"] = np.where(
        geo_df.FROMHN > geo_df.TOHN,
        geo_df.FROMHN,
        geo_df.TOHN)
    geo_df["small"] = pd.to_numeric(geo_df["small"], errors="coerce").fillna(0).astype(np.int64) 
    geo_df["big"] = pd.to_numeric(geo_df["big"], errors="coerce").fillna(0).astype(np.int64)
    return(geo_df)    

class ZGeo(BaseZRP):
    """
    This class geocodes addresses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key ='ZEST_KEY'      
            
    def fit(self):
        return self
    
    
    def geo_match(self, geo_df):
        """
        Returns match indicators 
        """
        geo_df["HN_Match"] = np.where(
            (geo_df[self.house_number].astype(float) <= geo_df.big.astype(float)) &
            (geo_df[self.house_number].astype(float) >= geo_df.small.astype(float)),
            1,
            0)
        geo_df["ZIP_Match"] = np.where(geo_df.ZEST_ZIP.astype(float) == geo_df[self.zip_code].astype(float), 1, 0)
        # update use l/rfrom/toadd + side
        geo_df["Parity_Match"] = np.where(geo_df.small.astype(float) % 2 == geo_df[self.house_number].astype(float) % 2, 1, 0)
        return(geo_df)    
    
    def transform(self, input_data, geo, processed, replicate, save_table=True):
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
            
        if self.key in data.columns:
            data["ZEST_KEY_COL"] = data[self.key]
        else:
            data["ZEST_KEY_COL"] = data.index
            
        prg = ProcessGeo()
        data = prg.transform(data, processed=processed, replicate=replicate)  
        print("   [Start] Mapping geo data")
        data[self.house_number] = pd.to_numeric(data[self.house_number], errors="coerce").fillna(0).astype(np.int64) 
        state = most_common(list(data[self.state].unique()))
        
            
        if len(geo)>2:
            file_list = geo_search(out_geo_path, self.year, geo)
            aef = geo_read(file_list)
        if len(geo)<=2:
            aef = load_file(os.path.join(out_geo_path, f"Zest_Geo_Lookup_{self.year}_State_{geo}.parquet"))

        
        
        data["ZEST_FULLNAME"] = data[self.street_address]
        print("      ...merge user input & lookup table")
        geo_df = aef.merge(data, on="ZEST_FULLNAME", how="right")
        geo_df = geo_range(geo_df)
        geo_df = self.geo_match(geo_df)
        print("      ...mapping")
        all_keys = list(geo_df[self.key].unique())
        odf = geo_df.copy()
        geo_df = geo_zoom(geo_df)
        geocoded_keys = list(geo_df[self.key].unique())
        add_na_keys = list(set(all_keys) - set(geocoded_keys))
        odf = odf[odf[self.key].isin(add_na_keys)]
        
        geo_df = pd.concat([geo_df, odf])
        
        geo_df = get_reduced(geo_df)
        
        
        if save_table:
            make_directory()
            if self.runname is not None:
                file_name = f"Zest_Geocoded_{self.runname}_{self.year}__{geo}.parquet"
            else:
                file_name = f"Zest_Geocoded__{self.year}__{geo}.parquet"
            save_dataframe(geo_df, self.out_path, file_name)        
        print("   [Completed] Mapping geo data")
        return(geo_df)

