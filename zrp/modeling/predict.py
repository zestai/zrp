from os.path import dirname, join, expanduser
from zrp.prepare.preprocessing import *
from zrp.prepare.geo_geocoder import *
from zrp.prepare.acs_mapper import *
from zrp.prepare.base import BaseZRP
from zrp.prepare.utils import *
import pandas as pd
import numpy as np
import warnings
import glob
import json
import tqdm
import sys
import os
import re
import pycm
import pickle
import joblib
import surgeo



from zrp.modeling import src
from zrp.modeling.src.app_preprocessor import HandleCompoundNames
from zrp.modeling.src.acs_scaler import CustomRatios
from zrp.modeling.src.app_fe import AppFeatureEngineering, NameAggregation
from zrp.modeling.src.set_key import SetKey


class PredictPass(BaseZRP):
    """
    Generates proxies
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
    key: str 
        Key to set as index. If not provided, a key will be generated.
    race: str
        Name of race column 
    proxy_data: str, pd.Series, or pd.DataFrame
        File path to proxy data
    ground_truth: str, pd.Series, or pd.DataFrame
        File path to ground truth data
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
            
        if self.proxy =='labels':
            proxies = pd.DataFrame({'race' : None}, index = fe_data.index)
        if self.proxy =='probs':
            proxies = pd.DataFrame({"AAPI":None, "AIAN":None, "BLACK":None,
                                    "HISPANIC": None, "WHITE": None}, index = fe_data.index)
        return(proxies) 

def validate_case(data, key, last_name):
    df = data.copy()
    new_row = df.tail(1).reset_index(drop=False)
    new_row[key] = "validate_case_001"
    new_row[last_name] = "SMITH JONES"
    new_row = new_row.set_index(key)
    df = pd.concat([df, new_row])
    return(df)
    
def validate_drop(data):
    data = data.drop('validate_case_001')
    return(data) 


class BISGWrapper(BaseZRP):
    """Wrapper function for bisg"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def fit(self, data):
        if self.last_name not in data.columns:
            raise ValueError('Last name needs to be provided when initializing this class. Please provide last name data in a column named "last_name" or set the custom name of the last name column in the data')
        if self.zip_code not in data.columns:
            raise ValueError('Zip or postal code name needs to be provided when initializing this class. Please provide zip code data in a column named "zip_ode" or set the custom name of the zip code column in the data')
        return self
    
    def transform(self, data):
        df = data.copy()
        df = df[df.index.duplicated(keep='last')]
        df = df.filter([self.last_name, self.zip_code, self.census_tract])
        bisg = surgeo.SurgeoModel()
        bisg_results = bisg.get_probabilities(names  = df[self.last_name] ,  geo_df = df[self.zip_code].astype(int))
        
        combo = df.merge(bisg_results, 
                           how="left", 
                           left_on=[self.last_name, self.zip_code], 
                           right_on=["name", "zcta5"]
                          )
        combo.drop(columns = ['zcta5', 'name'], inplace=True)
        
        combo.rename(columns = {
            'white': 'WHITE',
            'black': 'BLACK',
            'api': 'AAPI',
            'native': 'AIAN',
            'multiple': 'OTHER',
            'hispanic': 'HISPANIC'
        }, inplace=True) 
        
        # Generate proxy at threshold
        subset = combo[['WHITE', 'BLACK', 'AAPI', 'AIAN', 'OTHER', 'HISPANIC'
                       ]]
        identifiedRaces = subset.idxmax(axis=1)
        combo[self.race] = identifiedRaces
        combo['source_bisg'] = 1
        if self.proxy =='labels':
            proxies = combo[[self.race, "source_bisg"]]
        if self.proxy =='probs':
            proxies = combo[['WHITE', 'BLACK', 'AAPI', 'AIAN', 'OTHER', 'HISPANIC', "source_bisg"]]
        return(proxies)


class ZRP_Predict_ZipCode(BaseZRP):
    """
    Generates proxies
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
    key: str 
        Key to set as index. If not provided, a key will be generated.
    race: str
        Name of race column 
    proxy_data: str, pd.Series, or pd.DataFrame
        File path to proxy data
    ground_truth: str, pd.Series, or pd.DataFrame
        File path to ground truth data
    """

    def __init__(self, pipe_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        src_path = os.path.join(self.pipe_path, "zip_code")
        sys.path.append(src_path)        
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
            
        numeric_cols = list(data.filter(regex='^B|^C16').columns)
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce') 
        
        model = pd.read_pickle(os.path.join(src_path,"model.pkl") )
        pipe = pd.read_pickle(os.path.join(src_path, "pipe.pkl") )
        
        data = validate_case(data, self.key, self.last_name)
        fe_data = pipe.transform(data)
        fe_data = validate_drop(fe_data)
        
        if self.proxy =='labels':
            proxies = pd.DataFrame({'race' : model.predict(fe_data)}, index = fe_data.index)
       
        if self.proxy =='probs':
            proxies = pd.DataFrame(model.predict_proba(fe_data), index = fe_data.index)
            proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies['source_zip_code'] = 1
        return(proxies)
        


class ZRP_Predict_BlockGroup(BaseZRP):
    """
    Generates proxies
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
    key: str 
        Key to set as index. If not provided, a key will be generated.
    race: str
        Name of race column 
    proxy_data: str, pd.Series, or pd.DataFrame
        File path to proxy data
    ground_truth: str, pd.Series, or pd.DataFrame
        File path to ground truth data
    """

    def __init__(self, pipe_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        src_path = os.path.join(self.pipe_path,"block_group")
        sys.path.append(src_path)
        
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
            
        numeric_cols = list(data.filter(regex='^B|^C16').columns)
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce') 
        
        model = pd.read_pickle(os.path.join(src_path,"model.pkl") )
        pipe = pd.read_pickle(os.path.join(src_path, "pipe.pkl") )
        
        data = validate_case(data, self.key, self.last_name)
        fe_data = pipe.transform(data)
        fe_data = validate_drop(fe_data)
        
        if self.proxy =='labels':
            proxies = pd.DataFrame({'race' : model.predict(fe_data)}, index = fe_data.index)
       
        if self.proxy =='probs':
            proxies = pd.DataFrame(model.predict_proba(fe_data), index = fe_data.index)
            proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies['source_block_group'] = 1
        return(proxies)
        



class ZRP_Predict_CensusTract(BaseZRP):
    """
    Generates proxies
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
    key: str 
        Key to set as index. If not provided, a key will be generated.
    race: str
        Name of race column 
    proxy_data: str, pd.Series, or pd.DataFrame
        File path to proxy data
    ground_truth: str, pd.Series, or pd.DataFrame
        File path to ground truth data
    """

    def __init__(self, pipe_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        src_path = os.path.join(self.pipe_path,"census_tract")
        sys.path.append(src_path)
        
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
            
        numeric_cols = list(data.filter(regex='^B|^C16').columns)
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce') 
        
        model = pd.read_pickle(os.path.join(src_path,"model.pkl") )
        pipe = pd.read_pickle(os.path.join(src_path, "pipe.pkl") )
        
        data = validate_case(data, self.key, self.last_name)
        fe_data = pipe.transform(data)
        fe_data = validate_drop(fe_data)
        
        if self.proxy =='labels':
            proxies = pd.DataFrame({'race' : model.predict(fe_data)}, index = fe_data.index)
        if self.proxy =='probs':
            proxies = pd.DataFrame(model.predict_proba(fe_data), index = fe_data.index)
            proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies['source_census_tract'] = 1
        return(proxies)
    
    



class ZRP_Predict(BaseZRP):
    """
    Generates proxies
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with processed user data
    pipe_path: str
        Folder path to directory containing pipeline
    """

    def __init__(self, pipe_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
        
    def fit(self):
        return self
    
    def transform(self, input_data, save_table=True):
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
        cur_path = dirname(__file__)

        flb = load_json(f'{cur_path}/feature_list_bg.json')
        flc = load_json(f'{cur_path}/feature_list_ct.json')
        flz = load_json(f'{cur_path}/feature_list_zp.json')


        if 'acs_source' in data.columns:    
            df_0 = data[data['acs_source'] == 'BG']
            keys_0 = list(df_0.index.values)
            df_1 = data[~(data.index.isin(keys_0)) 
                         & (data['acs_source'] == 'CT')]
            keys_1 = keys_0 + list(df_1.index.values)
            df_2 = data[~(data.index.isin(keys_1)) 
                          & (data['acs_source'] == 'ZIP')]
            keys_2 = keys_1 + list(df_2.index.values)
            df_3 = data[~(data.index.isin(keys_2))
                                & (data['acs_source'].isna())]
            keys_3 = keys_2 + list(df_3.index.values)
            df_4 = data[~(data.index.isin(keys_3)) 
                            & (data['acs_source'].isna())]
        else:
            raise KeyError("Processed data is required for ZRP_Predict. Please use EnginetoPredict if supplying feature engineered data to generate predictions" )
            
        out_list = []
        if not df_0.empty:
            zrp_bg = ZRP_Predict_BlockGroup(self.pipe_path)
            out_0 = zrp_bg.transform(df_0.filter(flb))
            out_list.append(out_0)    
        if not df_1.empty:
            zrp_ct = ZRP_Predict_CensusTract(self.pipe_path)
            out_1 = zrp_ct.transform(df_1.filter(flc))
            out_list.append(out_1)    
        if not df_2.empty:
            zrp_zp = ZRP_Predict_ZipCode(self.pipe_path)
            out_2 = zrp_zp.transform(df_2.filter(flz))
            out_list.append(out_2)    
        if not df_3.empty:
            bisgw = BISGWrapper()
            bisgw.fit(df_3)
            out_3 = bisgw.transform(df_3)
            out_list.append(out_3)    
        if not df_4.empty:
            pass_o = PredictPass()
            out_4 = pass_o.transform(df_4)
            out_list.append(out_4)    
        
        proxies_out = pd.concat(out_list)
        
        if save_table:
            make_directory()
            file_name = f"proxy_{self.proxy}.feather"
            save_feather(proxies_out,
                         self.out_path,
                         file_name)      
        
        return(proxies_out)

    
    
class FEtoPredict(BaseZRP):
    """
    Generates proxies
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with processed user data
    pipe_path: str
        Folder path to directory containing pipeline
    pip_type: str, (default='census_tract')
        Type of pipeline that generated the engineered data.
        Options: 'block_group', 'census_tract', or 'zip_code'
    """

    def __init__(self, pipe_path, pipe_type='census_tract', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        self.pipe_type = pipe_type
        
        
    def fit(self):
        return self
    
    def transform(self, input_data, save_table=True):

        # Load Data
        try:
            fe_data = input_data.copy()
        except AttributeError:
            fe_data = load_file(self.proxy_data)
            
        model = pd.read_pickle(os.path.join(self.pipe_path, f"{pipe_type}/model.pkl") )
         
        if self.proxy =='labels':
            proxies = pd.DataFrame({'race' : model.predict(fe_data)}, index=fe_data.index)
        if self.proxy =='probs':
            proxies = pd.DataFrame(model.predict_proba(fe_data), index=fe_data.index)
            proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies['source_census_tract'] = 1        
        
        proxies_out = proxies.copy()
        
        if save_table:
            make_directory()
            file_name = f"{self.pipe_type}_proxy_{self.proxy}.feather"
            save_feather(proxies_out,
                         self.out_path,
                         file_name)
            
        return(proxies_out)
    
