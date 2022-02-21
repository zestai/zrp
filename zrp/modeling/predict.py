from os.path import dirname, join, expanduser
from zrp.prepare.preprocessing import *
from zrp.prepare.geo_geocoder import *
from zrp.prepare.acs_mapper import *
from zrp.prepare.base import BaseZRP
from zrp.prepare.utils import *
from zrp.validate import *
import pandas as pd
import numpy as np
import warnings
import glob
import json
import tqdm
import sys
import os
import re
import pickle
import joblib
import surgeo
import xgboost

import warnings
warnings.filterwarnings(action='ignore')

from zrp.modeling import src
from zrp.modeling.src.app_preprocessor import HandleCompoundNames
from zrp.modeling.src.acs_scaler import CustomRatios
from zrp.modeling.src.app_fe import AppFeatureEngineering, NameAggregation


class PredictPass(BaseZRP):
    """
    Generates proxies

    Parameters
    ----------
    pipe_path: str
        Folder path to directory containing pipeline
    """

    def __init__(self, pipe_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        """
        Processes input data and generates BISG predictions.

        Parameters
        -----------
        input_data: pd.DataFrame
            Dataframe to be transformed
        """
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
        proxies = pd.DataFrame({"AAPI": None, "AIAN": None, "BLACK": None,
                                "HISPANIC": None, "WHITE": None, f"{self.race}_proxy": None}, index = data.index)
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
    """
    Wrapper function for Bayesian Improved Surname Geocoding

    Generates proxies using BISG algorithm.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def fit(self, input_data):
        """

        Parameters
        -----------
        input_data: pd.DataFrame
            Dataframe to be transformed
        """        
        if (self.last_name not in input_data.columns):
            raise ValueError('Last name needs to be provided when initializing this class. Please provide last name data in a column named "last_name" or set the custom name of the last name column in the data')
        if self.zip_code not in input_data.columns:
            raise ValueError('Zip or postal code name needs to be provided when initializing this class. Please provide zip code data in a column named "zip_code" or set the custom name of the zip code column in the data')
        return self
    
    def transform(self, input_data):
        """
        Processes input data and generates BISG predictions.

        Parameters
        -----------
        data: pd.Dataframe
            Dataframe to be transformed
        """
        df = input_data.copy()
        df = df[~df.index.duplicated(keep='first')]

        df = df.filter([self.last_name, self.zip_code, self.census_tract])
        df[self.zip_code] = np.where((df[self.zip_code] == "") | (df[self.zip_code] == " ") | (df[self.zip_code] == None) | (df[self.zip_code].str.len()<5), "99999", df[self.zip_code])
        
        bisg = surgeo.SurgeoModel()
        bisg_results = bisg.get_probabilities(names  = df[self.last_name].reset_index(drop=True),
                                              geo_df = df[self.zip_code].astype(int))
        combo = df.reset_index().merge(bisg_results, 
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
        combo = combo.set_index(self.key)
        combo = combo[~combo.index.duplicated(keep='first')]
        
        # Generate proxy at threshold
        subset = combo.filter(['WHITE', 'BLACK', 'AAPI', 'AIAN', 'HISPANIC'
                       ])
        identifiedRaces = subset.idxmax(axis=1)
        combo[f"{self.race}_proxy"] = identifiedRaces
        combo['source_bisg'] = 1
        proxies = combo.filter(["AAPI", "AIAN", "BLACK", "HISPANIC",
                         "WHITE", f"{self.race}_proxy", "source_bisg"])   
        return(proxies)



class ZRP_Predict_ZipCode(BaseZRP):
    """
    Generates proxies using model trained on zip code features.

    Parameters
    ----------
    pipe_path: str
        Folder path to directory containing pipeline
    """

    def __init__(self, pipe_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        """
        Processes input data and generates ZRP proxy predictions.

        Parameters
        -----------
        input_data: pd.DataFrame
            Dataframe to be transformed
        """

        src_path = os.path.join(self.pipe_path, "zip_code")
        sys.path.append(src_path)        
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
            
        numeric_cols = list(data.filter(regex='^B|^C16').columns)
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce') 
        
        model = xgboost.Booster()
        model.load_model(os.path.join(src_path,"model.txt"))
#         model = pd.read_pickle(os.path.join(src_path,"model.pkl") )
        pipe = pd.read_pickle(os.path.join(src_path, "pipe.pkl") )
        
        
        data = validate_case(data, self.key, self.last_name)
        fe_data = pipe.transform(data)
        fe_data = validate_drop(fe_data)
        fe_matrix = xgboost.DMatrix(fe_data)
        
        proxies = pd.DataFrame(model.predict(fe_matrix), index = fe_data.index)
        proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies[f"{self.race}_proxy"] = proxies.idxmax(axis=1)
        proxies['source_zip_code'] = 1
        return(proxies)
        


class ZRP_Predict_BlockGroup(BaseZRP):
    """
    Generates proxies using model trained on block group features.

    Parameters
    ----------
    pipe_path: str
        Folder path to directory containing pipeline
    """

    def __init__(self, pipe_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        """
        Processes input data and generates ZRP proxy predictions.

        Parameters
        -----------
        input_data: pd.DataFrame
            Dataframe to be transformed
        """        

        src_path = os.path.join(self.pipe_path,"block_group")
        sys.path.append(src_path)
        
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
            
        numeric_cols = list(data.filter(regex='^B|^C16').columns)
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        model = xgboost.Booster()
        model.load_model(os.path.join(src_path,"model.txt"))
#         model = pd.read_pickle(os.path.join(src_path,"model.pkl") )
        pipe = pd.read_pickle(os.path.join(src_path, "pipe.pkl") )
        
        data = validate_case(data, self.key, self.last_name)
        fe_data = pipe.transform(data)
        fe_data = validate_drop(fe_data)
        fe_matrix = xgboost.DMatrix(fe_data)
        
        proxies = pd.DataFrame(model.predict(fe_matrix), index = fe_data.index)
        proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies[f"{self.race}_proxy"] = proxies.idxmax(axis=1)
        proxies['source_block_group'] = 1
        return(proxies)
        



class ZRP_Predict_CensusTract(BaseZRP):
    """
    Generates proxies using model trained on census tract features.

    Parameters
    ----------
    pipe_path: str
        Folder path to directory containing pipeline
    """

    def __init__(self, pipe_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_path = pipe_path
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        """
        Processes input data and generates ZRP proxy predictions.

        Parameters
        -----------
        input_data: pd.DataFrame
            Dataframe to be transformed
        """        

        src_path = os.path.join(self.pipe_path,"census_tract")
        sys.path.append(src_path)
        
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.proxy_data)
            
        numeric_cols = list(data.filter(regex='^B|^C16').columns)
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce') 

        model = xgboost.Booster()
        model.load_model(os.path.join(src_path,"model.txt"))        
#         model = pd.read_pickle(os.path.join(src_path,"model.pkl") )
        pipe = pd.read_pickle(os.path.join(src_path, "pipe.pkl") )
        
        data = validate_case(data, self.key, self.last_name)
        fe_data = pipe.transform(data)
        fe_data = validate_drop(fe_data)
        fe_matrix = xgboost.DMatrix(fe_data)
        
        proxies = pd.DataFrame(model.predict(fe_matrix), index = fe_data.index)
        proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies[f"{self.race}_proxy"] = proxies.idxmax(axis=1)
        proxies['source_census_tract'] = 1
        return(proxies)
    


class ZRP_Predict(BaseZRP):
    """
    Generates race proxies.
    Attempts to predict on census tract, then block group, then zip code based on which level ACS data is found for. If
    Geo level data is unattainable, the BISG proxy is computed. No prediction returned if BISG cannot be computed either.

    Parameters
    ----------
    pipe_path: str
        Folder path to directory containing pipeline
    file_path: str
        Path indicating where to put artifacts folder its files (pipeline, model, and supporting data), generated during intermediate steps.
    """

    def __init__(self, pipe_path, file_path=None, *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.pipe_path = pipe_path
        self.params_dict = kwargs
        self.census_tract = 'GEOID_CT'
        self.block_group = 'GEOID_BG'
        self.zip_code = 'GEOID_ZIP'
        
        
    def fit(self, data):
        assert xgboost.__version__ == "1.0.2", "XGBoost version does not match requirements, required version is 1.0.2"  
        data_cols =  list(data.columns)
        self.required_cols = [self.first_name, self.middle_name, self.last_name, "GEOID", "B01003_001"]
        val_na = is_missing(data, self.required_cols)
        if val_na:
            assert True, f"Missing required data {val_na}"        

        print("   [Start] Validating pipeline input data")
        validator = ValidateInput()
        validator.fit()
        validator_in = validator.transform(data)
        save_json(validator_in, self.out_path, "input_predict_validator.json")
        print("   [Completed] Validating pipeline input data")
        print("")        
        return self
    
    def transform(self, input_data, save_table=True):
        """
        Processes input data and generates ZRP proxy predictions.

        Parameters
        -----------
        input_data: pd.DataFrame
            Dataframe to be transformed
        """        

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
        else:
            raise KeyError("Processed data is required for ZRP_Predict. Please use EnginetoPredict if supplying feature engineered data to generate predictions" )
            
        out_list = []
        if not df_0.empty:
            zrp_bg = ZRP_Predict_BlockGroup(self.pipe_path, **self.params_dict)
            out_0 = zrp_bg.transform(df_0.filter(flb))
            out_list.append(out_0)    
        if not df_1.empty:
            zrp_ct = ZRP_Predict_CensusTract(self.pipe_path, **self.params_dict)
            out_1 = zrp_ct.transform(df_1.filter(flc))
            out_list.append(out_1)    
        if not df_2.empty:
            zrp_zp = ZRP_Predict_ZipCode(self.pipe_path, **self.params_dict)
            out_2 = zrp_zp.transform(df_2.filter(flz))
            out_list.append(out_2)    
        if not df_3.empty:
            bisgw = BISGWrapper(**self.params_dict)
            bisgw.fit(df_3)
            out_3 = bisgw.transform(df_3)
            out_list.append(out_3)  
            
        proxies_out = pd.concat(out_list)

        source_cols = list(set(proxies_out.columns).intersection(set([
            'source_block_group', 'source_census_tract',
            'source_zip_code', 'source_bisg'])))
        proxies_out[source_cols] = proxies_out[source_cols].fillna(0)
        proxies_out = proxies_out.sort_values(source_cols)
        
        if save_table:
            make_directory(self.out_path)
            file_name = f"proxy_output.feather"
            save_feather(proxies_out,
                         self.out_path,
                         file_name)      
        
        return(proxies_out)

    
    
class FEtoPredict(BaseZRP):
    """
    Generates proxies from feature engineered data.

    Parameters
    ----------
    pipe_path: str
        Folder path to directory containing pipeline
    pipe_type: str, (default='census_tract')
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
        """
        Processes input data and generates ZRP proxy predictions.

        Parameters
        -----------
        input_data: pd.DataFrame
            Dataframe to be transformed
        """
        model = xgboost.Booster()
        model.load_model(os.path.join(src_path,"model.txt"))
#         model = pd.read_pickle(os.path.join(self.pipe_path, f"{pipe_type}/model.pkl") )
        # Load Data
        assert not input_data.empty, "Feature engineered data is empty or missing. Please provide the feature engineered data as `input_data` to generate predictions."
        fe_data = input_data.copy()
        fe_matrix = xgboost.DMatrix(fe_data)

        proxies = pd.DataFrame(model.predict(fe_matrix), index = fe_data.index)
        proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies[f"{self.race}_proxy"] = proxies.idxmax(axis=1)
        proxies[f'source_{pipe_type}'] = 1        

        
        if save_table:
            make_directory(self.out_path)
            file_name = f"{self.pipe_type}_proxy_output.feather"
            save_feather(proxies,
                         self.out_path,
                         file_name)
            
        return(proxies)
    
