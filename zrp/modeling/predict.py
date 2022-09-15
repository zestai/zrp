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
        race_col_ov = ['WHITE', 'BLACK', 'AAPI', 'AIAN', 'HISPANIC']
        subset = combo.filter(race_col_ov)
        subset = subset[race_col_ov].div(subset[race_col_ov].sum(axis=1), axis=0)              
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
    
    def transform(self, input_data, geo_only=False):
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
#         proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies.columns = sorted(pipe.steps[2][1].mlb_columns)
        proxies[f"{self.race}_proxy"] = proxies.idxmax(axis=1)
        if not geo_only:
            proxies['source_zrp_zip_code'] = 1
        else:
            proxies['source_zrp_zip_code_geo_only'] = 1
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
    
    def transform(self, input_data, geo_only=False, name_only=False):
        """
        Processes input data and generates ZRP proxy predictions.

        Parameters
        -----------
        input_data: pd.DataFrame
            Dataframe to be transformed.
        geo_only: Boolean
            Whether or not the predictions will be based on block group geo features only.
        name_only: Boolean
            Whether or not the predictions will be based on name features only.
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
#         proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies.columns = sorted(pipe.steps[2][1].mlb_columns)
        proxies[f"{self.race}_proxy"] = proxies.idxmax(axis=1)
        
        if not geo_only and not name_only:
            proxies['source_zrp_block_group'] = 1
        elif geo_only:
            proxies['source_zrp_block_group_geo_only'] = 1
        else:
            proxies['source_zrp_name_only'] = 1
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
    
    def transform(self, input_data, geo_only=False):
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
#         proxies.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        proxies.columns = sorted(pipe.steps[2][1].mlb_columns)
        proxies[f"{self.race}_proxy"] = proxies.idxmax(axis=1)
        if not geo_only:
            proxies['source_zrp_census_tract'] = 1
        else:
            proxies['source_zrp_census_tract_geo_only'] = 1
        return(proxies)
    


class ZRP_Predict(BaseZRP):
    """
    Generates race proxies.
    Attempts to predict on block group, then census tract, then zip code based on which level ACS data is found for. If
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
        if xgboost.__version__ != "1.0.2":
            raise AssertionError("XGBoost version does not match requirements, required version is 1.0.2")

        data_cols =  list(data.columns)
        self.required_cols = [self.first_name, self.middle_name, self.last_name, "GEOID", "B01003_001"]
        val_na = is_missing(data, self.required_cols)
        if val_na:
            raise ValueError(f"Missing required data {val_na}")     

        print("   [Start] Validating pipeline input data")
        validator = ValidateInput()
        validator.fit()
        validator_in = validator.transform(data)
        save_json(validator_in, self.out_path, "input_predict_validator.json")
        print("   [Completed] Validating pipeline input data")
        print("")
        return self
    
    def validate_data_has_names(self, data, is_input = True):
        """Identifies which rows have missing "first_name", "middle_name", or "last_name". Returns a df with columns for the 3 names and rows taking boolean
        values for whether or not this record contains said name or not.
        
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        is_input: bool
            Indicator if validating raw input data
        """
        name_cols =  [self.first_name, self.middle_name, self.last_name]
        
        data = data.copy()
        data["has_first_name"] = 0
        data["has_middle_name"] = 0
        data["has_last_name"] = 0
        has_name_columns = ["has_first_name", "has_middle_name", "has_last_name"]
        
        for name_col, has_name_col in zip(name_cols, has_name_columns):
            data.loc[~((data[name_col].astype(str).str.upper() == "NONE")
                                | (data[name_col].astype(str).str.upper() == " ")
                                | (data[name_col].isna()))
                , has_name_col] = 1 
#         return(data[has_name_columns])
        return(data)

    def standard_target_classes(self):
        """
        Checks if models were trained on standard set of target classes. Returns True for standard target classes and False for non-standard target classes. 
        
        Parameter
        ---------
        """
        model_types = ['block_group', 'census_tract', 'zip_code']
        all_model_cols = set()
        for model_type in model_types:
            src_path = os.path.join(self.pipe_path, model_type)
            sys.path.append(src_path)
            pipe = pd.read_pickle(os.path.join(src_path, "pipe.pkl"))
            cols = set(pipe.steps[2][1].mlb_columns)
            all_model_cols = all_model_cols.union(cols)
        standard_cols = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]
        target_is_standard = set(all_model_cols) == set(standard_cols)
        return(target_is_standard)

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
        
        # Determine records with names provided
        data = self.validate_data_has_names(input_data)
                    
        if 'acs_source' in data.columns:  
            # Select records for zrp geo and name proxy
            df_0 = data[
                (data['acs_source'] == 'BG') &
                ((data["has_first_name"] == 1) |
                 (data["has_middle_name"] == 1) |
                 (data["has_last_name"] == 1)
                
            )]
            keys_0 = list(df_0.index.values)
            
            df_1 = data[
                ~(data.index.isin(keys_0)) &
                (data['acs_source'] == 'CT') &
                ((data["has_first_name"] == 1) |
                 (data["has_middle_name"] == 1) |
                 (data["has_last_name"] == 1)
                
            )]
            keys_1 = keys_0 + list(df_1.index.values)
            
            df_2 = data[
                ~(data.index.isin(keys_1)) &
                (data['acs_source'] == 'ZIP') &
                ((data["has_first_name"] == 1) |
                 (data["has_middle_name"] == 1) |
                 (data["has_last_name"] == 1)
                
            )]
            keys_2 = keys_1 + list(df_2.index.values)
            
            # Select records for zrp geo only proxies
            df_3 = data[
                ~(data.index.isin(keys_2)) &
                (data['acs_source'] == 'BG') &
                ~((data["has_first_name"] == 1) &
                 (data["has_middle_name"] == 1) &
                 (data["has_last_name"] == 1)
            )]
            keys_3 = keys_2 + list(df_3.index.values)
            
            df_4 = data[
                ~(data.index.isin(keys_3)) &
                (data['acs_source'] == 'CT') &
                ~((data["has_first_name"] == 1) &
                 (data["has_middle_name"] == 1) &
                 (data["has_last_name"] == 1)
                
            )]
            keys_4 = keys_3 + list(df_4.index.values)
            
            df_5 = data[
                ~(data.index.isin(keys_4)) &
                (data['acs_source'] == 'ZIP') &
                ~((data["has_first_name"] == 1) &
                 (data["has_middle_name"] == 1) &
                 (data["has_last_name"] == 1)
                
            )]
            keys_5 = keys_4 + list(df_5.index.values)
            
            # Records for BISG, zrp name only, or No-proxy
            df_6 = data[~(data.index.isin(keys_5))]
            keys_6 = keys_5 + list(df_6.index.values)
            
            # Predict BISG
            # Check the indices where proxy is none
            # run ZRP name only on these left over
            # check where proxy is none
            # mark that these have source no_proxy
            
        else:
            raise KeyError("Processed data is required for ZRP_Predict. Please use EnginetoPredict if supplying feature engineered data to generate predictions" )
            
        out_list = []
        if not df_0.empty:    # BG & Names
            zrp_bg_complete = ZRP_Predict_BlockGroup(self.pipe_path, **self.params_dict)
            out_0 = zrp_bg_complete.transform(df_0.filter(flb))
            out_list.append(out_0)    
        if not df_1.empty:    # CT & Names
            zrp_ct_complete = ZRP_Predict_CensusTract(self.pipe_path, **self.params_dict)
            out_1 = zrp_ct_complete.transform(df_1.filter(flc))
            out_list.append(out_1)    
        if not df_2.empty:    # ZC & Names
            zrp_zp_complete = ZRP_Predict_ZipCode(self.pipe_path, **self.params_dict)
            out_2 = zrp_zp_complete.transform(df_2.filter(flz))
            out_list.append(out_2)    
        if not df_3.empty:    # BC Only
            zrp_bg_geo_only = ZRP_Predict_BlockGroup(self.pipe_path, **self.params_dict)
            out_3 = zrp_bg_geo_only.transform(df_3.filter(flb), geo_only=True)
            out_list.append(out_3)  
        if not df_4.empty:    # CT Only
            zrp_ct_geo_only = ZRP_Predict_CensusTract(self.pipe_path, **self.params_dict)
            out_4 = zrp_ct_geo_only.transform(df_4.filter(flc), geo_only=True)
            out_list.append(out_4)   
        if not df_5.empty:    # ZC Only
            zrp_zp_geo_only = ZRP_Predict_ZipCode(self.pipe_path, **self.params_dict)
            out_5 = zrp_zp_geo_only.transform(df_5.filter(flz), geo_only=True)
            out_list.append(out_5)  
        if not df_6.empty:    # BISG
            bisgw = BISGWrapper(**self.params_dict)
            bisgw.fit(df_6)
            out_6 = bisgw.transform(df_6)

            if self.standard_target_classes():
                # Capture successful BISG proxies 
                bisg_proxies = out_6[~(out_6[f"{self.race}_proxy"].isna())]
                out_list.append(bisg_proxies) 
                
                # Records that failed BISG proxy
                records_failed_bisg_proxy = df_6.loc[df_6.index.intersection(out_6[out_6[f"{self.race}_proxy"].isna()].index.values)]
            else:
                # For non-standard classes disregard BISG predictions 
                records_failed_bisg_proxy = df_6
            
            if not records_failed_bisg_proxy.empty:    # Attempt name only ZRP proxying
                df_7 = records_failed_bisg_proxy[
                    (records_failed_bisg_proxy['has_first_name'] == 1) |
                    (records_failed_bisg_proxy['has_middle_name'] == 1) |
                    (records_failed_bisg_proxy['has_last_name'] == 1)
                ]     
                if not df_7.empty:
                    zrp_names_only = ZRP_Predict_BlockGroup(self.pipe_path, **self.params_dict)
                    out_7 = zrp_names_only.transform(df_7.filter(flb), name_only=True)
                    out_list.append(out_7)  

                    record_indices_for_name_only_proxy = list(df_7.index.values)
                    cannot_proxy_records = records_failed_bisg_proxy[~(records_failed_bisg_proxy.index.isin(record_indices_for_name_only_proxy))]
                else:
                    cannot_proxy_records = records_failed_bisg_proxy

                if not cannot_proxy_records.empty:
                    failed_proxies = out_6.loc[out_6.index.intersection(cannot_proxy_records.index.values)]
                    failed_proxies.rename(columns={"source_bisg": "source_no_proxy"}, inplace=True)
                    out_list.append(failed_proxies)
                
        if df_6.empty or records_failed_bisg_proxy.empty or cannot_proxy_records.empty:
            print("   ...Proxies generated")

        proxies_out = pd.concat(out_list)
        
        # Rearegement of columns    
        all_source_cols = [
            'source_zrp_block_group', 'source_zrp_census_tract',
            'source_zrp_zip_code', 'source_bisg', 'source_zrp_block_group_geo_only',
            'source_zrp_census_tract_geo_only', 'source_zrp_zip_code_geo_only',
            'source_zrp_name_only', 'source_no_proxy']
        source_cols = [col for col in all_source_cols if col in proxies_out.columns]
        race_cols = list(set(proxies_out.columns) - set(source_cols) - set([f"{self.race}_proxy"]))
        race_cols.sort()
        ordered_columns = race_cols + [f"{self.race}_proxy"] + source_cols  
        proxies_out = proxies_out[ordered_columns]
        
        proxies_out[race_cols] = proxies_out[race_cols].fillna(0)
        proxies_out[source_cols] = proxies_out[source_cols].fillna(0)
        proxies_out = proxies_out.sort_values(source_cols)        
        
        if save_table:
            make_directory(self.out_path)
            if self.runname is not None:
                file_name = f'proxy_output_{self.runname}.feather'
            else:
                file_name = 'proxy_output.feather'
            save_feather(proxies_out, self.out_path, file_name)      
        
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
        src_path = os.path.join(self.pipe_path, self.pipe_type)
        model = xgboost.Booster()
        model.load_model(os.path.join(src_path,"model.txt"))
#         model = pd.read_pickle(os.path.join(self.pipe_path, f"{pipe_type}/model.pkl") )
        pipe = pd.read_pickle(os.path.join(src_path, "pipe.pkl"))
        # Load Data
        if input_data.empty:
            raise ValueError("Feature engineered data is empty or missing. Please provide the feature engineered data as `input_data` to generate predictions.")
        fe_data = input_data.copy()
        fe_data = fe_data.set_index('ZEST_KEY')
        fe_matrix = xgboost.DMatrix(fe_data)

        proxies = pd.DataFrame(model.predict(fe_matrix), index = fe_data.index)
        proxies.columns = sorted(pipe.steps[2][1].mlb_columns)
        proxies[f"{self.race}_proxy"] = proxies.idxmax(axis=1)
        proxies[f'source_{self.pipe_type}'] = 1        

        
        if save_table:
            make_directory(self.out_path)
            if self.runname is not None:
                file_name = f'proxy_output_{self.runname}.feather'
            else:
                file_name = 'proxy_output.feather'
            save_feather(proxies_out, self.out_path, file_name) 
            
        return(proxies)
    
