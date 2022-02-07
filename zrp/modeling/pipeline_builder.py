from os.path import expanduser, dirname
import pandas as pd
import numpy as np
import os
import re
import sys
import json
import joblib
import pickle 

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.selection import SmartCorrelatedSelection, DropFeatures

from zrp.modeling.src.app_preprocessor import HandleCompoundNames
from zrp.modeling.src.acs_scaler import CustomRatios
from zrp.modeling.src.app_fe import AppFeatureEngineering, NameAggregation
from zrp.modeling.src.set_key import SetKey
from zrp.prepare.utils import load_json, load_file, save_feather, make_directory
from zrp.prepare.base import BaseZRP

curpath = dirname(__file__)


class ZRP_Build_Pipeline(BaseZRP):
    """
    Fits a new ZRP pipeline from user input 
    
    Parameters
    ----------
    zrp_model_name: str
        Name of zrp_model
    zrp_model_source: str
        Indicates the source of zrp_modeling data to use. There are three optional sources 'block_group', 'census_tract', and 'zip_code'. By default 'census_tract' is inferred.
    """

    def __init__(self, zrp_model_name = 'zrp_0', zrp_model_source ='census_tract', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs_path = os.path.join(self.out_path,
                                     "experiments",
                                     self.zrp_model_name,
                                     self.zrp_model_source,
                                     "data")
    def fit(self, X, y):
        ### Build Pipeline
        print('\n---\nBuilding pipeline')

        self.pipe = Pipeline(
            [("Drop Features", DropFeatures(features_to_drop = ['GEOID_BG', 'GEOID_CT', 'GEOID_ZIP', 'ZEST_KEY_COL'])),
             ("Compound Name FE", HandleCompoundNames(last_name = self.last_name, first_name = self.first_name, middle_name = self.middle_name)),
             ("App FE", AppFeatureEngineering( key = self.key, geo_key = self.geo_key, first_name = self.first_name , middle_name = self.middle_name, last_name = self.last_name, race = self.race)),
             ("ACS FE", CustomRatios()), 
             ("Name Aggregation", NameAggregation(key = key, n_jobs = self.n_jobs)),
             ("Drop Features (2)", DropFeatures(features_to_drop = ['GEOID'])),
             ("Impute", MeanMedianImputer(imputation_method = "mean", variables=None)),
             ("Correlated Feature Selection",  SmartCorrelatedSelection(method = 'pearson',
                                                                        threshold=.95))],
            verbose=True
        )
        #### Fit the Pipeline
        print('\n---\nFitting pipeline')
        self.pipe.fit(X, y[self.race]) 

        return self
    


    def transform(self, X):
        make_directory(self.outputs_path)
        # Save pipeline
        pickle.dump(self.pipe, open( os.path.join(self.outputs_path, 'pipe.pkl'), 'wb'))
        #### Transform
        ##### This step creates the feature engineering data
        print('\n---\nTransforming FE data')
        X_train_fe = self.pipe.transform(X = X)


        # Save train fe data
        print('\n---\nSaving FE data')
        save_feather(X_train_fe, self.outputs_path, f"train_fe_data.feather")        
        return(X_train_fe)
    

    
class ZRP_Build_Model(BaseZRP):
    """
    Generate as ZRP model from input data & pre-trained pipeline. 
    
    Parameters
    ----------
    zrp_model_name: str
        Name of zrp_model
    zrp_model_source: str
        Indicates the source of zrp_modeling data to use. There are three optional sources 'block_group', 'census_tract', and 'zip_code'. By default 'census_tract' is inferred.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs_path = os.path.join(self.out_path,
                                     "experiments",
                                     self.zrp_model_name,
                                     self.zrp_model_source,
                                     "data")
    def fit(self, X, y):
        ### Build the zrp_model
        ##### specify zrp_model parameters
        print('\n---\nbuilding zrp_model')

        opt_params = {'gamma': 5,
                      'learning_rate': 0.01, 
                      'max_depth':3,
                      'min_child_weight': 500,
                      'n_estimators': 2000,
                      'subsample': 0.20}

        ##### Initialize the zrp_model
        self.zrp_model = XGBClassifier(objective='multi:softprob',
                            num_class=len(y[self.race].unique()),
                            **opt_params)  
        ##### Fit
        print('\n---\nfitting zrp_model')
        self.zrp_model.fit(
            X,  y[self.race],
                sample_weight = y.sample_weight
        )

        return self
    

    def transform(self, X):
        make_directory(self.outputs_path)
        # Save zrp_model
        pickle.dump(self.zrp_model, open(os.path.join(self.outputs_path,"zrp_model.pkl", "wb"))) 
        
        ##### Return Race Probabilities
        print('\n---\nGenerate & save race predictions (labels)')
        y_hat_train = pd.DataFrame({'race': self.zrp_model.predict(X)}, index=X.index)

        y_hat_train.reset_index(drop=False).to_feather(os.path.join(self.outputs_path, f"train_proxies.feather"))


        print('\n---\nGenerate & save race predictions (probabilities)')
        y_phat_train = pd.DataFrame(self.zrp_model.predict_proba(X), index=X.index)

        y_phat_train.columns = ["AAPI", "AIAN", "BLACK", "HISPANIC", "WHITE"]

        y_phat_train.reset_index(drop=False).to_feather(os.path.join(self.outputs_path, f"train_proxy_probs.feather"))
        
        return(y_hat_train, y_phat_train)   
    
    

class ZRP_DataSampling(BaseZRP):
    """
    Generate data splits from input data 
    
    Parameters
    ----------
    zrp_model_name: str
        Name of zrp_model
    zrp_model_source: str
        Indicates the source of zrp_modeling data to use. There are three optional sources 'block_group', 'census_tract', and 'zip_code'. By default 'census_tract' is inferred.
    """

    def __init__(self, zrp_model_name = 'zrp_0', zrp_model_source ='census_tract', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs_path = os.path.join(self.out_path,
                                     "experiments",
                                     self.zrp_model_name,
                                     self.zrp_model_source,
                                     "data")
    def fit(self):
        return self
    


    def transform(self, data):
        make_directory(self.outputs_path)
        df = data.copy()
        # Keep geocoded data & data with labels
        df = df[(df[self.geo_key].notna()) & (df[self.geo_key]!="None")]
        df = df[(df[self.race].notna()) & (df[self.race]!="None")]
        df_keys = list(df.index.unique())

        # sample weights normalizing to us population
        df["sample_weight"] = df[self.race].map(
                    {
                        "AAPI": (0.061/1.022)/df[self.race].value_counts(normalize=True).AAPI,
                        "BLACK": (0.134/1.022)/df[self.race].value_counts(normalize=True).BLACK,
                        "HISPANIC": (0.185/1.022)/df[self.race].value_counts(normalize=True).HISPANIC,
                        "AIAN": (0.013/1.022)/df[self.race].value_counts(normalize=True).AIAN,
                        "OTHER": (0.028/1.022) /df[self.race].value_counts(normalize=True).OTHER,
                        "WHITE": (0.601/1.022)/(df[self.race].value_counts(normalize=True).WHITE),
                    }
                )
        
        # Split working data
        X = df.copy()
        X.drop([race, "sample_weight"], axis=1, inplace=True)
        if key == df.index.name:
            y = df[[self.geo_key, self.race, "sample_weight"]]
        else:
            y = df[[self.key, self.geo_key, self.race, "sample_weight"]]

        # Train (80) + Test(20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)
        
        save_feather(X_train, self.outputs_path, f"X_train.feather")
        save_feather(y_train, self.outputs_path, f"y_train.feather")
        save_feather(X_test, self.outputs_path, f"X_test.feather")
        save_feather(y_test, self.outputs_path, f"y_test.feather")

        return(X_train, X_test, y_train, y_test)
    
    
    
class ZRP_Build(BaseZRP):
    """
    This class builds a new custom ZRP model trained off of user input data. Supply standard ZRP requirements including name, address, and race to build your custom model-pipeline. Race & ethnicity probablities and labels are returned from this class. The pipeline, model, and supporting data is saved automatically to "~/data/experiments/model_source/data/" in the support files path defined.
    
    Parameters
    ----------
    zrp_model_name: str
        Name of zrp_model
    zrp_model_source: str
        Indicates the source of zrp_modeling data to use. There are three optional sources 'block_group', 'census_tract', and 'zip_code'. By default 'census_tract' is inferred.
    """    

    def __init__(self, zrp_model_name = 'zrp_0', zrp_model_source ='census_tract', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs_path = os.path.join(self.out_path,
                                     "experiments",
                                     self.zrp_model_name,
                                     self.zrp_model_source,
                                     "data")
    def fit(self):
        return self 
    
    def transform(self, X, y):
        make_directory(self.outputs_path)

        sample_path = self.outputs_path

        dsamp = ZRP_DataSampling(BaseZRP)
        X_train, X_test, y_train, y_test = dsamp.transform(data)        
        cwd = os.getcwd()
        feature_list = load_json(os.path.join(cwd, f'feature_list_{zrp_model_source}.json'))
        
        y_train = y_train.drop_duplicates(self.key)
        train_keys = list(y_train[self.key].values)
        X_train = X_train[X_train[self.key].isin(train_keys)]
        X_train.drop_duplicates(self.key)

        y_train[[self.geo_key, self.key]] = y_train[[self.geo_key, self.key]].astype(str)
        sample_weights = y_train[[key, 'sample_weight']].copy() 

        assert  X_train.shape[0] == y_train.shape[0], "Unexpected mismatch between shapes. There are duplicates in the data, please remove duplicates & resubmit the data"

        #### Set Index
        X_train.set_index(self.key, inplace=True)
        y_train.set_index(self.key, inplace=True)
        sample_weights.set_index(self.key, inplace=True)
        X_train.sort_index(inplace=True)
        y_train.sort_index(inplace=True)
        sample_weights.sort_index(inplace=True)
        
        print('\n---\nSaving raw data')
        save_feather(X_train, self.outputs_path, "train_raw_data.feather" )
        save_feather(y_train, self.outputs_path, "train_raw_targets.feather" )
        
        # Build Pipeline
        build_pipe = ZRP_Build_Pipeline()
        build_pipe.fit(X_train, y_train) 
        X_train_fe = build_pipe.transform(X_train)
        
        # Build Model
        build_model = ZRP_Build_zrp_model()
        build_model.fit(X_train, y_train) 
        y_hat_train, y_phat_train = build_model.transform(X)
        
        pred_dict = {}
        pred_dict['labels'] = y_hat_train
        pred_dict['probablities'] = y_phat_train
        return(pred_dict)
